"""Compare two submission CSVs the way the metric sees them: per clip, match actions by content
(start, end, points) instead of by row position, so that a swapped order of two near-tied actions
does not look like a 0.7 coordinate error.  usage: python compare_submissions.py A.csv B.csv"""
import sys, json, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
A = pd.read_csv(sys.argv[1]).set_index("id"); B = pd.read_csv(sys.argv[2]).set_index("id")
A.index = A.index.astype(int); B.index = B.index.astype(int)
assert set(A.index) == set(B.index), f"different clip ids ({len(A)} vs {len(B)})"
def vec(p):
    acts = json.loads(p)["actions"]
    return np.array([[a["start"], a["end"]] + [a["spatial"]["action_cam"][v][k] for v in ("rank0", "rank1") for k in ("x", "y")] for a in acts]), np.array([a["confidence"] if "confidence" in a else a.get("score", 0) for a in acts])
pos_max = match_max = 0.0; swaps = 0; worst = None; n_act = 0; score_max = 0.0
for i in A.index:
    va, sa = vec(A.loc[i, "prediction"]); vb, sb = vec(B.loc[i, "prediction"])
    assert len(va) == len(vb), f"clip {i}: {len(va)} vs {len(vb)} actions"
    n_act += len(va)
    pos_max = max(pos_max, np.abs(va - vb).max())
    D = np.abs(va[:, None, :] - vb[None, :, :]).max(-1)          # (na, nb) max-abs distance
    r, c = linear_sum_assignment(D)
    m = D[r, c].max(); match_max = max(match_max, m)
    if m > (worst or (0, 0))[0]: worst = (m, i)
    swaps += int((c != np.arange(len(c))).sum() // 2)
    score_max = max(score_max, np.abs(sa[r] - sb[c]).max())
print(f"clips {len(A)} | actions {n_act}")
print(f"max abs diff, row-by-row (position)     : {pos_max:.6f}")
print(f"max abs diff after matching by content  : {match_max:.6f}   (worst clip {worst[1]})")
print(f"max abs confidence diff of matched pairs: {score_max:.6f}")
print(f"action pairs whose rank order is swapped: {swaps}")
