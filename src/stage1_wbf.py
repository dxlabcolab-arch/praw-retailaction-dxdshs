"""Temporal weighted fusion (WBF) of several stage-1 prediction caches.
Predictions of all models are pooled in score order; a prediction joins the cluster (at most one member per
model) whose score-weighted span it overlaps most with tIoU >= thr. Cluster span/points = score-weighted mean,
cluster score = sum of member scores / number of models. Keeps the top K clusters.
usage: python src/stage1_wbf.py OUT.pkl THR IN1.pkl IN2.pkl ..."""
import sys, pickle, numpy as np


def tiou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else 0


def wbf(sets, thr=0.5, K=10):
    M = len(sets)
    allp = sorted([(p, m) for m, P in enumerate(sets) for p in P], key=lambda t: -t[0][0])
    clusters = []
    for p, m in allp:
        best, bi = thr, None
        for ci, c in enumerate(clusters):
            if any(mm == m for _, mm in c):
                continue
            w = np.array([q[0] for q, _ in c]) + 1e-9
            fs = (np.array([q[1:3] for q, _ in c]) * w[:, None]).sum(0) / w.sum()
            t = tiou(p[1:3], fs)
            if t >= best:
                best, bi = t, ci
        if bi is None:
            clusters.append([(p, m)])
        else:
            clusters[bi].append((p, m))
    out = []
    for c in clusters:
        R = np.array([q for q, _ in c]); w = R[:, 0] + 1e-9
        out.append(np.r_[R[:, 0].sum() / M, (R[:, 1:] * w[:, None]).sum(0) / w.sum()])
    out = np.array(out)
    return out[np.argsort(-out[:, 0])][:K]


def as_array(c):
    return np.concatenate([c["score"][:, None], c["start"][:, None], c["end"][:, None], c["points"]], 1)


if __name__ == "__main__":
    out, thr, srcs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    C = [pickle.load(open(s, "rb")) for s in srcs]
    W = {}
    for n in C[0]:
        o = wbf([as_array(c[n]) for c in C], thr)
        W[n] = dict(score=o[:, 0].astype(np.float32), start=o[:, 1].astype(np.float32),
                    end=o[:, 2].astype(np.float32), points=o[:, 3:7].astype(np.float32))
    pickle.dump(W, open(out, "wb")); print("wrote", out, len(W))
