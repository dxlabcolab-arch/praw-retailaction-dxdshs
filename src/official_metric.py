"""Fast, faithful re-implementation of the OFFICIAL Kaggle metric.

Source: Kaggle notebook davidemaz/map-global-metric-code (by the dataset authors),
kept verbatim in external/official_metric/map_global.py. Semantics reproduced:

  * videos with no ground-truth action are skipped entirely (their predictions
    are never counted as false positives)
  * for every (spatial, tIoU, confidence) threshold, predictions with
    score >= confidence are greedily matched in score order to the unmatched
    target with the highest tIoU; tIoU uses (union + 1e-4)
  * spatial match if ANY view is within the threshold, distance = normalized
    distance x meters-per-normalized-unit factor from pose bone lengths
    (joints kept if score > 0 or coords non-zero; nanmedian; fallback 3.07)
  * AP = sum_c (R[c+1] - R[c]) * P_env[c] over the 101 confidence levels, where
    P_env is the max precision at any lower confidence level

Greedy matching in a fixed score order means the result for a higher
confidence cut is a prefix of the full run, so each (spatial, tIoU) pair needs
one matching pass instead of 101.
"""
import numpy as np

from metric import BONE_LENGTH_MEANS

CONF = np.linspace(1, 0, 101)
TIOU = np.linspace(0.05, 0.5, 10)
SPAT = np.linspace(1, 0.05, 20)
EPS = 1e-4
M_PX_FACTOR_AVG = 3.07


def official_factor(poses):
    """Meters per normalized unit for one view, as in compute_m_px_factor_from_bones."""
    ratios = []
    for fr in poses or []:
        if fr is None or len(fr.get("pose", {})) == 0:
            continue
        sc = fr.get("joint_scores", {})
        P = {j: np.asarray(c, float) for j, c in fr["pose"].items()
             if sc.get(j, 0.0) > 0 or np.any(np.asarray(c) != 0)}
        for (a, b), m in BONE_LENGTH_MEANS.items():
            if a in P and b in P:
                L = np.sqrt(np.sum((P[a] - P[b]) ** 2))
                if L > 0:
                    ratios.append(m / L)
    if not ratios:
        return M_PX_FACTOR_AVG
    f = float(np.nanmedian(ratios))
    return M_PX_FACTOR_AVG if np.isnan(f) else f


def evaluate_official(samples, return_grid=False):
    """samples: list of dict(gt=[(s,e,x0,y0,x1,y1)], pred=[(score,s,e,x0,y0,x1,y1)], factor=(f0,f1))"""
    S, T, C = len(SPAT), len(TIOU), len(CONF)
    tp = np.zeros((S, T, C))
    fp = np.zeros((S, T, C))
    fn = np.zeros((S, T, C))
    for smp in samples:
        G = np.asarray(smp["gt"], float).reshape(-1, 6)
        if not len(G):
            continue                                   # skipped by the official metric
        P = np.asarray(smp["pred"], float).reshape(-1, 7)
        if not len(P):
            fn += len(G)
            continue
        P = P[np.argsort(-P[:, 0], kind="stable")]
        f0, f1 = smp["factor"]
        d0 = np.hypot(P[:, None, 3] - G[None, :, 2], P[:, None, 4] - G[None, :, 3]) * f0
        d1 = np.hypot(P[:, None, 5] - G[None, :, 4], P[:, None, 6] - G[None, :, 5]) * f1
        D = np.minimum(d0, d1)
        inter = np.minimum(P[:, None, 2], G[None, :, 1]) - np.maximum(P[:, None, 1], G[None, :, 0])
        union = np.maximum(P[:, None, 2], G[None, :, 1]) - np.minimum(P[:, None, 1], G[None, :, 0])
        TI = np.maximum(0, inter / (union + EPS))
        ok = (TI[None, None] >= TIOU[None, :, None, None]) & (D[None, None] <= SPAT[:, None, None, None])
        used = np.zeros((S, T, len(G)), bool)
        hit = np.zeros((S, T, len(P)), bool)
        for i in range(len(P)):
            val = np.where(ok[:, :, i] & ~used, TI[i][None, None], 0.0)
            best = val.argmax(-1)
            h = val.max(-1) > 0                        # official: tiou > best_iou (starts at 0)
            hit[:, :, i] = h
            used[h, best[h]] = True
        # predictions included at each confidence level = score >= conf
        n_in = (P[:, 0][None, :] >= CONF[:, None]).sum(1)          # (C,)
        cum = np.concatenate([np.zeros((S, T, 1)), np.cumsum(hit, -1)], -1)
        tp_c = cum[:, :, n_in]                                      # (S,T,C)
        tp += tp_c
        fp += n_in[None, None] - tp_c
        fn += len(G) - tp_c
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    prec = np.flip(np.maximum.accumulate(np.flip(prec, -1), -1), -1)
    ap = np.sum((rec[..., 1:] - rec[..., :-1]) * prec[..., :-1], -1)
    out = {"mAP": float(ap.mean()), "mAPs": float(ap[:, 0].mean()), "mAPt": float(ap[0, :].mean())}
    if return_grid:
        out["ap_grid"] = ap
    return out
