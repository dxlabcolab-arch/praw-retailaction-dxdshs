"""Official-style evaluation metric for RetailAction (per ICCV25 paper, Sec. 7).

Matching: IoU_t > tau_t AND min(d_1, d_2) < tau_s, where d_i is the Euclidean
distance in METERS in view i, obtained by scaling normalized-image distance with
a per-video pixels-to-meters factor derived from 2D pose bone lengths.
"""
import numpy as np

BONE_LENGTH_MEANS = {
    ("neck", "nose"): 0.19354,
    ("left_shoulder", "left_elbow"): 0.27096,
    ("left_elbow", "left_wrist"): 0.21228,
    ("right_shoulder", "right_elbow"): 0.27210,
    ("right_elbow", "right_wrist"): 0.21316,
    ("left_hip", "left_knee"): 0.39204,
    ("left_knee", "left_ankle"): 0.39530,
    ("right_hip", "right_knee"): 0.39266,
    ("right_knee", "right_ankle"): 0.39322,
    ("left_shoulder", "right_shoulder"): 0.35484,
    ("left_hip", "right_hip"): 0.17150,
    ("neck", "left_shoulder"): 0.18136,
    ("neck", "right_shoulder"): 0.18081,
    ("left_shoulder", "left_hip"): 0.51375,
    ("right_shoulder", "right_hip"): 0.51226,
}
# meters per 1000 pixels -> meters per pixel
M_PX_FACTOR_AVG = 3.07 / 1000.0

# joint-score gate: ignore very low-confidence joints when measuring bones
MIN_JOINT_SCORE = 0.3

# Official threshold grid (Kaggle dataset description):
#   temporal IoU : 10 values from 0.05 to 0.5
#   spatial (m)  : 20 values from 1.0 down to 0.05
# Note the temporal range stops at 0.5 - high-IoU columns do not exist - while
# the spatial range reaches 0.05 m, finer than the 1/32 pose grid (~0.10 m).
TEMPORAL_THRESHOLDS = list(np.linspace(0.05, 0.5, 10))
SPATIAL_THRESHOLDS = list(np.linspace(1.0, 0.05, 20))


def m_px_factor(poses, img_size=600, min_score=MIN_JOINT_SCORE):
    """Meters-per-normalized-unit factor for one camera view.

    `poses` is the per-frame pose list from metadata. Returns a scalar to
    multiply a NORMALIZED-image distance by, to obtain meters.
    """
    ratios = []
    for p in poses or []:
        if not p or not p.get("pose"):
            continue
        pose = p["pose"]
        scores = p.get("joint_scores") or {}
        for (a, b), mean_len in BONE_LENGTH_MEANS.items():
            if a not in pose or b not in pose:
                continue
            if scores.get(a, 1.0) < min_score or scores.get(b, 1.0) < min_score:
                continue
            ax, ay = pose[a]
            bx, by = pose[b]
            # normalized -> pixels
            px = np.hypot((ax - bx) * img_size, (ay - by) * img_size)
            if px < 1e-3:
                continue
            ratios.append(mean_len / px)  # meters per pixel
    if not ratios:
        return M_PX_FACTOR_AVG * img_size  # meters per normalized unit
    return float(np.median(ratios)) * img_size


def temporal_iou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], a[0]) - min(a[0], b[0]) if False else (max(a[1], b[1]) - min(a[0], b[0]))
    if union <= 0:
        return 0.0
    return inter / union


def _ap_from_pr(scores, tps, n_gt):
    """COCO-style AP with Pascal-VOC monotone envelope."""
    if n_gt == 0:
        return np.nan
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # monotone envelope
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    # 101-point interpolation (COCO)
    q = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, q, side="left")
    out = np.zeros_like(q)
    valid = idx < len(prec)
    out[valid] = prec[idx[valid]]
    return float(out.mean())


def evaluate(samples, spatial_thresholds=None, temporal_thresholds=None):
    """samples: list of dicts with keys
         'gt':   list of {label, start, end, xy: {rank0:(x,y), rank1:(x,y)}}
         'pred': list of {label, score, start, end, xy: {...}}
         'factor': {rank0: meters_per_normalized_unit, rank1: ...}
    Returns dict with mAP, mAPs, mAPt and the full AP grid.
    """
    if spatial_thresholds is None:
        spatial_thresholds = SPATIAL_THRESHOLDS
    if temporal_thresholds is None:
        temporal_thresholds = TEMPORAL_THRESHOLDS

    # Precompute per-sample pred/gt distance & iou matrices
    prepared = []
    n_gt_total = 0
    for s in samples:
        gts, preds = s["gt"], s["pred"]
        n_gt_total += len(gts)
        if not preds:
            prepared.append((gts, preds, None, None))
            continue
        D = np.full((len(preds), len(gts)), np.inf)
        T = np.zeros((len(preds), len(gts)))
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                ds = []
                for view in ("rank0", "rank1"):
                    if view not in p["xy"] or view not in g["xy"]:
                        continue
                    px, py = p["xy"][view]
                    gx, gy = g["xy"][view]
                    f = s["factor"].get(view, M_PX_FACTOR_AVG * 600)
                    ds.append(np.hypot(px - gx, py - gy) * f)
                D[pi, gi] = min(ds) if ds else np.inf
                T[pi, gi] = temporal_iou((p["start"], p["end"]), (g["start"], g["end"]))
        prepared.append((gts, preds, D, T))

    ap_grid = np.zeros((len(spatial_thresholds), len(temporal_thresholds)))
    for si, ts in enumerate(spatial_thresholds):
        for ti, tt in enumerate(temporal_thresholds):
            scores, tps = [], []
            for gts, preds, D, T in prepared:
                if not preds:
                    continue
                order = np.argsort([-p["score"] for p in preds])
                used = set()
                for pi in order:
                    p = preds[pi]
                    best, best_g = -1.0, -1
                    for gi in range(len(gts)):
                        if gi in used:
                            continue
                        if T[pi, gi] >= tt and D[pi, gi] <= ts:
                            if T[pi, gi] > best:
                                best, best_g = T[pi, gi], gi
                    scores.append(p["score"])
                    if best_g >= 0:
                        used.add(best_g)
                        tps.append(1.0)
                    else:
                        tps.append(0.0)
            ap_grid[si, ti] = _ap_from_pr(scores, tps, n_gt_total)

    # ap_grid is indexed [spatial, temporal]; spatial_thresholds descend 1.0 -> 0.05
    # so index 0 is the loosest distance and index 0 of temporal is the loosest IoU.
    return {
        "mAP": float(np.nanmean(ap_grid)),
        # mAPs: spatial accuracy with temporal alignment effectively ignored
        # (loosest temporal IoU column)
        "mAPs": float(np.nanmean(ap_grid[:, 0])),
        # mAPt: temporal accuracy with spatial precision effectively ignored
        # (loosest distance row)
        "mAPt": float(np.nanmean(ap_grid[0, :])),
        "ap_grid": ap_grid,
        "spatial_thresholds": spatial_thresholds,
        "temporal_thresholds": temporal_thresholds,
        "n_gt": n_gt_total,
    }
