"""Run the model over a split and score it with the official-style metric."""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from metric import evaluate, m_px_factor, M_PX_FACTOR_AVG
from dataset import LABELS, VIEWS


def factors_for(sample_dir):
    meta = json.load(open(os.path.join(sample_dir, "metadata.json")))["content"]
    out = {}
    for v in VIEWS:
        cam = meta["action_cam"].get(v) or {}
        out[v] = m_px_factor(cam.get("poses"))
    gts = []
    for a in (meta.get("labels") or {}).get("action") or []:
        gts.append({
            "label": a["label"],
            "start": a["start"], "end": a["end"],
            "xy": {v: (a["spatial"]["action_cam"][v]["x"],
                       a["spatial"]["action_cam"][v]["y"]) for v in VIEWS},
        })
    return out, gts


@torch.no_grad()
def predict(model, loader, device, amp_dtype=torch.bfloat16):
    model.eval()
    preds = {}
    for batch in loader:
        clips = batch["clips"].to(device, non_blocking=True)
        poses = batch["poses"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            out = model(clips, poses)
        prob = out["logits"].float().softmax(-1)[..., :-1]     # drop "no action"
        score, cls = prob.max(-1)
        if "quality" in out:
            # the metric ranks by score alone and ignores the class, so rank by
            # predicted localization quality, gated by objectness
            objness = 1.0 - out["logits"].float().softmax(-1)[..., -1]
            score = out["quality"].float().squeeze(-1).sigmoid() * objness
        st = out["start"].float()
        en = out["end"].float()
        pts = out["points"].float()
        for b, name in enumerate(batch["names"]):
            rows = []
            for q in range(score.shape[1]):
                rows.append({
                    "label": LABELS[int(cls[b, q])],
                    "score": float(score[b, q]),
                    "start": float(st[b, q]), "end": float(en[b, q]),
                    "xy": {"rank0": (float(pts[b, q, 0]), float(pts[b, q, 1])),
                           "rank1": (float(pts[b, q, 2]), float(pts[b, q, 3]))},
                })
            preds[name] = rows
    return preds


def score_predictions(preds, root, split, cache={}):
    """Score with the OFFICIAL Kaggle metric (see official_metric.py), so that
    in-training model selection uses the same yardstick as the leaderboard."""
    from official_metric import evaluate_official, official_factor
    samples = []
    for name, rows in preds.items():
        d = os.path.join(root, split, name)
        if d not in cache:
            meta = json.load(open(os.path.join(d, "metadata.json")))["content"]
            _, gts = factors_for(d)
            fac = tuple(official_factor((meta["action_cam"].get(v) or {}).get("poses"))
                        for v in VIEWS)
            G = [(g["start"], g["end"], *g["xy"]["rank0"], *g["xy"]["rank1"]) for g in gts]
            cache[d] = (G, fac)
        G, fac = cache[d]
        P = [(r["score"], r["start"], r["end"], *r["xy"]["rank0"], *r["xy"]["rank1"]) for r in rows]
        samples.append({"gt": G, "pred": P, "factor": fac})
    return evaluate_official(samples)
