"""Stage-1 cache -> N refinement passes -> Kaggle CSV (and optional scoring on a labelled split)."""
import argparse, json, os, pickle, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from refine import Refiner, refine_cache, score_cache

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--refiner", default="runs/refine_r50/best.pt")
ap.add_argument("--topk", type=int, default=3)
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--out", default="")
ap.add_argument("--score", action="store_true")
ap.add_argument("--bs", type=int, default=64, help="refiner windows per forward pass")
ap.add_argument("--no-time", action="store_true", help="ignore a refiner's time head (keep stage-1 spans)")
ap.add_argument("--save-cache", default="", help="also pickle the refined cache here")
ap.add_argument("--shift", type=float, default=0.0, help="shift TTA: +-d window-centre offsets")
ap.add_argument("--flip", action="store_true", help="mirror TTA in the refiner")
ap.add_argument("--alpha", type=float, default=0.0, help="re-rank by refiner confidence (0 = off)")
a = ap.parse_args()

import refine
m = []
for path in a.refiner.split(","):                   # several refiners -> their points are averaged
    ck = torch.load(path, map_location="cpu")
    refine.PATCH = ck["args"].get("patch", 128)      # window resolution the refiner was trained at
    refine.T = ck["args"].get("frames", 16)
    refine.HALF = ck["args"].get("half", 0.15)
    refine.TZOOM = ck["args"].get("tzoom", 0.0)                 # frames around each prediction's span
    refine.USE_TIME = ck["args"].get("w_time", 0.0) > 0 and not a.no_time
    r = Refiner(ck["args"]["backbone"], conf=ck["args"].get("conf", False),
                fpn=ck["args"].get("fpn", False),
                levels=tuple(int(v) for v in ck["args"]["levels"].split(",")) if ck["args"].get("levels") else None,
                time_head=ck["args"].get("w_time", 0.0) > 0,
                temporal_layers=ck["args"].get("temporal_layers", 0)).cuda().eval()
    r.load_state_dict(ck["model"])
    r.patch = refine.PATCH                           # each refiner crops at its own resolution
    m.append(r)
d = a.shift
shifts = [(0.0, 0.0)] + ([(d, 0), (-d, 0), (0, d), (0, -d)] if d > 0 else [])
p = pickle.load(open(a.cache, "rb"))
for _ in range(a.passes):
    p = refine_cache(m, p, a.root, a.split, a.topk, bs=a.bs, flip=a.flip, shifts=shifts)
if a.save_cache:
    pickle.dump(p, open(a.save_cache, "wb"))
if a.alpha > 0:
    from refine import rescore
    p = rescore(p, a.alpha)
if a.score:
    r = score_cache(p, a.root, a.split)
    print(f"{a.split}: mAP {r['mAP']*100:.2f} mAPs {r['mAPs']*100:.2f} mAPt {r['mAPt']*100:.2f}", flush=True)
if a.out:
    rows = []
    for n, c in p.items():
        o = np.argsort(-c["score"])
        rows.append({"id": int(n), "prediction": json.dumps({"actions": [
            {"start": float(c["start"][q]), "end": float(c["end"][q]),
             "spatial": {"action_cam": {
                 "rank0": {"x": float(np.clip(c["points"][q, 0], 0, 1)), "y": float(np.clip(c["points"][q, 1], 0, 1))},
                 "rank1": {"x": float(np.clip(c["points"][q, 2], 0, 1)), "y": float(np.clip(c["points"][q, 3], 0, 1))}}},
             "score": float(c["score"][q])} for q in o]})})
    pd.DataFrame(rows).sort_values("id").to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(rows)} rows", flush=True)
