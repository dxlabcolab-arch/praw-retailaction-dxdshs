"""Apply soft-NMS to a submission CSV: a prediction whose time span overlaps (tIoU >= THR) any
higher-scored prediction of the same clip gets score x FACTOR. Coordinates and count are unchanged.
usage: python src/softnms_csv.py IN.csv OUT.csv [THR=0.1] [FACTOR=0.01]"""
import sys, json
import numpy as np, pandas as pd


def tiou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else 0


def softnms(actions, thr=0.1, factor=0.01):
    acts = sorted(actions, key=lambda a: -a["score"])
    span = [(a["start"], a["end"]) for a in acts]
    out = []
    for j, a in enumerate(acts):
        a = dict(a)
        if any(tiou(span[i], span[j]) >= thr for i in range(j)):
            a["score"] = a["score"] * factor
        out.append(a)
    return sorted(out, key=lambda a: -a["score"])


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    factor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    df = pd.read_csv(src)
    df["prediction"] = [json.dumps({"actions": softnms(json.loads(p)["actions"], thr, factor)}) for p in df["prediction"]]
    df.to_csv(dst, index=False); print(f"wrote {dst}: {len(df)} rows (soft-NMS tIoU>={thr}, x{factor})")
