"""Score checkpoints under the official metric on a labelled split (validation or HF test)."""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from torch.utils.data import DataLoader
from dataset import RetailActionDataset, collate
from backbone import build_backbone
from evaluate import predict, score_predictions


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu")
    a = ck.get("args") or json.load(open(os.path.join(os.path.dirname(ckpt), "config.json")))
    if a["backbone"] in ("videomae_ssv2", "vjepa2_l", "vjepa2_g"):
        from backbone_hf import build_hf_backbone
        bb, fd, spt = build_hf_backbone(a["backbone"], a["frames"])
    elif a["backbone"] == "mvit_v2_s":
        from backbone_mvit import build_mvit_backbone
        bb, fd, spt = build_mvit_backbone(False, a["img"], a["frames"])
    elif a["backbone"].startswith("swin3d"):
        from backbone_video import build_video_backbone
        bb, fd, spt, _ = build_video_backbone(a["backbone"], False, a["img"], a["frames"],
                                              hires=not a.get("no_hires", False))
    else:
        bb, fd, spt = build_backbone(a["backbone"], False, a["img"])
    kind = a.get("model", "v1")
    if kind == "fact":
        from model_fact import MultiViewPoseDETRFACT as Net
    elif kind == "plain":
        from model_plain import PlainMultiViewDETR as Net
    elif kind == "v2":
        from model_v2 import MultiViewPoseDETR as Net
    else:
        from model import MultiViewPoseDETR as Net
    m = Net(bb, fd, num_queries=a["queries"], num_frames=a["frames"], tokens_per_frame=spt)
    m.load_state_dict(ck["model"])
    return m.cuda().eval(), a, ck.get("epoch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--root", default="data/RetailAction/data")
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--out", default="runs/rescore_official.jsonl")
    args = p.parse_args()
    for ck in args.ckpts:
        m, a, ep = load(ck)
        for split in args.splits.split(","):
            ds = RetailActionDataset(args.root, split, a["frames"], a["img"], train=False)
            dl = DataLoader(ds, batch_size=16, num_workers=12, collate_fn=collate)
            r = score_predictions(predict(m, dl, "cuda"), args.root, split, cache={})
            rec = dict(ckpt=ck, epoch=ep, split=split, n=len(ds),
                       mAP=r["mAP"], mAPs=r["mAPs"], mAPt=r["mAPt"])
            print("%-34s ep%-3s %-10s mAP %6.2f  mAPs %6.2f  mAPt %6.2f" % (
                ck, ep, split, r["mAP"] * 100, r["mAPs"] * 100, r["mAPt"] * 100), flush=True)
            open(args.out, "a").write(json.dumps(rec) + "\n")
