"""Run a FACT checkpoint once and cache everything hedging needs.

Per sample: score (Q), start/end (Q), final points (Q,4), and for each query and
view the top-M anchor candidates (frame x hand-joint) with the query's own offset
applied, plus their weights.
"""
import argparse, os, pickle, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from torch.utils.data import DataLoader
from dataset import RetailActionDataset, collate
from rescore import load

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--root", required=True)
p.add_argument("--split", required=True)
p.add_argument("--out", required=True)
p.add_argument("--M", type=int, default=8)
p.add_argument("--flip", action="store_true", help="horizontal-flip TTA pass; outputs are un-flipped")
a_ = p.parse_args()

m, a, _ = load(a_.ckpt)
ds = RetailActionDataset(a_.root, a_.split, a["frames"], a["img"], train=False)
dl = DataLoader(ds, batch_size=16, num_workers=12, collate_fn=collate)
cache = {}
with torch.no_grad():
    for b in dl:
        clips, poses = b["clips"].cuda(), b["poses"].cuda()
        if a_.flip:
            clips = clips.flip(4)                                  # (B,V,T,H,W,3): mirror W
            poses = poses.clone()
            poses[..., 0] = torch.where(poses[..., 2] > 0, 1 - poses[..., 0], poses[..., 0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(clips, poses)
        if a_.flip:                                                # map predictions back
            o["points"] = o["points"].clone()
            o["points"][..., 0::2] = 1 - o["points"][..., 0::2]
        prob = o["logits"].float().softmax(-1)
        score = prob[..., :-1].max(-1).values                      # (B,Q)
        w = m._anchor_w.float()                                    # (B,Q,V,T,K)
        hands = m._hands.float()                                   # (B,V,T,K,2)
        off = m._offset.float()                                    # (B,Q,V,2)
        B, Q, V, T, K = w.shape
        wf = w.flatten(3)                                          # (B,Q,V,T*K)
        topw, topi = wf.topk(a_.M, -1)                             # (B,Q,V,M)
        hf = hands.flatten(2, 3)                                   # (B,V,T*K,2)
        cand = torch.stack([torch.stack([hf[bb, v][topi[bb, :, v]] for v in range(V)], 1)
                            for bb in range(B)])                   # (B,Q,V,M,2)
        cand = (cand + off[:, :, :, None]).clamp(0, 1)
        if a_.flip:
            cand[..., 0] = 1 - cand[..., 0]
        for i, n in enumerate(b["names"]):
            extra = {}
            if "frame_logits" in o:    # FACT frame branch: per-frame P(interaction), last block
                extra["frame_prob"] = o["frame_logits"][i, -1].float().softmax(-1)[:, 1].cpu().numpy()
                extra["f2a"] = o["f2a_logit"][i].float().softmax(-1).cpu().numpy()   # (Q,T) query->frame attention
            cache[n] = dict(**extra, score=score[i].cpu().numpy(), start=o["start"][i].float().cpu().numpy(),
                            end=o["end"][i].float().cpu().numpy(), points=o["points"][i].float().cpu().numpy(),
                            cand=cand[i].cpu().numpy(), cand_w=topw[i].cpu().numpy())
pickle.dump(cache, open(a_.out, "wb"))
print(f"{a_.split}: cached {len(cache)} samples -> {a_.out}")
