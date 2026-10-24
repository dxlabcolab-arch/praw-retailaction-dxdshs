"""Stage-2 point refinement.

Stage 1 (pose-anchored DETR + FACT) gets the contact point to a median of
~0.125 m, and the official metric loses most of its points below 0.2 m. Stage 1
never looks at full-resolution pixels near its answer: its backbone grid is
7x7 over a 224 px resize (~0.46 m per cell). This stage does.

For one view of one clip it takes a ~1 m window around a coarse point, cut from
the original 600x600 frames at 16 time steps (cameras are static, so the window
shows the hand entering and leaving), plus a hand-joint channel and a per-frame
"inside the predicted span" flag, and predicts the contact point in the window
with a heatmap + soft-argmax. The loss is the distance in METERS, the unit the
metric thresholds are in.

Training windows are centred on the ground truth plus noise shaped like stage-1
error, so stage 1 need not be run on the training set.
"""
import argparse, json, math, os, pickle, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import read_video, pose_to_array, JOINT_NAMES, HAND_JOINTS, VIEWS
from official_metric import official_factor

cv2.setNumThreads(0)
HAND_IDX = [JOINT_NAMES.index(j) for j in HAND_JOINTS]
HALF = 0.15          # window half-size, normalized (~0.96 m across at the median scale); --half
CE_W = 0.05          # heatmap CE weight; --ce
PATCH = 128          # window resolution fed to the CNN
T = 16               # time steps (overridable: --frames)

# Radial error of the current stage-1 (V-JEPA2) top-3 points on validation, per view,
# as quantiles at 0, 25, 50, 75, 90, 95, 100 %. "--noise empirical" samples window
# offsets from this so the refiner trains on the displacements it will actually see.
EMP_R = np.interp(np.linspace(0, 1, 41), [0, .25, .5, .75, .9, .95, 1.0],
                  [0, .030, .048, .073, .106, .135, .20])


def crop_window(frames, cx, cy):
    """frames (T,H,W,3) uint8 -> (T,PATCH,PATCH,3); window centred at normalized (cx,cy)."""
    Tn, H, W, _ = frames.shape
    hw, hh = HALF * W, HALF * H
    M = np.float32([[PATCH / (2 * hw), 0, -(cx * W - hw) * PATCH / (2 * hw)],
                    [0, PATCH / (2 * hh), -(cy * H - hh) * PATCH / (2 * hh)]])
    return np.stack([cv2.warpAffine(f, M, (PATCH, PATCH), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT) for f in frames])


def hand_channel(pose, cx, cy, sigma=3.0):
    """pose (T,J,3) normalized -> (T,PATCH,PATCH) gaussian blobs at visible hand joints."""
    out = np.zeros((len(pose), PATCH, PATCH), np.float32)
    yy, xx = np.mgrid[0:PATCH, 0:PATCH].astype(np.float32)
    for t in range(len(pose)):
        for j in HAND_IDX:
            x, y, s = pose[t, j]
            if s <= 0:
                continue
            u = (x - (cx - HALF)) / (2 * HALF) * PATCH
            v = (y - (cy - HALF)) / (2 * HALF) * PATCH
            if -3 * sigma < u < PATCH + 3 * sigma and -3 * sigma < v < PATCH + 3 * sigma:
                out[t] = np.maximum(out[t], np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2)))
    return out


def read_all_frames(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f[:, :, ::-1])
    cap.release()
    return np.stack(fr) if fr else None


def zoom_index(span, n, Tn, margin):
    """Tn frame indices spread over [start - m, end + m] (m = margin x span length, >= 2 frames), clipped."""
    s, e = float(min(span)), float(max(span))
    m = max(margin * (e - s), 2.0 / n)
    lo, hi = max(0.0, s - m), min(1.0, e + m)
    if hi - lo < Tn / n * 0.5:                          # very short clip/span: fall back to the whole clip
        lo, hi = 0.0, 1.0
    t = lo + (np.arange(Tn) + 0.5) / Tn * (hi - lo)
    return np.clip((t * n).astype(np.int64), 0, n - 1)


def build_input(frames, pose, ftimes, cx, cy, span):
    rgb = crop_window(frames, cx, cy)                                  # (T,P,P,3)
    hand = hand_channel(pose, cx, cy)                                  # (T,P,P)
    x = np.concatenate([rgb.astype(np.float32) / 255.0, hand[..., None]], -1)
    flag = ((ftimes >= span[0]) & (ftimes <= span[1])).astype(np.int64)
    return x.transpose(0, 3, 1, 2), flag                               # (T,4,P,P), (T,)


class RefineTrainSet(Dataset):
    """One (view, action) per item; window centre = GT + stage-1-like noise."""

    def __init__(self, root, split, train=True, neg_p=0.0, noise="mix", rev_p=0.0, strong=False, fold=-1, nfolds=0):
        self.items = []
        for sp in split.split(","):                       # e.g. "train,test" to pool splits
            for d in sorted(os.listdir(os.path.join(root, sp))):
                p = os.path.join(root, sp, d)
                if os.path.isdir(p):
                    self.items.append(p)
        if nfolds > 0:          # k-fold: drop the clips of fold `fold` (stable split by clip folder name)
            import hashlib
            self.items = [p for p in self.items
                          if int(hashlib.md5(os.path.basename(p).encode()).hexdigest(), 16) % nfolds != fold]
        self.train = train
        self.neg_p = neg_p
        self.noise = noise
        self.rev_p = rev_p      # time reversal: the contact point does not depend on time direction
        self.strong = strong    # 90-degree rotations + colour jitter (top-down view has no "up")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        rng = np.random.default_rng(None if self.train else i)
        meta = json.load(open(os.path.join(d, "metadata.json")))["content"]
        acts = (meta.get("labels") or {}).get("action") or []
        if not acts:
            return None
        a = acts[rng.integers(len(acts))]
        v = VIEWS[rng.integers(2)] if self.train else VIEWS[i % 2]
        span = np.array([a["start"], a["end"]]) + (rng.normal(0, 0.04, 2) if self.train else 0)
        if TZOOM > 0:
            allf = read_all_frames(os.path.join(d, f"{v}_video.mp4"))
            if allf is None:
                return None
            n = len(allf); idx = zoom_index(span, n, T, TZOOM); frames = allf[idx]
        else:
            frames, idx, n = read_video(os.path.join(d, f"{v}_video.mp4"), T, self.train, rng)
            if frames is None:
                return None
        cam = meta["action_cam"].get(v) or {}
        pose = pose_to_array(cam.get("poses"), idx, T)
        fac = official_factor(cam.get("poses"))
        gx, gy = a["spatial"]["action_cam"][v]["x"], a["spatial"]["action_cam"][v]["y"]
        # stage-1-like error: mostly a few cm, sometimes much larger; keep GT inside the window
        neg = (rng.random() < self.neg_p) if self.train else (i % 4 == 0)
        if neg:   # window that does NOT contain the contact point: teaches the confidence head
            ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(1.4, 3.0) * HALF
            cx, cy = gx + r * np.cos(ang), gy + r * np.sin(ang)
        elif self.noise == "empirical":
            r = float(rng.choice(EMP_R)) * (1.0 if rng.random() < 0.7 else 0.3)   # 30%: later passes
            ang = rng.uniform(0, 2 * np.pi)
            noise = np.clip([r * np.cos(ang), r * np.sin(ang)], -0.8 * HALF, 0.8 * HALF)
            cx, cy = gx + noise[0], gy + noise[1]
        else:
            sd = 0.02 if rng.random() < 0.6 else 0.06
            noise = np.clip(rng.normal(0, sd, 2), -0.8 * HALF, 0.8 * HALF)
            cx, cy = gx + noise[0], gy + noise[1]
        x, flag = build_input(frames, pose, idx / n, cx, cy, span)
        u = (gx - (cx - HALF)) / (2 * HALF)
        vv = (gy - (cy - HALF)) / (2 * HALF)
        if self.train and rng.random() < 0.5:                              # mirror the window
            x = x[..., ::-1].copy()
            u = 1 - u
        if self.train and self.rev_p > 0 and rng.random() < self.rev_p:    # play the window backwards
            x = x[::-1].copy()                                              # (point unchanged)
            flag = flag[::-1].copy()
        if self.train and self.strong:
            k = int(rng.integers(4))                                        # rotate window (H,W axes)
            if k:
                x = np.rot90(x, k, axes=(2, 3)).copy()
                for _ in range(k):                                          # np.rot90: (u,v) -> (v, 1-u)
                    u, vv = vv, 1 - u
            rgb = x[:, :3]                                                  # brightness/contrast/per-channel gain
            mean = rgb.mean()
            rgb = (rgb - mean) * rng.uniform(0.75, 1.25) + mean * rng.uniform(0.75, 1.25)
            rgb = rgb * rng.uniform(0.9, 1.1, (1, 3, 1, 1))
            x = np.concatenate([np.clip(rgb, 0, 1), x[:, 3:]], 1).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(flag),
                torch.tensor([u, vv], dtype=torch.float32),
                torch.tensor(fac * 2 * HALF, dtype=torch.float32),      # meters per window unit
                torch.tensor(0.0 if neg else 1.0),
                torch.from_numpy((idx / n).astype(np.float32)),          # time of each input frame
                torch.tensor([a["start"], a["end"]], dtype=torch.float32))


def collate_skip(batch):
    batch = [b for b in batch if b is not None]
    return [torch.stack(z) for z in zip(*batch)]


class VitPoseFeatures(nn.Module):
    """ViTPose(++) backbone from transformers as a timm-like `features_only` net.
    Patch embedding gets a 4th (hand) input channel initialised to zero; the 16x12 position grid of the
    256x192 pose model is resized to the square window grid; ViTPose++ uses the COCO-WholeBody expert."""

    class _Info:
        def __init__(self, ch): self.ch = ch
        def channels(self): return self.ch

    def __init__(self, repo, layers, size, in_chans=4):
        super().__init__()
        from transformers import VitPoseForPoseEstimation
        bb = VitPoseForPoseEstimation.from_pretrained(repo).backbone
        cfg = bb.config
        C, p = cfg.hidden_size, cfg.patch_size[0] if isinstance(cfg.patch_size, (list, tuple)) else cfg.patch_size
        old = bb.embeddings.patch_embeddings.projection
        self.proj = nn.Conv2d(in_chans, C, p, p)
        with torch.no_grad():
            self.proj.weight.zero_(); self.proj.weight[:, :3] = old.weight; self.proj.bias.copy_(old.bias)
        pe = bb.embeddings.position_embeddings.data                      # (1, 1+gh*gw, C)
        gh, gw = cfg.image_size[0] // p, cfg.image_size[1] // p
        self.g = size // p
        grid = pe[:, 1:].reshape(1, gh, gw, C).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(self.g, self.g), mode="bicubic", align_corners=False)
        self.pos = nn.Parameter(grid.permute(0, 2, 3, 1).reshape(1, -1, C) + pe[:, :1])
        self.encoder, self.norm = bb.encoder, bb.layernorm
        self.layers = list(layers)
        n_exp = getattr(cfg, "num_experts", 1)
        self.expert = n_exp - 1 if n_exp > 1 else None                  # last expert = COCO-WholeBody
        self.feature_info = self._Info([C] * len(self.layers))

    def forward(self, x):
        h = self.proj(x).flatten(2).transpose(1, 2) + self.pos
        idx = (torch.full((x.shape[0],), self.expert, dtype=torch.long, device=x.device)
               if self.expert is not None else None)
        hs = self.encoder(h, dataset_index=idx, output_hidden_states=True).hidden_states
        out = []
        for l in self.layers:                                            # hidden_states[0] = embeddings
            t = self.norm(hs[l + 1])
            out.append(t.transpose(1, 2).reshape(x.shape[0], -1, self.g, self.g))
        return out


class Refiner(nn.Module):
    def __init__(self, name="resnet18.a1_in1k", conf=False, fpn=False, levels=None, drop_path=0.0, time_head=False,
                 temporal_layers=0):
        super().__init__()
        import timm
        self.fpn = fpn
        self.levels = levels
        if levels:   # e.g. (0,1,2) = stride 4/8/16: every level is resized to the finest one and fused
            if name.startswith("usyd-community/"):                       # ViTPose(++) from transformers
                self.net = VitPoseFeatures(name, levels, PATCH)
            elif name.startswith("vjepa21-"):                              # V-JEPA 2.1 video backbone (refine_vjepa.py)
                from refine_vjepa import VJepa21Features
                self.net = VJepa21Features(levels, T, name=name)
            else:
                kw = dict(img_size=PATCH) if name.startswith("vit") else {}   # ViT: window res = model res
                if drop_path > 0:
                    kw["drop_path_rate"] = drop_path
                self.net = timm.create_model(name, pretrained=True, features_only=True,
                                             out_indices=tuple(levels), in_chans=4, **kw)
            chans = self.net.feature_info.channels()
            C = max(chans[1] if len(chans) > 1 else chans[0], 128)
            self.fuse = nn.Sequential(nn.Conv2d(sum(chans), C, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True))
        else:
            self.net = timm.create_model(name, pretrained=True, features_only=True,
                                         out_indices=(2, 3) if fpn else (2,), in_chans=4)  # stride 8 (+16)
            chans = self.net.feature_info.channels()
            C = chans[0]
        if fpn and not levels:   # bring the deeper, more semantic map up to stride 8 and fuse
            self.fuse = nn.Sequential(nn.Conv2d(chans[0] + chans[1], C, 1), nn.ReLU(inplace=True))
        self.span_emb = nn.Embedding(2, C)
        # optional: temporal self-attention across the frames at every spatial location of the feature map,
        # so the refiner sees motion (hand approaching / stopping). Gated residual starting at 0 = the
        # pretrained per-frame refiner at initialisation.
        self.tmix = None
        if temporal_layers > 0:
            layer = nn.TransformerEncoderLayer(C, 8, 2 * C, dropout=0.1, batch_first=True, norm_first=True)
            self.tmix = nn.TransformerEncoder(layer, temporal_layers)
            self.tpos = nn.Parameter(torch.zeros(1, 64, C)); nn.init.trunc_normal_(self.tpos, std=0.02)
            self.tflag = nn.Embedding(2, C)
            self.tgate = nn.Parameter(torch.zeros(1))
        # optional: start/end of the action as a soft-argmax over the input frames' times
        self.tdec = nn.Sequential(nn.Conv1d(C, C, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv1d(C, 2, 3, padding=1)) if time_head else None
        self._times = None
        self.t_score = nn.Linear(C, 1)
        self.head = nn.Sequential(nn.Conv2d(2 * C, C, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(C, 1, 1))
        # P(refined point is within 0.1 m of a real contact) - used to re-rank predictions
        self.conf = nn.Sequential(nn.Linear(2 * C + 1, C), nn.ReLU(inplace=True),
                                  nn.Linear(C, 1)) if conf else None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406, 0.0]).view(1, 4, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225, 1.0]).view(1, 4, 1, 1))

    def forward(self, x, flag, ft=None):
        """x (B,T,4,P,P), flag (B,T) -> (u,v) in [0,1], heatmap logits (B,H,W).
        With a time head and frame times ft (B,T), the predicted (start, end) is left in self._times."""
        B, Tn = x.shape[:2]
        feats = self.net((x.flatten(0, 1) - self.mean) / self.std)
        if self.levels:
            hw = feats[0].shape[-2:]
            f = self.fuse(torch.cat([feats[0]] + [F.interpolate(t, size=hw, mode="bilinear",
                                                                align_corners=False) for t in feats[1:]], 1))
        elif self.fpn:
            f2, f3 = feats
            f = self.fuse(torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                                                        align_corners=False)], 1))
        else:
            f = feats[-1]                                                 # (B*T,C,h,w)
        C, h, w = f.shape[1:]
        f = f.view(B, Tn, C, h, w)
        if self.tmix is not None:
            z = f.permute(0, 3, 4, 1, 2).reshape(B * h * w, Tn, C)              # one sequence per location
            z = z + self.tpos[:, :Tn] + self.tflag(flag).repeat_interleave(h * w, 0)
            z = self.tmix(z) - z
            f = f + self.tgate * z.view(B, h, w, Tn, C).permute(0, 3, 4, 1, 2)
        g = f.mean((3, 4)) + self.span_emb(flag)                        # (B,T,C)
        if self.tdec is not None and ft is not None:
            tl = self.tdec(g.transpose(1, 2).float())                   # (B,2,T)
            self._times = (tl.softmax(-1) * ft[:, None].float()).sum(-1)   # (B,2) start, end
        a = self.t_score(g).squeeze(-1).softmax(1)                      # which frames matter
        fused = torch.cat([(a[..., None, None, None] * f).sum(1), f.max(1).values], 1)
        logit = self.head(fused).squeeze(1)                             # (B,h,w)
        P2 = x.shape[-1] // 2                                           # heatmap at half the window res
        logit = F.interpolate(logit[:, None], size=(P2, P2), mode="bilinear",
                              align_corners=False)[:, 0]                # (B,P/2,P/2)
        H, W = logit.shape[1:]
        p = logit.flatten(1).softmax(-1).view(B, H, W)
        ys = (torch.arange(H, device=x.device) + 0.5) / H
        xs = (torch.arange(W, device=x.device) + 0.5) / W
        uv = torch.stack([(p.sum(1) * xs).sum(-1), (p.sum(2) * ys).sum(-1)], -1)
        conf = None
        if self.conf is not None:
            feat = torch.cat([fused.mean((2, 3)), p.flatten(1).max(-1, keepdim=True).values * H * W / 100], -1)
            conf = self.conf(feat.float()).squeeze(-1)
        return uv, logit, conf


HIT_W, GAUSS_M = 0.0, 0.0      # set from --w-hit / --gauss-sigma
RS_W, RS_SIGMA, RS_DELTA = 0.0, 0.02, 0.5   # --w-rs / --rs-sigma / --rs-delta  (Spatial Rank & Sort loss)


class SpatialRankSort(torch.autograd.Function):
    """Spatial Rank & Sort loss (RSPose, Keles et al. 2025; RS loss of Oksuz et al. 2021) on one heatmap.
    Positives = pixels with a gaussian label > 0 (label in (0,1], 1 at the annotation); the rest are negatives.
    Rank: push positives above negatives. Sort: order positives by their label. Step functions are smoothed
    with width 2*delta and gradients come from the error-driven update of the RS loss (vectorised here)."""

    @staticmethod
    def forward(ctx, logits, labels, delta=0.5, eps=1e-10):
        grads = torch.zeros_like(logits)
        fgm = labels > 0
        f, t = logits[fgm], labels[fgm]
        P = f.numel()
        thr = f.min() - delta
        bgm = (labels == 0) & (logits >= thr)
        g = logits[bgm]
        FR = ((f[None, :] - f[:, None]) / (2 * delta) + 0.5).clamp(0, 1)        # (P,P) j above i
        BR = ((g[None, :] - f[:, None]) / (2 * delta) + 0.5).clamp(0, 1)        # (P,Nb)
        rank_pos = 1 + FR.sum(1); FP = BR.sum(1); rank = rank_pos + FP
        rerr = FP / rank
        cur_sort = (FR * (1 - t)[None]).sum(1) / rank_pos
        rel = (t[None, :] >= t[:, None]).float()
        tso = rel * FR
        tgt_sort = (tso * (1 - t)[None]).sum(1) / tso.sum(1).clamp(min=eps)
        serr = cur_sort - tgt_sort
        fg_grad = torch.zeros_like(f); bg_grad = torch.zeros_like(g)
        hasfp = FP > eps
        fg_grad = fg_grad - torch.where(hasfp, rerr, torch.zeros_like(rerr))
        bg_grad = bg_grad + (BR * torch.where(hasfp, rerr / FP.clamp(min=eps), torch.zeros_like(rerr))[:, None]).sum(0)
        mis = (1 - rel) * FR
        den = mis.sum(1); hasmis = den > eps
        fg_grad = fg_grad - torch.where(hasmis, serr, torch.zeros_like(serr))
        fg_grad = fg_grad + (mis * torch.where(hasmis, serr / den.clamp(min=eps), torch.zeros_like(serr))[:, None]).sum(0)
        grads[fgm] = fg_grad / P
        grads[bgm] = bg_grad / P
        ctx.save_for_backward(grads)
        return rerr.mean() + serr.mean()

    @staticmethod
    def backward(ctx, gout):
        g, = ctx.saved_tensors
        return g * gout, None, None, None


def spatial_rs_loss(logit, tgt, scale, m):
    """logit (B,H,W), tgt (B,2) window units, scale (B,) metres per window unit, m (B,) bool positives-only"""
    B, H, W = logit.shape
    ys = (torch.arange(H, device=logit.device) + 0.5) / H
    xs = (torch.arange(W, device=logit.device) + 0.5) / W
    sig = (RS_SIGMA / scale.clamp(min=1e-6))[:, None, None]
    lab = torch.exp(-((xs[None, None, :] - tgt[:, None, None, 0]) ** 2 + (ys[None, :, None] - tgt[:, None, None, 1]) ** 2) / (2 * sig ** 2))
    lab = torch.where(lab > 0.05, lab, torch.zeros_like(lab))           # truncated gaussian: labels in (0.05, 1]
    out = []
    for b in range(B):
        if m[b] and (lab[b] > 0).any():
            out.append(SpatialRankSort.apply(logit[b].flatten().float(), lab[b].flatten(), RS_DELTA))
    return torch.stack(out).mean() if out else logit.sum() * 0
TZOOM, W_TIME, USE_TIME = 0.0, 0.0, False   # --tzoom / --w-time; USE_TIME: write refined start/end at inference


def loss_fn(uv, logit, tgt, scale, pos=None, conf=None, w_conf=0.5):
    dist_m = torch.norm(uv - tgt, dim=-1) * scale                       # meters
    pos = torch.ones_like(dist_m) if pos is None else pos
    m = pos > 0                                                         # location loss on positives only
    B, H, W = logit.shape
    if GAUSS_M > 0:   # soft heatmap target: gaussian of GAUSS_M metres around the label (label noise ~ cm)
        ys = (torch.arange(H, device=uv.device) + 0.5) / H
        xs = (torch.arange(W, device=uv.device) + 0.5) / W
        sig = (GAUSS_M / scale.clamp(min=1e-6))[:, None, None]          # in window units
        g = torch.exp(-((xs[None, None, :] - tgt[:, None, None, 0]) ** 2
                        + (ys[None, :, None] - tgt[:, None, None, 1]) ** 2) / (2 * sig ** 2))
        g = (g / g.flatten(1).sum(-1)[:, None, None].clamp(min=1e-12)).flatten(1)
        ce = -(g * logit.flatten(1).log_softmax(-1)).sum(-1)
        loss = dist_m[m].mean() + CE_W * ce[m].mean()
    else:
        ti = (tgt[:, 1].clamp(0, 0.999) * H).long() * W + (tgt[:, 0].clamp(0, 0.999) * W).long()
        loss = dist_m[m].mean() + CE_W * F.cross_entropy(logit.flatten(1)[m], ti[m])
    if RS_W > 0:      # Spatial Rank & Sort loss on the heatmap (RSPose)
        loss = loss + RS_W * spatial_rs_loss(logit, tgt, scale, m)
    if HIT_W > 0:     # metric-shaped term: push predictions inside the 5 cm / 10 cm thresholds
        hit = torch.sigmoid((0.05 - dist_m) / 0.01) + torch.sigmoid((0.10 - dist_m) / 0.01)
        loss = loss + HIT_W * (2 - hit[m]).mean()
    if conf is not None:
        target = (pos * (dist_m.detach() < 0.1).float())
        loss = loss + w_conf * F.binary_cross_entropy_with_logits(conf.float(), target)
    return loss, dist_m[m]


# ---------------------------------------------------------------- inference
@torch.no_grad()
def refine_cache(model, cache, root, split, topk=3, bs=64, dev="cuda", flip=False, shifts=((0.0, 0.0),)):
    """Replace the points of the top-k queries in a stage-1 cache with refined ones.
    model : one refiner or a list (their points are averaged). Refiners trained at different window
            resolutions are run group by group, each at its own resolution (attribute `patch`).
    flip  : also run the horizontally mirrored window
    shifts: window-centre offsets (normalized image units); every (model, shift, flip) estimate of
            the absolute point is averaged."""
    global PATCH
    models = model if isinstance(model, (list, tuple)) else [model]
    out = {n: {k: (v.copy() if hasattr(v, "copy") else v) for k, v in c.items()} for n, c in cache.items()}
    acc = {}                                              # (n, q, vi) -> [sum_x, sum_y, count, conf list]
    keep = PATCH
    for p in sorted({getattr(m, "patch", PATCH) for m in models}):
        PATCH = p
        _refine_cache_one([m for m in models if getattr(m, "patch", keep) == p], cache, root, split,
                          topk, bs, dev, flip, shifts, acc)
    PATCH = keep
    for key in [k for k in acc if k[0] == "time"]:
        _, n, q = key; t = np.mean(acc.pop(key), 0)                 # both views' estimates of the action's span
        if USE_TIME:
            out[n]["start"][q], out[n]["end"][q] = float(min(t)), float(max(t))
    for (n, q, vi), (sx, sy, k, cfs) in acc.items():
        if "conf" not in out[n]:
            out[n]["conf"] = np.full((len(out[n]["score"]), 2), np.nan, np.float32)
        out[n]["conf"][q, vi] = np.mean(cfs) if cfs else np.nan
        out[n]["points"][q, 2 * vi] = sx / k
        out[n]["points"][q, 2 * vi + 1] = sy / k
    return out


@torch.no_grad()
def _refine_cache_one(models, cache, root, split, topk, bs, dev, flip, shifts, acc):
    """Replace the points of the top-k queries in a stage-1 cache with refined ones.
    model : one refiner or a list (their points are averaged)
    flip  : also run the horizontally mirrored window
    shifts: window-centre offsets (normalized image units); every (model, shift, flip) estimate of
            the absolute point is averaged."""
    from concurrent.futures import ThreadPoolExecutor
    # one job per (clip, view): the video is decoded once and every refined query of that clip
    # is cut from it (decoding dominated the run time when it was done per query)
    jobs = [(n, [int(q) for q in np.argsort(-c["score"])[:topk]], vi, v)
            for n, c in cache.items() for vi, v in enumerate(VIEWS)]

    def make(job):
        n, qs, vi, v = job
        d = os.path.join(root, split, n)
        c = cache[n]
        meta = json.load(open(os.path.join(d, "metadata.json")))["content"]
        if TZOOM > 0:
            allf = read_all_frames(os.path.join(d, f"{v}_video.mp4"))
            if allf is None:
                return []
            nf = len(allf)
        else:
            frames, idx, nf = read_video(os.path.join(d, f"{v}_video.mp4"), T, False, None)
            if frames is None:
                return []
        pose = pose_to_array((meta["action_cam"].get(v) or {}).get("poses"), idx, T)
        out_ = []
        for q in qs:
            for dx, dy in shifts:
                cx = float(c["points"][q, 2 * vi]) + dx
                cy = float(c["points"][q, 2 * vi + 1]) + dy
                if TZOOM > 0:
                    idx = zoom_index((c["start"][q], c["end"][q]), nf, T, TZOOM); frames = allf[idx]
                    pose = pose_to_array((meta["action_cam"].get(v) or {}).get("poses"), idx, T)
                x, flag = build_input(frames, pose, idx / nf, cx, cy, (c["start"][q], c["end"][q]))
                out_.append(((n, q, vi, v), x, flag, cx, cy, (idx / nf).astype(np.float32)))
        return out_

    for m in models:
        m.eval()
    with ThreadPoolExecutor(32) as ex:
        per = 64                                          # (clip, view) videos decoded in parallel per step
        chunks = (b[s:s + bs] for st in range(0, len(jobs), per)
                  for b in [[w for ww in ex.map(make, jobs[st:st + per]) for w in ww]]
                  for s in range(0, len(b), bs))
        for built in chunks:
            if not built:
                continue
            x = torch.from_numpy(np.stack([b[1] for b in built])).to(dev)
            fl = torch.from_numpy(np.stack([b[2] for b in built])).to(dev)
            ft = torch.from_numpy(np.stack([b[5] for b in built])).to(dev)
            tms = []
            ests = []                                     # list of ((N,2) uv in window units, conf or None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for m in models:
                    uv, _, cf = m(x, fl, ft)
                    if getattr(m, "tdec", None) is not None and m._times is not None:
                        tms.append(m._times.float().cpu().numpy())
                    ests.append((uv.float().cpu().numpy(), cf))
                    if flip:
                        uv2, _, cf2 = m(x.flip(-1), fl)
                        uv2 = uv2.float().cpu().numpy()
                        ests.append((np.stack([1 - uv2[:, 0], uv2[:, 1]], 1), cf2))
            tm = np.mean(tms, 0) if tms else None                  # (N,2) refined start/end, if any model has a time head
            for i, (job, _, _, cx, cy, _) in enumerate(built):
                n, q, vi, _ = job
                if tm is not None:
                    acc.setdefault(("time", n, q), []).append(tm[i])
                a_ = acc.setdefault((n, q, vi), [0.0, 0.0, 0, []])
                for uv, cf in ests:
                    a_[0] += cx - HALF + uv[i, 0] * 2 * HALF
                    a_[1] += cy - HALF + uv[i, 1] * 2 * HALF
                    a_[2] += 1
                    if cf is not None:
                        a_[3].append(float(torch.sigmoid(cf[i].float())))


def rescore(cache, alpha=1.0):
    """score <- score * max_view_conf^alpha for refined queries (metric accepts either view)."""
    out = {}
    for n, c in cache.items():
        c = dict(c)
        if "conf" in c:
            cf = np.nanmax(c["conf"], 1)
            ok = ~np.isnan(cf)
            s = c["score"].copy()
            s[ok] = s[ok] * cf[ok] ** alpha
            c["score"] = s
        out[n] = c
    return out


def score_cache(cache, root, split):
    from official_metric import evaluate_official
    samples = []
    for n, c in cache.items():
        meta = json.load(open(os.path.join(root, split, n, "metadata.json")))["content"]
        G = [(a["start"], a["end"], *(a["spatial"]["action_cam"][v][k] for v in VIEWS for k in "xy"))
             for a in meta["labels"]["action"]]
        fac = tuple(official_factor((meta["action_cam"].get(v) or {}).get("poses")) for v in VIEWS)
        P = np.concatenate([c["score"][:, None], c["start"][:, None], c["end"][:, None], c["points"]], 1)
        samples.append(dict(gt=G, pred=P.tolist(), factor=fac))
    return evaluate_official(samples)


# ---------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/RetailAction/data")
    ap.add_argument("--stage1-val-cache", default="runs/cache/fact_full_val.pkl")
    ap.add_argument("--out", default="runs/refine_r18")
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=0.0, help="separate backbone lr (0 = same as --lr)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--patch", type=int, default=128, help="window resolution (180 px = native)")
    ap.add_argument("--conf", action="store_true", help="train a confidence head (+negative windows)")
    ap.add_argument("--neg-p", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--train-splits", default="train", help='comma-separated, e.g. "train,test"')
    ap.add_argument("--half", type=float, default=0.15, help="window half-size, normalized (0.15 = ~1 m)")
    ap.add_argument("--ce", type=float, default=0.05, help="heatmap CE weight")
    ap.add_argument("--rev-p", type=float, default=0.0, help="time-reversal probability")
    ap.add_argument("--fpn", action="store_true")
    ap.add_argument("--strong-aug", action="store_true", help="90-degree rotations + colour jitter")
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--save-epochs", action="store_true", help="also save <out>/epNN.pt after every epoch")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/resume.pt if it exists")
    ap.add_argument("--resume-every", type=int, default=0, help="save <out>/resume.pt every N steps and after each epoch")
    ap.add_argument("--fold", type=int, default=-1); ap.add_argument("--nfolds", type=int, default=0, help="k-fold: hold out fold --fold of --nfolds")
    ap.add_argument("--init", default="", help="start from these refiner weights (continued fine-tuning)")
    ap.add_argument("--freeze-layers", type=int, default=0, help="ViTPose: freeze the first N transformer layers")
    ap.add_argument("--w-rs", type=float, default=0.0, help="Spatial Rank & Sort heatmap loss weight (RSPose)")
    ap.add_argument("--rs-sigma", type=float, default=0.02, help="gaussian label width for RS positives, metres")
    ap.add_argument("--rs-delta", type=float, default=0.5, help="RS step-function smoothing")
    ap.add_argument("--temporal-layers", type=int, default=0, help="temporal transformer layers over frames (0 = per-frame refiner)")
    ap.add_argument("--tzoom", type=float, default=0.0, help="sample frames only around the (noisy/predicted) span; margin x span length")
    ap.add_argument("--w-time", type=float, default=0.0, help="time head: L1 weight on start/end (0 = no time head)")
    ap.add_argument("--w-hit", type=float, default=0.0, help="weight of the soft 5/10 cm hit loss")
    ap.add_argument("--gauss-sigma", type=float, default=0.0, help="gaussian heatmap target width in metres (0 = one-hot)")
    ap.add_argument("--save-last", action="store_true", help="also save the last epoch as last.pt")
    ap.add_argument("--patience", type=int, default=0, help="early stop after this many epochs without a new best (0 = off)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights used for eval/saving (0 = off)")
    ap.add_argument("--levels", default="", help='feature levels to fuse, e.g. "0,1,2" (stride 4/8/16)')
    ap.add_argument("--noise", default="mix", choices=["mix", "empirical"])
    args = ap.parse_args()
    global PATCH, T, HALF, CE_W, HIT_W, GAUSS_M, TZOOM, W_TIME, USE_TIME, RS_W, RS_SIGMA, RS_DELTA
    PATCH = args.patch      # module-level, inherited by forked DataLoader workers
    T = args.frames
    HALF = args.half
    CE_W = args.ce
    HIT_W, GAUSS_M = args.w_hit, args.gauss_sigma
    TZOOM, W_TIME = args.tzoom, args.w_time
    RS_W, RS_SIGMA, RS_DELTA = args.w_rs, args.rs_sigma, args.rs_delta
    USE_TIME = args.w_time > 0
    assert not (args.w_time > 0 and args.rev_p > 0), "time targets assume forward playback"

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    dev = "cuda"

    tr = RefineTrainSet(args.root, args.train_splits, True, neg_p=args.neg_p if args.conf else 0.0, noise=args.noise, strong=args.strong_aug,
                        fold=args.fold, nfolds=args.nfolds,
                        rev_p=args.rev_p)
    va = RefineTrainSet(args.root, "validation", False, noise=args.noise)
    dl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate_skip,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    dlv = DataLoader(va, args.bs, num_workers=args.workers, collate_fn=collate_skip)
    lv = tuple(int(v) for v in args.levels.split(",")) if args.levels else None
    model = Refiner(args.backbone, conf=args.conf, fpn=args.fpn, levels=lv, drop_path=args.drop_path,
                    time_head=args.w_time > 0, temporal_layers=args.temporal_layers).to(dev)
    ema = None
    if args.ema > 0:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=args.ema)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        print(f"initialised from {args.init}", flush=True)
    if args.freeze_layers > 0:     # keep the first N pretrained transformer layers fixed (patch embedding stays trainable)
        for layer in model.net.encoder.layer[:args.freeze_layers]:
            for p_ in layer.parameters():
                p_.requires_grad = False
        print(f"froze {args.freeze_layers} of {len(model.net.encoder.layer)} ViTPose layers: "
              f"{sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)/1e6:.1f}M trainable", flush=True)
    if args.lr_backbone > 0:   # pretrained backbone gets its own (lower) learning rate
        bb = [p_ for p_ in model.net.parameters() if p_.requires_grad]; ids = {id(p) for p in model.net.parameters()}
        groups = [dict(params=bb, lr=args.lr_backbone),
                  dict(params=[p for p in model.parameters() if id(p) not in ids], lr=args.lr)]
    else:
        groups = [dict(params=[p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr)]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    s1 = pickle.load(open(args.stage1_val_cache, "rb"))
    base = score_cache(s1, args.root, "validation")
    print(f"stage-1 only on validation: mAP {base['mAP']*100:.2f} mAPs {base['mAPs']*100:.2f}", flush=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best = -1; stale = 0; start_ep, skip = 0, 0
    RP = os.path.join(args.out, "resume.pt")

    def save_resume(ep, step):          # everything needed to continue exactly where we stopped (atomic write)
        st = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, step=step,
                  best=best, stale=stale, ema=ema.module.state_dict() if ema is not None else None, args=vars(args))
        torch.save(st, RP + ".tmp"); os.replace(RP + ".tmp", RP)

    if args.resume and os.path.exists(RP):
        st = torch.load(RP, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if ema is not None and st["ema"] is not None:
            ema.module.load_state_dict(st["ema"])
        best, stale, start_ep, skip = st["best"], st["stale"], st["epoch"], st["step"]
        print(f"resumed from {RP}: epoch {start_ep}, {skip} steps already done", flush=True)
    for ep in range(start_ep, args.epochs):
        model.train(); t0 = time.time(); agg = []
        todo = len(dl) - (skip if ep == start_ep else 0)        # keep the lr schedule aligned after a resume
        done_before = len(dl) - todo
        for j, (x, fl, tgt, sc, pos, ftb, ttb) in enumerate(dl):
            if j >= todo:
                break
            i = j + done_before
            x, fl, tgt, sc, pos = (x.to(dev, non_blocking=True), fl.to(dev), tgt.to(dev),
                                   sc.to(dev), pos.to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                uv, logit, cf = model(x, fl, ftb.to(dev))
            loss, dm = loss_fn(uv.float(), logit.float(), tgt, sc, pos, cf)
            if W_TIME > 0 and model._times is not None:     # L1 on the action's start/end (clip-normalised time)
                loss = loss + W_TIME * F.l1_loss(model._times, ttb.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if ema is not None:
                ema.update(model)
            agg.append(dm.detach().median().item())
            if args.resume_every and (i + 1) % args.resume_every == 0 and i + 1 < len(dl):
                save_resume(ep, i + 1)
            if i % 100 == 0:
                print(f"[{time.strftime('%m-%d %H:%M:%S KST', time.localtime(time.time() + 9 * 3600 - (time.localtime().tm_gmtoff)))}] ep{ep} {i}/{len(dl)} median dist {np.mean(agg[-100:]):.3f} m", flush=True)
        # (a) synthetic windows on validation, (b) the real thing: refine stage-1 predictions
        net = ema.module if ema is not None else model       # evaluate / save the EMA weights
        net.eval(); dv = []
        with torch.no_grad():
            for x, fl, tgt, sc, pos, _, _ in dlv:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    uv, _, _ = net(x.to(dev), fl.to(dev))
                d = (torch.norm(uv.float() - tgt.to(dev), dim=-1) * sc.to(dev)).cpu()
                dv.append(d[pos > 0])
        dv = torch.cat(dv)
        refined = refine_cache(net, s1, args.root, "validation", args.topk, bs=args.bs)
        r = score_cache(refined, args.root, "validation")
        rec = dict(epoch=ep, time=time.time() - t0, train_med=float(np.mean(agg)),
                   val_synth_med=float(dv.median()), mAP=r["mAP"], mAPs=r["mAPs"], mAPt=r["mAPt"])
        if USE_TIME:      # same refined points, stage-1 spans: separates the point gain from the time gain
            pts_only = {n: {**c, "start": s1[n]["start"], "end": s1[n]["end"]} for n, c in refined.items()}
            rp = score_cache(pts_only, args.root, "validation")
            rec.update(mAP_points_only=rp["mAP"], mAPt_points_only=rp["mAPt"])
            print(f"   points only (stage-1 spans): mAP {rp['mAP']*100:.2f} mAPt {rp['mAPt']*100:.2f} | with refined time: mAPt {r['mAPt']*100:.2f}", flush=True)
        msg = ""
        if args.conf:     # re-ranking by the confidence head, a couple of strengths
            for alpha in (0.5, 1.0):
                rr = score_cache(rescore(refined, alpha), args.root, "validation")
                rec[f"mAP_rescore{alpha}"] = rr["mAP"]
                msg += f" | rescore a={alpha} mAP {rr['mAP']*100:.2f}"
            r = {**r, "mAP": max(r["mAP"], rec["mAP_rescore0.5"], rec["mAP_rescore1.0"])}
        print(f"[{time.strftime('%m-%d %H:%M:%S KST', time.localtime(time.time() + 9 * 3600 - (time.localtime().tm_gmtoff)))}] == ep{ep} synth-val median {rec['val_synth_med']:.3f} m | stage1+refine "
              f"mAP {rec['mAP']*100:.2f} (stage1 {base['mAP']*100:.2f}) mAPs {r['mAPs']*100:.2f}{msg}", flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if args.save_last:      # full-data runs: val is in the training set, so the last epoch is what we use
            torch.save({"model": net.state_dict(), "args": vars(args), "epoch": ep, "mAP": r["mAP"]},
                       os.path.join(args.out, "last.pt"))
        if args.save_epochs:    # keep every epoch's (EMA) weights so any intermediate checkpoint can be used
            torch.save({"model": net.state_dict(), "args": vars(args), "epoch": ep, "mAP": r["mAP"]},
                       os.path.join(args.out, f"ep{ep:02d}.pt"))
        if r["mAP"] > best:
            best = r["mAP"]; stale = 0
            torch.save({"model": net.state_dict(), "args": vars(args), "epoch": ep, "mAP": best},
                       os.path.join(args.out, "best.pt"))
        else:
            stale += 1
        if args.resume_every:
            save_resume(ep + 1, 0)
        if args.patience and stale >= args.patience:
            print(f"early stop at ep{ep}: no new best for {stale} epochs", flush=True)
            break
    print(f"best stage1+refine val mAP {best*100:.2f}")


if __name__ == "__main__":
    main()
