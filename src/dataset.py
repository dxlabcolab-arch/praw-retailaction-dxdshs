"""RetailAction dataset: dual-view clips + pose, for DETR-style set prediction."""
import json, os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

cv2.setNumThreads(0)

LABELS = ["take", "put", "touch"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
VIEWS = ("rank0", "rank1")

# 13 joints actually present in the metadata (paper mentions 18; released data has these)
JOINT_NAMES = [
    "top_of_head", "neck",
    "right_shoulder", "left_shoulder",
    "right_elbow", "left_elbow",
    "right_wrist", "left_wrist",
    "right_hand", "left_hand",
    "middle_of_waist", "right_hip", "left_hip",
]
J2I = {j: i for i, j in enumerate(JOINT_NAMES)}
NUM_JOINTS = len(JOINT_NAMES)
HAND_JOINTS = ["left_hand", "right_hand", "left_wrist", "right_wrist"]


def read_video(path, num_frames, train=False, rng=None):
    """Decode mp4 -> (T,H,W,3) uint8 RGB, temporally resampled to num_frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])  # BGR->RGB
    cap.release()
    if not frames:
        return None, None, 0
    n = len(frames)
    # uniform sampling with (optional) jitter inside each bin
    idx = np.linspace(0, n, num_frames + 1)[:-1]
    if train and rng is not None:
        step = n / num_frames
        idx = idx + rng.random(num_frames) * step
    idx = np.clip(idx.astype(np.int64), 0, n - 1)
    return np.stack([frames[i] for i in idx]), idx, n


def pose_to_array(poses, idx, num_frames):
    """-> (T, J, 3) with (x, y, score); score 0 where missing."""
    out = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)
    if not poses:
        return out
    n = len(poses)
    for t, i in enumerate(idx):
        i = min(int(i), n - 1)
        p = poses[i]
        if not p or not p.get("pose"):
            continue
        sc = p.get("joint_scores") or {}
        for j, xy in p["pose"].items():
            if j not in J2I:
                continue
            k = J2I[j]
            out[t, k, 0] = xy[0]
            out[t, k, 1] = xy[1]
            out[t, k, 2] = min(float(sc.get(j, 1.0)), 1.0)
    return out


class RetailActionDataset(Dataset):
    def __init__(self, root, split, num_frames=16, img_size=224, train=False,
                 max_actions=8, rev_p=0.0):
        # split may be comma-separated ("train,validation,test") to pool labelled splits
        self.dirs = sorted(
            d for sp in split.split(",")
            for d in glob.glob(os.path.join(root, sp, "*")) if os.path.isdir(d)
        )
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train
        self.max_actions = max_actions
        self.split = split
        self.rev_p = rev_p          # time-reversal augmentation probability (train only)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        rng = np.random.default_rng(None if self.train else i)
        meta = json.load(open(os.path.join(d, "metadata.json")))["content"]

        # ---- augmentation params (shared across views so geometry stays consistent
        #      with each view's own labels; flip is per-view-safe since labels are 2D
        #      per view and we flip both image and label coordinates together) ----
        do_flip = bool(self.train and rng.random() < 0.5)
        if self.train:
            scale = 0.8 + 0.2 * rng.random()          # crop 80-100% of the frame
            cx = rng.random() * (1 - scale)
            cy = rng.random() * (1 - scale)
        else:
            scale, cx, cy = 1.0, 0.0, 0.0

        do_rev = bool(self.train and self.rev_p > 0 and rng.random() < self.rev_p)
        clips, poses_all = [], []
        ftimes = None; idx0 = None; n0 = None
        for v in VIEWS:
            vid, idx, n = read_video(
                os.path.join(d, f"{v}_video.mp4"), self.num_frames, self.train, rng
            )
            if vid is None:
                vid = np.zeros((self.num_frames, self.img_size, self.img_size, 3), np.uint8)
                idx = np.zeros(self.num_frames, np.int64)
            elif ftimes is None:
                # normalized time of each sampled frame. Labels are frame indices / n
                # (97% of start/end sit exactly on that grid), so this is lossless.
                ftimes = idx.astype(np.float32) / n
                idx0, n0 = idx.copy(), n
            H, W = vid.shape[1:3]
            x0, y0 = int(cx * W), int(cy * H)
            x1, y1 = x0 + int(scale * W), y0 + int(scale * H)
            vid = vid[:, y0:y1, x0:x1]
            vid = np.stack([cv2.resize(f, (self.img_size, self.img_size)) for f in vid])
            if do_flip:
                vid = vid[:, :, ::-1]
            clips.append(np.ascontiguousarray(vid))

            cam = meta["action_cam"].get(v) or {}
            pa = pose_to_array(cam.get("poses"), idx, self.num_frames)
            # apply the same crop/flip to pose coordinates
            vis = pa[..., 2] > 0
            pa[..., 0] = (pa[..., 0] - cx) / scale
            pa[..., 1] = (pa[..., 1] - cy) / scale
            if do_flip:
                pa[..., 0] = 1.0 - pa[..., 0]
            # joints pushed outside the crop lose their confidence
            oob = (pa[..., 0] < 0) | (pa[..., 0] > 1) | (pa[..., 1] < 0) | (pa[..., 1] > 1)
            pa[..., 2][oob | ~vis] = 0.0
            pa[..., :2] = np.clip(pa[..., :2], 0.0, 1.0)
            poses_all.append(pa)

        # ---- time reversal: a reversed "take" is a "put" (hand brings the item in,
        #      lets go, leaves). Contact points are unchanged; spans mirror to [1-e, 1-s];
        #      sampled frame i (covering [i/n,(i+1)/n)) moves to (n-1-i)/n. Losses that
        #      depend on the exact temporal boundaries are skipped for these samples
        #      because take/put boundary conventions are not perfectly symmetric.
        if do_rev:
            clips = [np.ascontiguousarray(c[::-1]) for c in clips]
            poses_all = [np.ascontiguousarray(p[::-1]) for p in poses_all]
            if idx0 is not None:
                ftimes = ((n0 - 1 - idx0[::-1]).astype(np.float32) / n0)
        swap = {LABEL2ID["take"]: LABEL2ID["put"], LABEL2ID["put"]: LABEL2ID["take"]}

        # ---- targets ----
        acts = (meta.get("labels") or {}).get("action") or []
        acts = acts[: self.max_actions]
        cls, spans, pts = [], [], []
        for a in acts:
            c = LABEL2ID.get(a["label"], 0)
            cls.append(swap.get(c, c) if do_rev else c)
            spans.append([1.0 - a["end"], 1.0 - a["start"]] if do_rev else [a["start"], a["end"]])
            p = []
            for v in VIEWS:
                sp = a["spatial"]["action_cam"][v]
                x, y = (sp["x"] - cx) / scale, (sp["y"] - cy) / scale
                if do_flip:
                    x = 1.0 - x
                p += [x, y]
            pts.append(p)

        sample = {
            "clips": torch.from_numpy(np.stack(clips)),          # (2,T,H,W,3) uint8
            "poses": torch.from_numpy(np.stack(poses_all)),      # (2,T,J,3)
            "labels": torch.tensor(cls, dtype=torch.long),       # (N,)
            "spans": torch.tensor(spans, dtype=torch.float32).reshape(-1, 2),
            "points": torch.tensor(pts, dtype=torch.float32).reshape(-1, 4),
            "ftimes": torch.from_numpy(ftimes if ftimes is not None else
                                       np.linspace(0, 1, self.num_frames, endpoint=False,
                                                   dtype=np.float32)),
            "rev": do_rev,
            "index": i,
            "name": os.path.basename(d),
        }
        return sample


def collate(batch):
    return {
        "clips": torch.stack([b["clips"] for b in batch]),
        "poses": torch.stack([b["poses"] for b in batch]),
        "targets": [
            {"labels": b["labels"], "spans": b["spans"], "points": b["points"],
             "ftimes": b["ftimes"], "rev": b.get("rev", False)}
            for b in batch
        ],
        "names": [b["name"] for b in batch],
    }
