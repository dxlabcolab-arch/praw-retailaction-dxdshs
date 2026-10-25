# DX_DSHS — PRAW RetailAction Challenge: method

Final submissions: `ens6_vitl` (public LB 0.9762) and `single_vitl` (public LB 0.9756); see README.
Task: two synchronised ceiling cameras (rank0, rank1), ≤ 32 frames at 6 fps, 600×600, plus the shopper's 2D pose per frame
(13 joints: x, y, confidence). For every shopper–product interaction, predict start/end time (normalised to [0, 1]),
the contact point (x, y) in **each** camera, and a confidence score.

The pipeline has four steps. Only steps 1 and 3 are neural networks.

```
 rank0 video ─┐                                                                         ┌─ refiner A: ViTPose++-B (image) ─┐
 rank1 video ─┼─► STAGE 1 ×4 models ─► 4 × 10 predictions ─► TEMPORAL WBF ─► 10 preds ─┤                                  ├─ mean ─► SOFT-NMS ─► CSV
 poses ───────┘   (V-JEPA 2 + FACT-DETR)                     (fuse the 4)    (top 3)   └─ refiner B: V-JEPA 2.1-B (video) ┘  points   (scores)
                  detection + time + coarse point                                         ±1 m crop → precise point          ×2 passes
```

## 1. Stage 1 — detection, timing and a coarse point (`src/model_fact.py`, `src/model.py`, `src/backbone_hf.py`)

Per camera view:
- **Video backbone: V-JEPA 2** (self-supervised video ViT, Meta). Input 32 frames × 256×256 px.
  Tubelets of 2 frames × 16×16 px → **16 (time) × 16×16 (space) tokens**, width 1024 (ViT-L) or 1408 (ViT-g).
  Fine-tuned end-to-end (backbone lr 1e-5).
- `proj`: linear to 256-d, + learned factorised (t, h, w) position embeddings + view embedding.
- **Video encoder**: 2 Transformer layers (8 heads) over the 16·256 tokens of a view.
- **Pose encoder**: per frame, 13 joints × (x, y, conf) and joint velocity → linear → 2 Transformer layers → 1 token per frame.

Both views together:
- **Memory** = video tokens of both views + pose tokens of both views.
- **Per-frame features** (for the frame branch): video tokens pooled over space, resampled to 32 frames, + pose token;
  the two views are concatenated and projected to 256-d.

Decoder (**FACT**, CVPR 2024 "Frame-Action Cross-attention Temporal modeling"):
- **Frame branch 1**: MS-TCN-style dilated temporal convolutions (dilations 1, 2, 4, 8) → per-frame action / background logits.
- **Action branch**: 10 learned queries, **DETR decoder** (4 layers, 8 heads) attending to the memory.
- **Update block**: bidirectional cross-attention between queries and frames (query ← frames, frames ← queries),
  then **frame branch 2**. The attention maps are supervised to cover the frames of the matched action.
- **Heads** per query:
  - class (take / put / touch / no-action) → the score is the max non-background probability,
  - span: (centre, width) → start, end,
  - point per view, **pose-anchored**: softmax over (32 frames × 4 hand joints: left/right hand, left/right wrist),
    weighted mean of those joint positions, plus a learned offset (tanh × 0.25).
- Training: Hungarian matching (class + L1 span + temporal IoU + point distance + attention IoU),
  losses = CE + L1 span + IoU + L2 point + frame CE + smoothing + attention supervision. 32 frames, 40 epochs.

**Four stage-1 models** (same architecture):

| model | backbone | trained on |
|---|---|---|
| A | V-JEPA 2 ViT-L | train + val + HF test (all labelled data) |
| B | V-JEPA 2 ViT-g | all labelled data |
| C | V-JEPA 2 ViT-L, 20 epochs | train |
| D | V-JEPA 2 ViT-L, backbone lr 2e-5 | train |

## 2. Temporal Weighted Box Fusion (`src/stage1_wbf.py`)

The 4 × 10 predictions of a clip are pooled and sorted by score. A prediction joins the cluster whose
(score-weighted) time span it overlaps most with **tIoU ≥ 0.5** (at most one prediction per model per cluster),
otherwise starts a new cluster. Each cluster becomes one prediction:
span and points = score-weighted mean, score = sum of scores / 4. The top 10 clusters are kept.

## 3. Stage 2 — point refiner (`src/refine.py`, `src/refine_vjepa.py`)

Applied to the **3 highest-scoring predictions**, separately for each camera view, **twice** (the second pass is centred
on the first pass's output). With several refiners their (x, y) outputs are averaged (plain mean).

Input window, centred on the current point:
- crop of ±0.15 image width (≈ 1 m) from the full-resolution video, resized to **256×256**, all 32 frames,
- 4 channels: RGB + a "hand heat" channel (Gaussian blobs at the visible hand/wrist joints),
- a per-frame flag: is this frame inside the predicted time span.

Network:
1. **Backbone: V-JEPA 2.1 ViT-L** (Meta, self-supervised *video* model, HF port at a pinned revision). The whole window is
   one video (2-frame tubelets × 16×16 patches), so hand motion around the contact is seen directly. The 4th input channel is
   added to the patch embedding (initialised to zero). Features of layers 11, 17 and 23 (each 16×16 per frame, width 1024;
   each tubelet's map is repeated for its 2 frames).
2. **Fuse**: concat the 3 levels → 1×1 conv → 3×3 conv.
3. **Temporal pooling**: per-frame score = linear(spatial mean + span-flag embedding), softmax over the 32 frames;
   attention-weighted sum of the frame maps, concatenated with the max over frames.
4. **Heatmap head**: 3 convs → 16×16 logits → bilinear upsampling to 128×128 → softmax → **soft-argmax** = (u, v),
   mapped back to image coordinates of the view.

Training: windows centred on the ground-truth point plus noise drawn from stage 1's empirical error distribution;
loss = distance in metres (pose-based pixel→metre factor) + 0.05 × heatmap cross-entropy; horizontal flip, 90° rotations and
colour jitter; EMA of the weights (0.999); AdamW (backbone lr 5e-5, rest 2e-4), batch 16, 20 epochs (one seed 30),
on **all labelled data** (train + validation + HF test); the last epoch is used.
`ens6_vitl` averages six such refiners that differ only in the random seed (two were trained on a second server with the same recipe); `single_vitl` uses the first one.

## 4. Temporal soft-NMS (`src/softnms_csv.py`)

Predictions of a clip in score order: a prediction whose span overlaps (tIoU ≥ 0.1) any higher-scored one keeps its
time and points but its score is multiplied by 0.01. Ten predictions per clip are submitted.

## Contribution of each part (public leaderboard)

| change | LB |
|---|---|
| FACT stage 1 only (early backbone) | 0.7045 |
| + point refiner | 0.7901 |
| 16 → 32 frames | 0.8569 |
| V-JEPA 2 backbone | 0.9138 |
| better refiner + retrain on all labelled data | 0.9388 |
| ConvNeXt refiner on all data + soft-NMS | 0.9596 |
| ViTPose++ refiner | 0.9615 |
| 4-model temporal WBF in stage 1 | 0.9682–0.9685 |
| V-JEPA 2.1 ViT-B video refiner (replaces ViTPose++) | 0.9723 |
| V-JEPA 2.1 ViT-L refiner, regularised, all labelled data (`single_vitl`) | 0.9756 |
| mean of 4 seeds of that refiner (`ens4_vitl`) | 0.9762 |
| **mean of 6 seeds (`ens6_vitl`, final)** | **0.9762** |
