# PRAW RetailAction Challenge (WACV 2027) — team DX_DSHS, inference code

Reproduces our submitted predictions on the challenge test set with **one Docker command**.
Training code is not included (not required). Method description: [ARCHITECTURE.md](ARCHITECTURE.md).

| preset | what | public LB |
|---|---|---|
| `ens6_vitl` (default) — **final submission 1** | 4 stage-1 models + temporal fusion → 6 V-JEPA 2.1 ViT-L refiners (seeds) averaged → soft-NMS | **0.9762** |
| `single_vitl` — **final submission 2** | same, with one refiner (seed 1) | 0.9756 |
| `ens4_vitl` | earlier 4-seed version of the ensemble | 0.9762 |

## Requirements
- Linux, NVIDIA GPU with **≥ 48 GB** memory (inference of the 6-refiner preset peaks around 40 GB), NVIDIA driver ≥ 570 (CUDA 12.8)
- Docker with the NVIDIA Container Toolkit (`docker run --gpus all` must work)
- ~60 GB free disk (image ≈ 30 GB incl. weights), internet access **at build time** (weights + public initialisations are
  downloaded into the image; the container itself runs offline)

## Data layout
Mount the challenge test set so that the container sees
```
/data/test/<id>/rank0_video.mp4
/data/test/<id>/rank1_video.mp4
/data/test/<id>/metadata.json
```
With the Kaggle download the clip folders are in `vam-dataset-challenge_test_no_labels/data/`; mount that folder as
`/data/test` (mount the real folder, not a symlink).

## Run (single command)
```bash
git clone https://github.com/dxlabcolab-arch/praw-retailaction-dxdshs.git && cd praw-retailaction-dxdshs
docker build -t praw-dxdshs .        # ~30-40 min: pip, public initialisations, ~16 GB of our weights (md5-checked)
docker run --rm --gpus all --shm-size=16g \
  -v /path/to/vam-dataset-challenge_test_no_labels/data:/data/test:ro \
  -v $(pwd)/output:/output praw-dxdshs
# -> output/submission.csv   (preset ens6_vitl = final submission 1)

# the other submitted file:
docker run --rm --gpus all --shm-size=16g -v .../data:/data/test:ro -v $(pwd)/output:/output praw-dxdshs single_vitl /data /output/submission_single.csv
# compare with the submitted file (md5) at the end of the run:
docker run --rm --gpus all --shm-size=16g -e VERIFY=1 -v .../data:/data/test:ro -v $(pwd)/output:/output praw-dxdshs
```
Runtime on one RTX PRO 6000 (96 GB) for the 2,500 test clips: stage 1 ≈ 30 min, refiners ≈ 16 min each
(`ens6_vitl` ≈ 2 h in total, measured 1 h 50 min – 2 h 10 min end to end; `single_vitl` ≈ 50 min).

## Without Docker
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python prefetch_pretrained.py && bash download_weights.sh
bash run.sh ens6_vitl /path/to/challenge ./submission.csv       # /path/to/challenge contains test/<id>/...; presets as above
```

## Weights
`weights/` is downloaded at build time from our Hugging Face model repo (see `download_weights.sh`, md5 list in
`weights/MD5SUMS`, HF repo `juuuncha/praw-retailaction-dxdshs-weights`): 4 stage-1 checkpoints (V-JEPA 2 ViT-L ×3, ViT-g ×1; 1.35–4.2 GB each) and 6 refiner checkpoints
(V-JEPA 2.1 ViT-L, 1.4 GB each).

| file | model |
|---|---|
| `stage1_1.pt` … `stage1_4.pt` | stage 1 (V-JEPA 2 + FACT-DETR), four models of the same architecture (ViT-L ×3, ViT-g ×1), fused at inference; details in `ARCHITECTURE.md` |
| `refiner_1.pt` … `refiner_6.pt` | stage 2 point refiner, V-JEPA 2.1 ViT-L, all labelled data, six random seeds (`ens6_vitl` = all six, `single_vitl` = `refiner_1.pt`, `ens4_vitl` = 1, 2, 3, 6) |
| `expected_<preset>.csv` | the submitted prediction file of each preset (used by `VERIFY=1`) |

Public initialisations are fetched from their original repos at pinned revisions:
`facebook/vjepa2-vitl-fpc64-256`, `facebook/vjepa2-vitg-fpc64-256` (V-JEPA 2, MIT), `apiantonio/vjepa2.1-vit-large-384`
(HF port of Meta's V-JEPA 2.1 ViT-L; the remote code is pinned by revision), `usyd-community/vitpose-plus-base` (ViTPose++,
Apache-2.0; only its config is used to build the refiner head).

## Reproducibility
`VERIFY=1` compares the produced CSV with the submitted one. On our hardware (NVIDIA Blackwell, torch 2.11.0 + CUDA 12.8)
the output is byte-identical. On other GPUs or library versions the last digits of the bf16 outputs can differ; the score
difference is negligible. The pipeline is deterministic (no sampling, fixed checkpoints, fixed fusion rules).

## Files
```
run.sh                       entry point: stage 1 ×4 -> temporal WBF -> refiner(s) -> soft-NMS
src/cache_preds.py           stage-1 inference        src/stage1_wbf.py      temporal weighted fusion
src/refine_vjepa.py          V-JEPA 2.1 refiner (uses src/refine.py, src/make_refined_submission.py)
src/softnms_csv.py           soft-NMS                 src/model_fact.py, backbone_hf.py, dataset.py, ...  model code
Dockerfile, requirements.txt, prefetch_pretrained.py, download_weights.sh, weights/MD5SUMS
```

## License
Code: MIT (see LICENSE). Third-party weights keep their own licenses (see "Weights").
