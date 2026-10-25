#!/bin/bash
# Single entry point: challenge test videos -> submission CSV.
# usage: bash run.sh [PRESET] [DATA_ROOT] [OUT_CSV]
#   PRESET     ens6_vitl (default; public LB 0.9762, final submission 1)  |  single_vitl (public LB 0.9756, final submission 2)
#              ens4_vitl (public LB 0.9762, earlier 4-seed version)
#   DATA_ROOT  folder containing test/<id>/{rank0_video.mp4,rank1_video.mp4,metadata.json}   (default /data)
#   OUT_CSV    default /output/submission.csv
# Optional: VERIFY=1 compares the result with the submitted file (weights/expected_<preset>.csv) by md5.
set -e
cd "$(dirname "$0")"
PRESET=${1:-ens6_vitl}; ROOT=${2:-/data}; OUT=${3:-/output/submission.csv}
W=weights; mkdir -p "$(dirname "$OUT")"; TMP=$(mktemp -d)
case $PRESET in
  ens6_vitl)   REF=$W/refiner_1.pt,$W/refiner_2.pt,$W/refiner_3.pt,$W/refiner_6.pt,$W/refiner_4.pt,$W/refiner_5.pt ;;
  ens4_vitl)   REF=$W/refiner_1.pt,$W/refiner_2.pt,$W/refiner_3.pt,$W/refiner_6.pt ;;
  single_vitl) REF=$W/refiner_1.pt ;;
  *) echo "unknown preset $PRESET"; exit 1 ;;
esac
EXPECTED=$W/expected_$PRESET.csv
echo "[1/4] stage 1: four V-JEPA 2 + FACT-DETR models (ViT-L all data, ViT-g all data, 2x ViT-L train split), 10 predictions per clip each"
i=0; C=""
for ck in stage1_1 stage1_2 stage1_3 stage1_4; do      # ViT-L all data, ViT-g all data, ViT-L train split (20 ep), ViT-L train split (backbone lr 2e-5)
  python src/cache_preds.py --ckpt $W/$ck.pt --root "$ROOT" --split test --out $TMP/s1_$i.pkl; C="$C $TMP/s1_$i.pkl"; i=$((i+1))
done
echo "[2/4] temporal weighted fusion of the four stage-1 outputs (tIoU >= 0.5)"
python src/stage1_wbf.py $TMP/stage1.pkl 0.5 $C
echo "[3/4] stage 2: V-JEPA 2.1 ViT-L video refiner(s) on the top-3 predictions, 2 passes, points averaged over refiners"
python src/refine_vjepa.py submit --cache $TMP/stage1.pkl --root "$ROOT" --split test --topk 3 --passes 2 --bs 32 --refiner "$REF" --out $TMP/refined.csv
echo "[4/4] temporal soft-NMS (a prediction overlapping a higher-scored one with tIoU >= 0.1 gets score x 0.01)"
python src/softnms_csv.py $TMP/refined.csv "$OUT" 0.1 0.01
rm -rf $TMP; echo "wrote $OUT"
if [ "${VERIFY:-0}" = "1" ] && [ -f "$EXPECTED" ]; then
  a=$(md5sum "$OUT" | cut -d' ' -f1); b=$(md5sum "$EXPECTED" | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "VERIFY: byte-identical to the submitted file ($b)" || echo "VERIFY: differs from the submitted file (ours $b, this run $a) - see README 'Reproducibility'"
fi
