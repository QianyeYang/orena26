#!/bin/bash
#SBATCH --job-name=frame-v2-requant
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/datasets/engs2732/orena/track-frame/v2/logs/requant-%j.log
#SBATCH --error=/datasets/engs2732/orena/track-frame/v2/logs/requant-%j.log

# Re-quantize an ALREADY-MERGED model, skipping the merge.
#
# Exists because the shipped int8-mlp export was written as a single 35 GiB
# model.safetensors (transformers' default max_shard_size is ~50 GB, so nothing
# under that gets split). transformers mmaps a checkpoint file whole, so the
# submission container died with:
#
#   RuntimeError: unable to mmap 37613882728 bytes from
#   </opt/app/resources/model/model.safetensors>: Cannot allocate memory (12)
#
# export_merged.py now passes max_shard_size=4GB. Re-running the full export
# would redo a merge we already have on disk, so this quantize-only path points
# --base at the merged directory instead.
#
# NOTE: --base must be the MERGED model, not the original base. export_merged.py
# quantizes `merged or base`, so omitting --out-merged while pointing --base at
# the original weights would silently drop the LoRA.
#
# Usage: sbatch requantize.slurm <merged-model-dir> <out-dir>
#   sbatch requantize.slurm \
#     os-models/Qwen3.6-27B-frame-ep24-merged \
#     os-models/Qwen3.6-27B-frame-ep24-int8mlp-sharded
set -euo pipefail
REPO=/datasets/engs2732/orena
source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

MERGED="${1:?merged model dir is required}"
OUT="${2:?output dir is required}"

[[ -f "$REPO/$MERGED/config.json" ]] || { echo "not a model dir: $MERGED" >&2; exit 2; }
[[ -e "$REPO/$OUT" ]] && { echo "refusing to overwrite existing $OUT" >&2; exit 2; }

echo "requantize: merged=$MERGED out=$OUT host=$(hostname)"

python track-frame/v2/src/export_merged.py \
    --base "$MERGED" \
    --out-quant "$OUT"

echo "=== shard layout ==="
ls -la "$REPO/$OUT"/*.safetensors* | sed 's/^/  /'
python - "$REPO/$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
shards = sorted(out.glob("*.safetensors"))
biggest = max((f.stat().st_size for f in shards), default=0)
print(f"shards={len(shards)} largest={biggest/2**30:.2f} GiB "
      f"total={sum(f.stat().st_size for f in shards)/2**30:.2f} GiB")
index = out / "model.safetensors.index.json"
print(f"index present: {index.is_file()}")
if index.is_file():
    weight_map = json.loads(index.read_text())["weight_map"]
    print(f"index maps {len(weight_map)} tensors over {len(set(weight_map.values()))} files")
if biggest > 8 * 2**30:
    raise SystemExit(f"largest shard {biggest/2**30:.2f} GiB is too big to be safe")
PY
echo "done -> $OUT"
