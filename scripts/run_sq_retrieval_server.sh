#!/bin/bash
# Start the SearchQA E5 retrieval server (LatentSkill's memory-safe numpy server).
#
# The 64.56 GB faiss Flat index is NOT resident in memory: the server memmaps it and
# does chunked numpy inner-product search (RSS ~1-2 GB). It exposes /retrieve and
# /retrieve_batch (one index scan for all pending queries of a wave).
#
# Index provenance: assembled from the two released parts of the E5 faiss Flat
# index (part_aa + part_ab = 64559075373 B = 45-byte faiss header +
# 21,015,324 x 768 float32); place it at $LS/wiki_index/e5_Flat.index.
#
#   usage: GPU=1 bash scripts/run_sq_retrieval_server.sh
set -u
LS=${LATENTSKILL_ROOT:-./LatentSkill}
PY=${PY:-python}
GPU=${GPU:-1}
PORT=${PORT:-8200}
LOG=${LOG:-logs/sq_retrieval_server.log}

mkdir -p "$(dirname "$LOG")"
cd "$LS"
echo "[retrieval] index=$LS/wiki_index/e5_Flat.index -> $(readlink -f "$LS/wiki_index/e5_Flat.index")"
echo "[retrieval] port=$PORT gpu=$GPU log=$LOG"

# keep the datasets arrow cache in fast local storage
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/dev/shm/hf_datasets_cache}
export HF_HOME=${HF_HOME:-/dev/shm/hf_home}
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  setsid nohup "$PY" -u evals_vllm/retrieval_server_np.py \
  > "$LOG" 2>&1 &
echo $! > logs/sq_retrieval_server.pid
echo "[retrieval] pid $(cat logs/sq_retrieval_server.pid)"
