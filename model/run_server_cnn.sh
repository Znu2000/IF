#!/usr/bin/env bash
# GPU 서버: 1D-CNN + MIL 실행 (cnn_mil.py). 압축 해제한 server_cnnmil/model 폴더 안에서 실행
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
# GPU 0 은 다른 작업(vLLM)이 사용 중 → 기본 GPU 1. 바꾸려면 CUDA_VISIBLE_DEVICES=2 bash run_server_cnn.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python cnn_mil.py --device cuda --seeds 3 2>&1 | tee results/cnnmil.log
