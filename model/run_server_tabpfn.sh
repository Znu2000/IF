#!/usr/bin/env bash
# 서버: TabPFN v2 비교 (GPU). server_cnnmil/model 안에서 실행
#   TABPFN_CKPT : 가중치 경로 (기본 ~/IF/tabpfn-v2-classifier.ckpt)
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
CKPT=${TABPFN_CKPT:-$HOME/IF/tabpfn-v2-classifier.ckpt}
# 0) 가중치 로딩 확인 (작은 데이터로 1초 안팎)
python -c "
from tabpfn import TabPFNClassifier; import numpy as np
X=np.random.rand(200,5); y=(X[:,0]>.5).astype(int)
m=TabPFNClassifier(model_path='$CKPT', device='cuda', n_estimators=2).fit(X,y)
print('가중치 로딩 OK, 정확도', (m.predict(X)==y).mean())"
python tabpfn_exp.py --ckpt "$CKPT" --device cuda 2>&1 | tee results/tabpfn.log
echo "✅ 전체 완료"
