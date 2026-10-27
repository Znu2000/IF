#!/usr/bin/env bash
# GPU 서버에서 TabM(SOTA) Pool A 실행 → 로컬 결과와 앙상블 합치기
#
# 서버로 복사할 것 (원본 CSV 불필요):
#   model/*.py
#   model/features/ship_features.parquet     (선박 7,372 × 348)
#   model/results/fold_rankings.json         (fold 별 피처 순위 + k*)
#   model/results/oof_A_*.npy                (로컬에서 돌린 Pool A 모델 OOF, 앙상블용)
#
# 설치: pip install pandas numpy pyarrow scikit-learn lightgbm catboost optuna matplotlib tabm rtdl_num_embeddings
#       torch 는 서버 CUDA 버전에 맞는 휠로 (https://pytorch.org/get-started/locally/)
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 1) TabM Pool A (피처 ~250개, 로컬 CPU 로는 2시간+ 예상 → GPU)
python train_classifiers.py --pools A --models tabm --device cuda --tag A_tabm | tee results/train_A_tabm.log

# 2) 저장된 OOF(로컬 lgbm/catboost/hier_hyp + 방금 tabm)로 앙상블만 재계산
python train_classifiers.py --pools A --models --reuse-oof --tag A_final | tee results/train_A_final.log

# (선택) TabPFN: Hugging Face 라이선스 동의·로그인을 본인이 먼저 한 뒤
# python train_classifiers.py --pools A B --models tabpfn --tabpfn --device cuda --tag tabpfn
