#!/usr/bin/env bash
# v2 피처(신규 범주) 실험 일괄 실행: (시간 피처 추출 완료 대기) → 집계 → 랭킹 → 핵심 모델 → v1 비교
# 튜닝 LightGBM·TabM 은 제외 (튜닝 효과 없음, TabM 은 GPU 서버용)
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
LOG=results/run_v2.log
until [ "$(grep -c '✅' features/fe_hourly.log 2>/dev/null)" -ge 6 ]; do sleep 60; done
echo "[$(date +%T)] 시간 피처 추출 완료 → 집계" | tee $LOG
python fe_aggregate.py 2>&1 | grep -v -i warning | tee -a $LOG
echo "[$(date +%T)] 랭킹" | tee -a $LOG
python rank_features.py 2>&1 | grep -v -i warning | tee results/rank.log | tee -a $LOG
echo "[$(date +%T)] 모델 (A, B)" | tee -a $LOG
python train_classifiers.py --pools A B --models base rf lgbm_base catboost hier_hyp --tag v2 2>&1 \
  | grep -v -i warning | tee results/train_v2.log | tee -a $LOG
echo "[$(date +%T)] v1 비교" | tee -a $LOG
python compare_versions.py 2>&1 | grep -v -i warning | tee -a $LOG
echo "[$(date +%T)] ✅ v2 완료" | tee -a $LOG
