#!/usr/bin/env bash
# 서버: 1단계 실험 (지역 사전확률 / 학습 목표 / 연도 차이 피처). CPU 작업 (LightGBM·CatBoost), GPU 불필요
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export SED_N_JOBS=${SED_N_JOBS:-$(nproc)}
python step1_experiments.py 2>&1 | tee results/step1.log
