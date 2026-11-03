#!/usr/bin/env bash
# 서버: 지역 정보 확장 (A 여러 범위 / B 같은 항구 / C 조업 격자 / D 거리 가중 / all). CPU 작업, GPU 불필요
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export SED_N_JOBS=${SED_N_JOBS:-$(nproc)}
python step3_region.py 2>&1 | tee results/step3.log
echo "✅ 전체 완료"
