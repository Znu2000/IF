r"""
모델링 파이프라인 공통 설정 (피처 추출 → 랭킹 → 분류 실험)

서버에서 돌릴 때:
  export SED_DATA_ROOT=/path/to/문제\ 데이터_수정_0928
  export SED_N_JOBS=32          # 피처 추출 프로세스 수 (CPU 코어 수 - 1 권장)
  export SED_CHUNK=200000       # CSV 청크 크기 (RAM 넉넉하면 크게)
"""
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
DATA_ROOT = Path(os.environ.get("SED_DATA_ROOT", PROJECT_ROOT / "문제 데이터_수정_0928"))
VPASS_GLOB = "**/vpass_*.csv"

FEAT_DIR = Path(os.environ.get("SED_FEAT_DIR", MODEL_ROOT / "features"))   # hourly_MM / shipday / ship_features
RESULT_DIR = Path(os.environ.get("SED_RESULT_DIR", MODEL_ROOT / "results"))  # 랭킹/실험 결과
FEAT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = int(os.environ.get("SED_N_JOBS", max(1, (os.cpu_count() or 2) - 2)))
CHUNK_SIZE = int(os.environ.get("SED_CHUNK", 100_000))

CLASSES = ["연안복합", "자망", "통발", "채낚기연승낚시"]
RANDOM_STATE = 42
N_FOLDS = 5


def vpass_files():
    # 1027 버전부터 테스트셋 파일명도 vpass_*.csv → 학습 데이터에 섞이지 않게 테스트 폴더 제외
    files = sorted(f for f in DATA_ROOT.glob(VPASS_GLOB) if "테스트" not in str(f))
    if not files:
        raise FileNotFoundError(f"VPASS CSV를 찾지 못함: {DATA_ROOT / VPASS_GLOB}")
    return files


# ─────────────────────────────────────────────
# 전처리 임계값 (HANDOFF §3.2 합의안)
# ─────────────────────────────────────────────
SPIKE_MIN_M = 100.0        # A→B→A 튐: A-B 100m 초과
SPIKE_RET_RATIO = 0.25     #          & A-C 가 왕복거리의 25% 미만
SPIKE_PASSES = 2
JUMP_PT_M = 15_000.0       # 점 간 15km 초과 → 거리에서 제외
JUMP_BIN_M = 7_000.0       # 10분 구간 간 7km 초과 → 제외
N_BINS = 6                 # 1시간 = 6구간(~10분) 중앙값 평활
STATIONARY_RANGE_M = 100.0 # 평활 경로 범위 < 100m → 정박
MOVE_STEP_M = 10.0         # 방향 계산용 최소 이동 step
K_STEP = 5                 # 지터 강건 k-step (민성)
STAY_R_M = 30.0            # 체류점 반경 (3σ, σ≈10m)
STAY_MIN_PTS = 3           # 체류점 최소 점 수(≈3분)
STRAIGHT_TURN_DEG = 15.0   # 직선 투망 판정 회전각
NIGHT_HOURS = set(range(18, 24)) | set(range(0, 6))

# 선박-일 / 선박 집계
PORT_GRID_M = 500          # 항구 격자 (나연)
PORT_MIN_VESSELS = 10
HOME_GRID_M = 1000         # 모항 격자 (민성)
HOME_RADIUS_KM = 2.0
REVISIT_GRID_M = 200       # 재방문 격자 (가설 A)
REVISIT_ABSENCE_H = 2
SITE_GRID_M = 1000         # 날짜 간 어장 겹침 격자
N_DAY_TYPES = 8            # bag-of-days KMeans 군집 수 (가설 G)
