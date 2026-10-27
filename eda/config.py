"""
EDA 공통 설정
- 데이터 경로 / 캐시 경로 / 샘플링 비율 / 한글 폰트
- 다른 컴퓨터(서버, Colab 등)에서 돌릴 때는 DATA_ROOT 만 바꾸면 됨
  (또는 환경변수 SED_DATA_ROOT 로 지정)
"""
import os
import sys
import glob
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 서버(GUI 없음)에서도 그림 저장 가능
import matplotlib.pyplot as plt

try:  # 한글 로그 깨짐 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─────────────────────────────────────────────
# 1. 경로
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 2026-09-28 수정본 (1월 rfid_time 오류값 / 4월 중복행 수정됨)
DATA_ROOT = Path(os.environ.get("SED_DATA_ROOT", PROJECT_ROOT / "문제 데이터_수정_0928"))

# 하위 폴더 구조가 달라도 찾을 수 있도록 재귀 검색
VPASS_GLOB = str(DATA_ROOT / "**" / "vpass_*.csv")
ACCIDENT_GLOB = str(DATA_ROOT / "**" / "*.xlsx")

EDA_ROOT = Path(__file__).resolve().parent
CACHE_DIR = EDA_ROOT / "cache"      # 00_build_cache.py 결과물 (parquet)
OUT_DIR = EDA_ROOT / "outputs"      # 그림/HTML 결과물
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def vpass_files():
    files = sorted(glob.glob(VPASS_GLOB, recursive=True))
    if not files:
        raise FileNotFoundError(f"VPASS CSV를 찾지 못함: {VPASS_GLOB}")
    return files


def accident_file():
    files = sorted(glob.glob(ACCIDENT_GLOB, recursive=True))
    if not files:
        raise FileNotFoundError(f"사고 엑셀을 찾지 못함: {ACCIDENT_GLOB}")
    return files[0]


# ─────────────────────────────────────────────
# 2. 샘플링 / 처리 파라미터  (★ 메모리·시간 조절은 여기서)
# ─────────────────────────────────────────────
CHUNK_SIZE = 500_000          # CSV 청크 크기 (행). RAM 8GB면 200_000 권장

# 선박(rfid_id) 단위 해시 샘플링 → 선택된 선박은 6개월 전체 항적이 다 들어감
FEATURE_SAMPLE_PCT = 5        # 행 단위 피처 계산할 선박 비율(%)  → 약 100만 행
POINT_SAMPLE_PCT = 1          # 좌표(점) 단위까지 저장할 선박 비율(%) → 시각화/클러스터링용

# 사고 매칭 후보 추출: 사고 지점 반경 / 날짜 범위
ACC_PREFILTER_KM = 30         # 1차 필터 (항적 첫 점 기준, 넉넉하게)
ACC_TIME_WINDOW_H = 12        # 사고시각 ±N시간 행만 후보로 저장 (사고 전 행동 분석용)

CLASSES = ["연안복합", "자망", "통발", "채낚기연승낚시"]
CLASS_COLORS = {
    "연안복합": "#4C72B0",
    "자망": "#DD8452",
    "통발": "#55A868",
    "채낚기연승낚시": "#C44E52",
}

RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 3. 한글 폰트
# ─────────────────────────────────────────────
def set_korean_font():
    system = platform.system()
    if system == "Windows":
        matplotlib.rc("font", family="Malgun Gothic")
    elif system == "Darwin":
        matplotlib.rc("font", family="AppleGothic")
    else:  # Linux / Colab:  apt-get install fonts-nanum 후 사용
        matplotlib.rc("font", family="NanumGothic")
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110


def savefig(name, fig=None):
    fig = fig or plt.gcf()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")
    return path
