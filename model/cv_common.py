"""
랭킹·분류 실험 공통 ─ 데이터 로드, 피처 풀(위치 A/B), 폴드, 지표, 피처 메타(범주·단위)
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import fe_config as C

KEYS = ["rfid_id", "fishing_code"]
# 위치 shortcut 계열 (HANDOFF §3.3-4) → Model A 에서 제외, Model B 에서만 사용
LOC_COLS = ["lon_c", "lat_c", "home_lon", "home_lat", "home_port_km", "fishing_lon", "fishing_lat"]
# 자망↔연안복합 가설 피처 (진행1 §6-2, A 재방문·B 직선 투망·C 속도 구조·D 선회·E 하루 리듬·G 이질성·H 계절)
HYP_PAT = re.compile(r"revisit|return_after|top3_cell|n_cells_200m|cross_day|site_jaccard|straight_max|"
                     r"slow_run|pca_elong|loops|active_block|max_block_gap|longest_streak|daytype|"
                     r"day_type_entropy|season_|sea_stationary|drift|speed_entropy|n_stays|stay_frac")


def set_font():
    import platform
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(
        platform.system(), "NanumGothic")  # Linux: apt-get install fonts-nanum
    plt.rcParams["axes.unicode_minus"] = False


def load_ship():
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet")
    S = S[S["fishing_code"].isin(C.CLASSES)].reset_index(drop=True)
    y = S["fishing_code"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()
    return S, y


def feature_pool(S, pool):
    cols = [c for c in S.columns if c not in KEYS and pd.api.types.is_numeric_dtype(S[c])]
    if pool == "A":
        cols = [c for c in cols if c not in LOC_COLS]
    return cols


def folds(S, y):
    sgk = StratifiedGroupKFold(C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    return list(sgk.split(np.zeros(len(y)), y, S["rfid_id"].to_numpy()))


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=list(range(len(C.CLASSES))))


def feat_meta(c):
    """피처 → (계산 단위, 대분류)  ─ 분류표는 feature_taxonomy.py 에서 직접 지정"""
    from feature_taxonomy import BASE_TO_TAX, base_of
    b, _, unit = base_of(c)
    return unit, BASE_TO_TAX[b][0]


def is_hyp(c):
    return bool(HYP_PAT.search(c)) or c.endswith("_std")  # _std: 선박 내 이질성(G)
