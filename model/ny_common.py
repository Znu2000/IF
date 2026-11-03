"""
[NY] 나연 실험 공통 ─ 데이터 로드와 LightGBM 설정 (팀 lgbm_base 와 같은 값)
  규칙(팀장 지침): 모델 학습은 2026 항적만. 2025 테스트 항적은 '검증'에만 사용.
  2025 검증 = 2026 학습 선박 중 2025 테스트셋에도 있는 배(테스트A 100척 제외)의 2025 항적 + 2026 라벨
             (선박의 업종이 해마다 같다고 가정). 예측·제출에는 ID 라벨을 쓰지 않는다.
"""
import openpyxl
import pandas as pd

import fe_config as C

LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced", max_bin=63,
           verbose=-1, n_jobs=C.N_JOBS, random_state=0)
SAMPLE = C.PROJECT_ROOT / "문제 데이터_수정_1027" / "테스트셋 및 결과제출 샘플" / "제출샘플.xlsx"


def test_a_ids():
    ws = openpyxl.load_workbook(SAMPLE).worksheets[0]
    return {str(r[0]).strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[0]}


def load():
    """S: 2026 학습 선박 피처 / T25: 검증용 2025 항적(2026 라벨) / TA: 테스트A 선박 2025 피처"""
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet")
    S = S[S["fishing_code"].isin(C.CLASSES)].copy()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    lab = S.set_index("rfid_id")["fishing_code"]
    ida = test_a_ids()
    T["rid"] = T["rfid_id"].astype(str)
    T25 = T[T["rfid_id"].isin(lab.index) & ~T["rid"].isin(ida)].copy()
    T25["fishing_code"] = T25["rfid_id"].map(lab)
    print(f"2026 학습 {len(S):,}척 / 2025 테스트 {len(T):,}척 → 검증용 2025 표본 {len(T25):,}척 (테스트A 제외)")
    S["year"], T25["year"] = 2026, 2025
    return S, T25.drop(columns="rid"), T[T["rid"].isin(ida)].drop(columns="rid")
