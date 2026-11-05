"""
[STEP10] "동일 선박의 과거 항적을 이용한 통계·행동 프로필 생성 금지" 공지 대비 피처 점검 (학습 없음, 이름 규칙 분류)
  등급 C : 여러 날에 걸친 그 배의 습관 (재방문·출어 주기·요일/계절·어장 고정성·평소 위치) → 좁게 해석해도 걸릴 수 있음
  등급 B : 6개월 항적을 통째로 요약한 행동 통계 (일별·시간별 값의 평균·분위수·상태 전이 등) → 넓게 해석하면 걸림
  등급 A : 다른 배와의 관계 (혼잡도) → 자기 항적만의 프로필은 아니나 위치는 자기 항적에서 옴
  상위 k 는 fold 별 목록(fold_rankings.json)의 합집합 기준 사용 빈도로 집계
  출력: results/step10_profile_features.csv, step10_profile_summary.csv (rfid_id 없음)
실행: python step10_profile_check.py
"""
import json
import re

import pandas as pd
import pyarrow.parquet as pq

import fe_config as C

SKIP = {"rfid_id", "fishing_code"}
HABIT = re.compile(r"^(rv_|pd_|db_|season_|daytype_|day_type|cross_day_revisit|site_jaccard|revisit_cell|return_after_absence|top3_cell"
                   r"|n_cells_200m|n_days|n_months|n_moving_days|moving_days|fishing_area|area_range|lon_c|lat_c|home_|fishing_lon|fishing_lat"
                   r"|moon_|spring_|night_bright|moving_illum|moving_spring)")
OTHERS = re.compile(r"^cg_")


def grade(c):
    if OTHERS.match(c):
        return "A"
    return "C" if HABIT.match(c) else "B"


def main():
    cols = [c for c in pq.ParquetFile(C.FEAT_DIR / "ship_features.parquet").schema_arrow.names if c not in SKIP]
    beh = [c for c in pq.ParquetFile(C.FEAT_DIR / "behav_feats.parquet").schema_arrow.names if c not in SKIP]
    D = pd.DataFrame({"feature": cols + beh})
    D["등급"] = D["feature"].map(grade)
    R = json.load(open(C.RESULT_DIR / "fold_rankings.json", encoding="utf-8"))["B"]
    k = R["k_star"]
    n = len(R["folds"])
    use = pd.Series([c for f in R["folds"] for c in f[:k]]).value_counts() / n
    D["top_k_빈도"] = D["feature"].map(use).fillna(0.0)
    add = D["feature"].str.match(r"^(rv_|pd_)")          # r4_rp 는 rv_·pd_ 를 상위 k 와 별도로 전부 넣음
    D["r4_rp_사용"] = (D["top_k_빈도"] > 0) | add
    D["all_rp_사용"] = ~D["feature"].str.match(r"^(cg_|db_)")
    D.sort_values(["등급", "top_k_빈도"], ascending=[False, False]).round(3).to_csv(
        C.RESULT_DIR / "step10_profile_features.csv", index=False, encoding="utf-8-sig")
    S = D.groupby("등급").agg(전체=("feature", "size"), r4_rp=("r4_rp_사용", "sum"), all_rp=("all_rp_사용", "sum"),
                              top_k_평균개수=("top_k_빈도", "sum")).reset_index()
    S.round(1).to_csv(C.RESULT_DIR / "step10_profile_summary.csv", index=False, encoding="utf-8-sig")
    print(f"k = {k}")
    print(S.round(1).to_string(index=False))
    print("\n등급 C 중 상위 k 자주 쓰는 피처:")
    print(D[(D["등급"] == "C") & (D["top_k_빈도"] > 0)].sort_values("top_k_빈도", ascending=False).head(25)[["feature", "top_k_빈도"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
