"""
[ANOM-2] 사고현황(선박 ID 없음) → 항적 매칭으로 약한 라벨 만들기 (검증 전용, 학습에 쓰지 않음)
  사고 1건마다 발생 시각 ±2시간 안의 시간대 기록 중
    · 업종 일치 (어업면허 → fishing_code, 연승 → 채낚기연승낚시)
    · 사고 위치와 그 시간 항적(시작·끝·중심점) 최소 거리
  신뢰 등급
    A : 가장 가까운 같은 업종 배가 1.5km 안이고, 두 번째 배보다 2배 이상 가깝거나 3km 안에 혼자
    B : 같은 업종 배가 3km 안 (애매)
  라벨 시각 = 사고 시각(정시 내림)에 그 배 기록이 있으면 그 시각, 없으면 직전 기록 시각
  출력 : results/anom_labels.csv  (실제 rfid_id 포함 → 깃헙 업로드 금지)
실행: python anomaly_labels.py
"""
import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C

ACC = C.PROJECT_ROOT / "문제 데이터_수정_1027" / "26년 1~6월 어선 사고현황(테스트 데이터).xlsx"
TYPE = {"연승": "채낚기연승낚시"}


def main():
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet",
                        columns=["rfid_id", "fishing_code", "ts", "lon_s", "lat_s", "lon_e", "lat_e", "lon_c", "lat_c"])
    acc = pd.read_excel(ACC)
    rows = []
    for i, a in acc.iterrows():
        t0 = a["발생일시"].floor("h")
        w = H[(H["ts"] >= t0 - pd.Timedelta(hours=2)) & (H["ts"] <= t0 + pd.Timedelta(hours=2))]
        typ = TYPE.get(a["어업면허"], a["어업면허"])
        w = w[w["fishing_code"] == typ]
        base = dict(acc_idx=i, acc_time=a["발생일시"], acc_type=a["발생유형"], gear=typ, area=a["발생해역"],
                    lat=a["위도(변환)"], lon=a["경도(변환)"])
        if w.empty:
            rows.append(base | dict(tier="none"))
            continue
        d = np.min([A.hav_km(w[f"lon_{s}"], w[f"lat_{s}"], a["경도(변환)"], a["위도(변환)"]) for s in "sec"], axis=0)
        per = pd.Series(d, index=w.index).groupby(w["rfid_id"]).min().sort_values()
        d1, d2 = per.iloc[0], (per.iloc[1] if len(per) > 1 else np.inf)
        tier = "A" if d1 <= 1.5 and (d2 >= 2 * d1 or d2 > 3) else ("B" if d1 <= 3 else "none")
        rid = per.index[0]
        own = H[(H["rfid_id"] == rid) & (H["ts"] <= t0 + pd.Timedelta(hours=1))]["ts"]
        lab = t0 if (own == t0).any() else (own.max() if len(own) else pd.NaT)
        rows.append(base | dict(tier=tier, rfid_id=rid, d1_km=round(d1, 3), d2_km=round(d2, 3),
                                n_cand=len(per), label_ts=lab))
    L = pd.DataFrame(rows)
    L["rfid_id"] = L["rfid_id"].astype("Int64")
    L.to_csv(C.RESULT_DIR / "anom_labels.csv", index=False, encoding="utf-8-sig")
    print(L["tier"].value_counts().to_string())
    print(L[L["tier"] == "A"].groupby("acc_type").size().sort_values(ascending=False).to_string())
    print(L[L["tier"] == "A"]["area"].value_counts().to_string())


if __name__ == "__main__":
    main()
