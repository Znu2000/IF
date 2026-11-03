"""
[STEP9-5] 라벨 품질 점검 + 지역 한정 재평가
  (1) 위약 검정: 사고 시각만 ±7·14일 옮겨 같은 위치·업종으로 같은 규칙(등급 A)을 적용.
      사고가 없는 날에도 A 가 나오는 비율 = 근처를 지나던 배를 사고 선박으로 잘못 고르는 비율의 추정치
  (2) 지역 한정: 주최 측 답변대로 사고현황은 5개 해경서 관할만 포함 → 정상 비교군을 사고 발생 범위(±0.2도)에 모항이 있는 배로 한정해
      선박 판별 AUC 를 다시 잼
  출력: results/step9_placebo.csv, step9_region_auc.csv
실행: python step9_placebo.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fe_aggregate as A
import fe_config as C
import step9_detect as D
import step9_labels as L9

SHIFTS = [-14, -7, 0, 7, 14]
SURE = ["AS", "A", "S"]


def tier_a(P, acc, shift):
    rows = []
    for _, a in acc.iterrows():
        t = a["발생일시"] + pd.Timedelta(days=shift)
        gear = L9.GEAR.get(a["어업면허"], a["어업면허"])
        W = P[(P["gear"] == gear) & (P["ts"] >= t.floor("h") - pd.Timedelta(hours=2)) & (P["ts"] <= t.floor("h") + pd.Timedelta(hours=2))]
        ok_a, d1 = False, np.inf
        if not W.empty:
            lon = W[[f"b_lon{j}" for j in range(6)]].to_numpy()
            lat = W[[f"b_lat{j}" for j in range(6)]].to_numpy()
            tm = W["ts"].to_numpy()[:, None] + (np.arange(6) * 10 + 5).astype("timedelta64[m]")
            ok = np.abs((tm - np.datetime64(t)) / np.timedelta64(1, "m")) <= L9.NEAR_MIN
            d = np.where(ok & ~np.isnan(lon), A.hav_km(lon, lat, a["경도(변환)"], a["위도(변환)"]), np.inf)
            per = pd.Series(d.min(1), index=W["rid"].to_numpy()).groupby(level=0).min().sort_values()
            per = per[np.isfinite(per)]
            if len(per):
                d1, d2 = per.iloc[0], (per.iloc[1] if len(per) > 1 else np.inf)
                ok_a = bool(d1 <= 1.0 and (d2 >= 2 * d1 or d2 > 3))
        rows.append(dict(shift=shift, group=L9.group(a), a=ok_a, near1=d1 <= 1.0))
    return rows


def main():
    acc = pd.read_excel(L9.ACC)
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code", "home_lat", "home_lon"])
    S["rid"] = S["rfid_id"].astype(str)
    gear = dict(zip(S["rid"], S["fishing_code"]))

    # (1) 옮긴 시각까지 모두 포함하도록 시간대를 넓혀 읽음 (범위를 벗어나는 날짜는 제외)
    shifted = pd.concat([acc.assign(발생일시=acc["발생일시"] + pd.Timedelta(days=s)) for s in SHIFTS])
    shifted = shifted[shifted["발생일시"].between("2026-01-01", "2026-06-30 23:59")]
    P = L9.load_points(shifted)
    P["gear"] = P["rid"].map(gear)
    rows = []
    for s in SHIFTS:
        ok = (acc["발생일시"] + pd.Timedelta(days=s)).between("2026-01-01", "2026-06-30 23:59")
        rows += tier_a(P, acc[ok], s)
    R = pd.DataFrame(rows)
    R["kind"] = np.where(R["shift"] == 0, "실제 시각", "옮긴 시각")
    T = R.groupby(["kind", "group"]).agg(n=("a", "size"), a_rate=("a", "mean"), near1_rate=("near1", "mean")).reset_index()
    T.round(4).to_csv(C.RESULT_DIR / "step9_placebo.csv", index=False, encoding="utf-8-sig")
    print(T.round(3).to_string(index=False))

    # (2) 지역 한정 선박 판별
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=D.COLS)
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.sort_values(["rid", "ts"], kind="stable").reset_index(drop=True)
    F = D.build(H)
    s = np.where(F["onset"], 1 + F["f_drift"] / 6, 0.0)
    vmax = pd.Series(s).groupby(F["rid"].to_numpy()).max()
    act = F.groupby("rid")["sea"].sum()
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv")
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    pos = L[L["tier"].isin(SURE)].drop_duplicates("rid")["rid"].to_numpy()
    box = (S["home_lat"].between(acc["위도(변환)"].min() - 0.2, acc["위도(변환)"].max() + 0.2)
           & S["home_lon"].between(acc["경도(변환)"].min() - 0.2, acc["경도(변환)"].max() + 0.2))
    in_box = set(S.loc[box, "rid"])
    all_neg = np.array(sorted(set(F["rid"]) - touched))
    reg_neg = np.array([r for r in all_neg if r in in_box])
    out = []
    for name, neg in [("전체 정상", all_neg), ("5개 관할 지역 정상", reg_neg)]:
        vid = np.r_[pos, neg]
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        a = act.reindex(vid).fillna(0).to_numpy()
        v = vmax.reindex(vid).fillna(0).to_numpy()
        out.append(dict(비교군=name, n_neg=len(neg), 활동량만_auc=roc_auc_score(y, a), 급정지_auc=roc_auc_score(y, v),
                        급정지_활동량보정_auc=D.strat_auc(y, v, a)))
    O = pd.DataFrame(out)
    O.round(4).to_csv(C.RESULT_DIR / "step9_region_auc.csv", index=False, encoding="utf-8-sig")
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
