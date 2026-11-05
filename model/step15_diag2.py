"""
[STEP15-2] 보고 주기 진단 ─ 시간당 점 수가 "끊김"인지 "배마다 다른 보고 주기"인지 구분
  배별 평소 점 수 = 바다·비정박 시간 n_points 의 75% 분위. 채움률 = n / 평소 점 수
  1) 배별 평소 점 수 분포와 업종별 차이
  2) 한 배 안에서 평소 점 수가 얼마나 일정한가 (시간별 n 의 75%/25% 비)
  3) 시계 검사 (배 기준): 직선 항해, 앞뒤 시간 채움률 0.8 이상. 경로비 ≈ 채움률 이면 그 시간은 잘린 것, ≈ 1 이면 고르게 퍼진 것
  4) 같은 실제 속도(10분 구간 sp_p50)에서 보고 주기별 ks_speed_med 와 상태 ─ 점 개수 기준 창(k=5)의 시간 척도 차이
  5) 평소 점 수 기준 보정 시 상태 변화
  출력: results/step15_diag2_<항목>_<시각>.csv (rfid_id 없음)
실행: python step15_diag2.py
"""
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C
import step15_diag as D0

STAMP = time.strftime("%m%d_%H%M")
RATE_BINS = [0, 15, 20, 30, 45, 60, 80, 100, 10000]
FILL_BINS = [0, 0.25, 0.5, 0.75, 0.9, 1.1, 100]
SP_BINS = [0, 0.3, 1, 2, 3, 5, 8, 100]
COLS = D0.COLS + ["sp_p50"]


def main():
    t0 = time.time()
    P = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(C.FEAT_DIR.glob("hourly_*.parquet"))], ignore_index=True)
    P = P[P["n_points"].notna()].sort_values(["rfid_id", "part_dt", "hour"], kind="stable").reset_index(drop=True)
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "part_dt", "hour", "at_sea", "port_stay", "ks_heading_R"])
    P = P.merge(H, on=["rfid_id", "part_dt", "hour"], how="left")
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code"])
    M = P[(P["at_sea"] == 1) & (P["is_stationary"] == 0)]
    g = M.groupby("rfid_id")["n_points"]
    R = pd.DataFrame({"rate": g.quantile(0.75), "q25": g.quantile(0.25), "hours": g.size()}).reset_index()
    R = R[R["hours"] >= 20].merge(S, on="rfid_id", how="left")
    R["spread"] = R["rate"] / R["q25"].clip(lower=1)
    R["bin"] = pd.cut(R["rate"], RATE_BINS)
    T = pd.crosstab(R["bin"], R["fishing_code"])
    T["척"] = T.sum(1)
    T["q75/q25 중앙"] = R.groupby("bin", observed=True)["spread"].median().round(2)
    T.to_csv(C.RESULT_DIR / f"step15_diag2_rate_{STAMP}.csv", encoding="utf-8-sig")
    print("▶ 배별 평소 점 수 (시간당)\n" + T.to_string(), flush=True)

    rate = P["rfid_id"].map(R.set_index("rfid_id")["rate"]).to_numpy()
    fill = P["n_points"].to_numpy() / rate
    n, st_, dist = P["n_points"].to_numpy(), P["straightness"].to_numpy(), P["dist_pts_m"].to_numpy()
    rid, pdt, hr = P["rfid_id"].to_numpy(), P["part_dt"].to_numpy(), P["hour"].to_numpy().astype(int)
    prev_ok = np.r_[False, (rid[1:] == rid[:-1]) & (pdt[1:] == pdt[:-1]) & (hr[1:] == hr[:-1] + 1)]
    i = np.flatnonzero(prev_ok & np.r_[prev_ok[1:], False])
    good = (fill[i - 1] >= 0.8) & (fill[i + 1] >= 0.8) & (st_[i - 1] >= 0.9) & (st_[i + 1] >= 0.9) & (dist[i - 1] >= 5000) & (dist[i + 1] >= 5000)
    i = i[good]
    span = A.hav_km(P["lon_e"].to_numpy()[i - 1], P["lat_e"].to_numpy()[i - 1], P["lon_s"].to_numpy()[i + 1], P["lat_s"].to_numpy()[i + 1]) * 1000
    Ck = pd.DataFrame({"fill": fill[i], "ratio": dist[i] / np.maximum(span, 1), "rate": rate[i], "span": span})
    Ck = Ck[(Ck["span"] > 3000) & (Ck["span"] < 40000)]
    Ck["채움률"] = pd.cut(Ck["fill"], FILL_BINS)
    Ck["주기"] = np.where(Ck["rate"] <= 30, "저빈도(<=30)", "고빈도(>30)")
    Tc = Ck.groupby(["주기", "채움률"], observed=True).agg(건수=("fill", "size"), 채움=("fill", "median"), 경로비=("ratio", "median")).reset_index()
    Tc.round(3).to_csv(C.RESULT_DIR / f"step15_diag2_clock_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print("▶ 시계 검사 (배 기준)\n" + Tc.round(3).to_string(index=False), flush=True)

    sea = P["at_sea"].to_numpy() == 1
    print("▶ 바다 시간 채움률 분포:", pd.cut(pd.Series(fill[sea]), FILL_BINS).value_counts(normalize=True, sort=False).round(3).to_dict(), flush=True)

    Q = P[sea & (P["is_stationary"].to_numpy() == 0)].copy()
    Q["rate"] = rate[sea & (P["is_stationary"].to_numpy() == 0)]
    Q = Q[(Q["n_points"] >= 0.8 * Q["rate"])]
    Q["주기"] = pd.cut(Q["rate"], [0, 30, 60, 10000], labels=["<=30", "30~60", ">60"])
    Q["실제속도"] = pd.cut(Q["sp_p50"], SP_BINS)
    Tv = Q.pivot_table(index="실제속도", columns="주기", values="ks_speed_med", aggfunc="median", observed=True).round(2)
    Tn = Q.pivot_table(index="실제속도", columns="주기", values="n_points", aggfunc="size", observed=True)
    Tv.to_csv(C.RESULT_DIR / f"step15_diag2_speed_{STAMP}.csv", encoding="utf-8-sig")
    print("▶ 같은 10분 구간 속도에서 ks_speed_med (채움 시간만)\n" + Tv.to_string() + "\n시간 수\n" + Tn.to_string(), flush=True)

    st = P["is_stationary"].to_numpy() == 1
    hr_R = P["ks_heading_R"].fillna(0).to_numpy()
    port = P["port_stay"].to_numpy() == 1
    v = np.nan_to_num(P["ks_speed_med"].to_numpy())
    fac = np.clip(np.nan_to_num(1 / fill, nan=1.0), 1, 20)
    s0 = D0.state_of(st, v, hr_R, port)
    s1 = D0.state_of(st, v * fac, hr_R, port)
    s2 = D0.state_of(st, np.nan_to_num(P["sp_p50"].to_numpy()), hr_R, port)
    names = {0: "항구", 1: "정지", 2: "표류", 3: "조업", 4: "이동"}
    print(f"▶ 평소 점 수 기준 보정 시 바뀌는 바다 시간 {(s0[sea] != s1[sea]).mean():.3f} / sp_p50 기준이면 {(s0[sea] != s2[sea]).mean():.3f}", flush=True)
    print(pd.crosstab(pd.Series(s0[sea]).map(names), pd.Series(s1[sea]).map(names), normalize="index").round(3).to_string(), flush=True)

    low = rate <= 30
    for name, m in [("<=30", low), (">30", ~low & ~np.isnan(rate))]:
        print(f"상태 분포 주기 {name}:", pd.Series(s0[sea & m]).map(names).value_counts(normalize=True).round(3).to_dict(), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
