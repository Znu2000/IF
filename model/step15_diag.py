"""
[STEP15-1] 전처리 조건 변경 진단 (전처리 조건변경 판단.pdf 후보 A·D). 재처리 없이 시간 parquet 만 사용
  A 균등 간격 가정: 점 수별 ks_speed_med 와 점 간격(1분 가정) 속도 비교
    시계 검사: 직선 항해 중 앞뒤 시간이 꽉 찬(50점 이상) 시간 h 만 골라, h 의 점 경로 길이 / (h-1 끝 ~ h+1 시작 직선 거리)
               ≈ n/60 이면 점 간격이 약 1분이고 나머지는 끊김 (가정이 틀림), ≈ 1 이면 점이 60분에 고르게 퍼짐 (가정이 맞음)
    상태 변화: 속도를 60/n 배로 보정할 때 바뀌는 시간 비율
  D 소규모 정박지: 바다 정지 시간 중 그 배가 5일 이상 밤을 보낸 1km 칸(모항 제외) 안 비율
  출력: results/step15_diag_<항목>_<시각>.csv (rfid_id 없음)
실행: python step15_diag.py
"""
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C

STAMP = time.strftime("%m%d_%H%M")
BINS = [0, 5, 10, 15, 20, 30, 40, 50, 55, 65, 10000]
COLS = ["rfid_id", "part_dt", "hour", "n_points", "n_segments", "is_stationary", "ks_speed_med", "step_mean_m", "dist_pts_m",
        "gap_dist_m", "straightness", "lon_s", "lat_s", "lon_e", "lat_e"]


def state_of(st, v, hr, port):
    return np.select([port, st | (v < 0.3), v < 2, v < 5, hr >= 0.8], [0, 1, 2, 3, 4], 3)


def main():
    t0 = time.time()
    P = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(C.FEAT_DIR.glob("hourly_*.parquet"))], ignore_index=True)
    P = P[P["n_points"].notna()].sort_values(["rfid_id", "part_dt", "hour"], kind="stable").reset_index(drop=True)
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "part_dt", "hour", "state", "at_sea", "port_stay", "ks_heading_R", "home_km", "lon_c", "lat_c"])
    P = P.merge(H, on=["rfid_id", "part_dt", "hour"], how="left")
    print(f"시간 {len(P):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    sea = P["at_sea"].to_numpy() == 1
    P["bin"] = pd.cut(P["n_points"], BINS, right=True)
    S = P[sea]
    Q = S[S["is_stationary"] == 0]
    rows = Q.groupby("bin", observed=True).agg(시간=("n_points", "size"), 끊김수=("n_segments", "median"), ks속도=("ks_speed_med", "median"), 점간격m=("step_mean_m", "median"))
    rows["비율"] = rows["시간"] / rows["시간"].sum()
    rows["1분가정속도"] = rows["점간격m"] / 1852 * 60
    rows = rows.reset_index()
    rows.round(3).to_csv(C.RESULT_DIR / f"step15_diag_bins_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print("▶ 바다·비정박 시간, 점 수별\n" + rows.round(3).to_string(index=False), flush=True)
    print("바다 시간 점 수 분위:", S["n_points"].quantile([.05, .25, .5, .75, .95, .99]).round(0).to_dict(), flush=True)

    # 시계 검사
    rid, pdt, hr = P["rfid_id"].to_numpy(), P["part_dt"].to_numpy(), P["hour"].to_numpy().astype(int)
    prev_ok = np.r_[False, (rid[1:] == rid[:-1]) & (pdt[1:] == pdt[:-1]) & (hr[1:] == hr[:-1] + 1)]
    next_ok = np.r_[prev_ok[1:], False]
    i = np.flatnonzero(prev_ok & next_ok)
    n, st_, dist = P["n_points"].to_numpy(), P["straightness"].to_numpy(), P["dist_pts_m"].to_numpy()
    good = (n[i - 1] >= 50) & (n[i + 1] >= 50) & (st_[i - 1] >= 0.9) & (st_[i + 1] >= 0.9) & (dist[i - 1] >= 5000) & (dist[i + 1] >= 5000)
    i = i[good]
    span = A.hav_km(P["lon_e"].to_numpy()[i - 1], P["lat_e"].to_numpy()[i - 1], P["lon_s"].to_numpy()[i + 1], P["lat_s"].to_numpy()[i + 1]) * 1000
    Ck = pd.DataFrame({"n": n[i], "ratio": dist[i] / np.maximum(span, 1), "seg": P["n_segments"].to_numpy()[i], "gap": P["gap_dist_m"].to_numpy()[i] / np.maximum(span, 1)})
    Ck = Ck[(span > 3000) & (span < 40000)]
    Ck["bin"] = pd.cut(Ck["n"], BINS, right=True)
    T = Ck.groupby("bin", observed=True).agg(건수=("n", "size"), 점수=("n", "median"), 경로비=("ratio", "median"), 끊김거리비=("gap", "median"), 끊김수=("seg", "median"))
    T["n_60"] = T["점수"] / 60
    T = T.reset_index()
    T.round(3).to_csv(C.RESULT_DIR / f"step15_diag_clock_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print("▶ 시계 검사 (직선 항해)\n" + T.round(3).to_string(index=False), flush=True)

    # 상태 변화
    v = P["ks_speed_med"].to_numpy()
    fac = 60 / np.clip(P["n_points"].to_numpy(), 1, 60)
    st = P["is_stationary"].to_numpy() == 1
    hr_R = P["ks_heading_R"].fillna(0).to_numpy()
    port = P["port_stay"].to_numpy() == 1
    s0 = state_of(st, np.nan_to_num(v), hr_R, port)
    s1 = state_of(st, np.nan_to_num(v * fac), hr_R, port)
    names = {0: "항구", 1: "정지", 2: "표류", 3: "조업", 4: "이동"}
    X = pd.crosstab(pd.Series(s0[sea]).map(names), pd.Series(s1[sea]).map(names), normalize="index").round(3)
    X.to_csv(C.RESULT_DIR / f"step15_diag_state_{STAMP}.csv", encoding="utf-8-sig")
    print(f"▶ 60/n 보정 시 상태 변화 (행=현재, 열=보정). 바다 시간 중 바뀌는 비율 {(s0[sea] != s1[sea]).mean():.3f}\n" + X.to_string(), flush=True)
    print("현재 상태 분포:", pd.Series(s0[sea]).map(names).value_counts(normalize=True).round(3).to_dict(), flush=True)

    # D 소규모 정박지
    H2 = P[sea & P["lat_c"].notna()][["rfid_id", "part_dt", "hour", "lon_c", "lat_c", "state", "is_stationary", "home_km"]].copy()
    H2["cell"] = A.grid(H2["lon_c"], H2["lat_c"], 1000)
    night = H2[H2["hour"].isin([22, 23, 0, 1, 2, 3]) & H2["state"].isin([1])]
    nights = night.groupby(["rfid_id", "cell"])["part_dt"].nunique().rename("nights").reset_index()
    anchor = nights[nights["nights"] >= 5]
    stop = H2[H2["state"] == 1].merge(anchor, on=["rfid_id", "cell"], how="left")
    share = stop["nights"].notna().mean()
    print(f"▶ D: 바다 정지 {len(stop):,}시간 중 그 배가 5일 이상 밤을 보낸 1km 칸 안 {share:.3f} / 그런 칸이 있는 배 {anchor['rfid_id'].nunique():,}척 / 칸 {len(anchor):,}", flush=True)
    print(f"   모항 10km 안 비율 {stop.loc[stop['nights'].notna(), 'home_km'].lt(10).mean():.3f}", flush=True)
    fs = P[sea]["state"].value_counts(normalize=True).round(3).to_dict()
    print("저장된 상태 분포:", fs, flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
