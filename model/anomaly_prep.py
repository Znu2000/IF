"""
[ANOM-1] 이상탐지용 시간대 테이블 ─ 시간(L1) 피처 + 상태 + 앞뒤 시간 문맥
  입력 : features/hourly_MM.parquet (학습) / test_hourly_MM.parquet (--test)
  출력 : features/anom_hourly.parquet / anom_hourly_test.parquet  (선박·시각 1행)

  문맥 피처 (사고 직전·직후에 달라질 것으로 보는 것)
    gap_prev_h / gap_next_h  : 앞뒤 기록까지 시간 (사고 후 신호 끊김, 마지막 기록이면 999)
    d_speed                  : 직전 시간 대비 속도 변화 (급정지)
    jump_prev_km             : 직전 시간 끝점 → 이번 시작점 거리 (위치 튐)
    at_sea                   : 항구 정박이 아니고 공용 항구에서 1km 이상
    hod_rarity / cell_rarity : 그 배가 평소 잘 안 나오는 시각·1km 격자 (1 - 비중)
    home_km_delta            : 모항 거리 변화 (사고 뒤 예인·귀항이면 계속 줄어듦)
  항구 격자는 학습에서 만든 port_cells.csv 를 학습·테스트 모두 고정 사용

실행: python anomaly_prep.py [--test]
"""
import argparse
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C

HOUR_COLS = ["part_dt", "rfid_id", "hour", "fishing_code", "n_points", "n_segments", "gap_count", "spike_ratio",
             "n_jumps", "lon_c", "lat_c", "lon_s", "lat_s", "lon_e", "lat_e", "rg_m", "dist_m", "net_disp_m",
             "straightness", "sp_mean", "sp_std", "sp_p50", "sp_p90", "sp_max", "sp_change_mean", "frac_stop",
             "frac_slow", "frac_fast", "n_stop_move_switch", "heading_R", "turn_mean", "uturn_ratio", "loops",
             "ks_speed_med", "ks_speed_p90", "ks_frac_stat", "ks_frac_drift", "ks_heading_R", "ks_turn_med",
             "n_stays", "stay_frac", "heading_autocorr", "drift_cycles", "is_stationary", "bearing_std"]
KEEP_STATE = ["state", "port_km", "home_km", "in_port", "port_stay", "moving"]


def load(test):
    files = sorted(C.FEAT_DIR.glob(f"{'test_hourly' if test else 'hourly'}_*.parquet"))
    H = pd.concat([pd.read_parquet(f, columns=HOUR_COLS) for f in files], ignore_index=True)
    H = H[H["n_points"].notna() & (H["hour"] >= 0)].reset_index(drop=True)
    H["fishing_code"] = H["fishing_code"].astype("object")
    fix = H["bearing_std"].isna() & (H["heading_R"] >= 0.999)
    H.loc[fix, "bearing_std"] = 0.0
    f64 = H.select_dtypes("float64").columns
    H[f64] = H[f64].astype(np.float32)
    H["ts"] = pd.to_datetime(H["part_dt"].astype(str), format="%Y%m%d") + pd.to_timedelta(H["hour"], unit="h")
    return H.sort_values(["rfid_id", "ts"], kind="stable").reset_index(drop=True)


def context(H):
    rid = H["rfid_id"].to_numpy()
    t = H["ts"].to_numpy().astype("datetime64[h]").astype(np.int64)
    same_prev = np.r_[False, rid[1:] == rid[:-1]]
    same_next = np.r_[rid[1:] == rid[:-1], False]
    H["gap_prev_h"] = np.where(same_prev, t - np.r_[t[0], t[:-1]], 999).astype(np.float32)
    H["gap_next_h"] = np.where(same_next, np.r_[t[1:], t[-1]] - t, 999).astype(np.float32)
    v = H["ks_speed_med"].fillna(H["sp_p50"]).fillna(0).to_numpy()
    consec = same_prev & (H["gap_prev_h"].to_numpy() == 1)
    H["d_speed"] = np.where(consec, v - np.r_[v[0], v[:-1]], np.nan).astype(np.float32)
    j = np.flatnonzero(consec)
    jump = np.full(len(H), np.nan, np.float32)
    jump[j] = A.hav_km(H["lon_e"].to_numpy()[j - 1], H["lat_e"].to_numpy()[j - 1],
                       H["lon_s"].to_numpy()[j], H["lat_s"].to_numpy()[j])
    H["jump_prev_km"] = jump
    hk = H["home_km"].to_numpy()
    H["home_km_delta"] = np.where(consec, hk - np.r_[hk[0], hk[:-1]], np.nan).astype(np.float32)
    H["at_sea"] = ((H["port_stay"] == 0) & (H["port_km"] > 1)).astype(np.float32)

    # 그 배의 평소 시각·위치 대비 드묾
    hod = H.groupby(["rfid_id", "hour"]).size() / H.groupby("rfid_id").size()
    H["hod_rarity"] = (1 - hod.reindex(pd.MultiIndex.from_arrays([H["rfid_id"], H["hour"]])).to_numpy()).astype(np.float32)
    cell = pd.Series(A.grid(H["lon_c"], H["lat_c"], 1000), index=H.index)
    cs = cell.groupby([H["rfid_id"], cell]).transform("size") / H.groupby("rfid_id")["hour"].transform("size")
    H["cell_rarity"] = (1 - cs).astype(np.float32)
    H["n_vessel_hours"] = H.groupby("rfid_id")["hour"].transform("size").astype(np.int32)
    return H


def main(test):
    t0 = time.time()
    H = load(test)
    print(f"{'test' if test else 'train'} hourly {len(H):,}행 / 선박 {H['rfid_id'].nunique():,}척")
    H = A.add_port_state(H, pd.read_csv(C.FEAT_DIR / "port_cells.csv"))
    H = context(H)
    drop = [c for c in H.columns if c in ("home_lon", "home_lat")]
    out = C.FEAT_DIR / f"anom_hourly{'_test' if test else ''}.parquet"
    H.drop(columns=drop).to_parquet(out, index=False)
    print(f"✅ {out.name} {H.shape}  {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args().test)
