"""
[FE-2] 시간(L1) → 선박-일(L2) → 선박(L3) 계층 집계  (HANDOFF §3.3 Feature Store)

입력 : features/hourly_MM.parquet (fe_hourly.py)
출력 : features/shipday.parquet       (선박·날짜 1행)
       features/ship_features.parquet (선박 1행, 분류 학습용)
       features/port_cells.csv        (공용 항구 격자)

단계
  1) 공용 항구 (나연)   : 정박 시간대 500m 격자 중 ≥10척이 정박한 칸
  2) 모항 (민성)        : 선박별 정박이 가장 많이 몰린 1km 격자 → 2km 안 정지 = 항구 정박(H4 제거 대상)
  3) 시간대 상태        : PORT / SEA_STOP / DRIFT / FISH / TRANSIT  (5-step 속도 기반, 지터 강건)
  4) 선박-일 집계       : 활동량·거리·범위·항차·조업시간·이동 시간대 행동(mv_*)·재방문(가설 A)·하루 리듬(가설 E)
  5) 선박 집계          : 출어일 통계(mean/median/std), 조업 시간대 풀링 분위수(민성 C),
                          어장 반복(cross-day revisit, site Jaccard), 계절(가설 H), bag-of-days(가설 G), 위치(LOC)

실행: python fe_aggregate.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

import fe_config as C

R_KM = 6371.0088
REF_COS = np.cos(np.radians(36.0))  # 전역 격자용 기준 위도
PORT, SEA_STOP, DRIFT, FISH, TRANSIT = 0, 1, 2, 3, 4


def hav_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def grid(lon, lat, m):
    gx = np.floor(np.asarray(lon) * 111_320 * REF_COS / m).astype(np.int64)
    gy = np.floor(np.asarray(lat) * 110_574 / m).astype(np.int64)
    return gx * 1_000_000 + gy


def load_hourly():
    files = sorted(C.FEAT_DIR.glob("hourly_*.parquet"))
    H = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    H = H[H["n_points"].notna()].reset_index(drop=True)
    H["fishing_code"] = H["fishing_code"].astype("object")
    # heading_R 가 부동소수 오차로 1을 넘으면 bearing_std 가 NaN → 일직선(0)으로 보정
    fix = H["bearing_std"].isna() & (H["heading_R"] >= 0.999)
    H.loc[fix, "bearing_std"] = 0.0
    H = H.sort_values(["rfid_id", "part_dt", "hour"], kind="stable").reset_index(drop=True)
    print(f"hourly {len(H):,}행 / 선박 {H['rfid_id'].nunique():,}척 / 파일 {len(files)}개")
    return H


# ─────────────────────────────────────────────
# 1~3. 항구 / 모항 / 상태
# ─────────────────────────────────────────────
def add_port_state(H):
    st = H["is_stationary"].to_numpy() == 1
    cell = grid(H["lon_c"], H["lat_c"], C.PORT_GRID_M)
    s = pd.DataFrame({"cell": cell[st], "rid": H["rfid_id"].to_numpy()[st],
                      "lon": H["lon_c"].to_numpy()[st], "lat": H["lat_c"].to_numpy()[st]})
    agg = s.groupby("cell").agg(n_vessels=("rid", "nunique"), lon=("lon", "median"), lat=("lat", "median"))
    ports = agg[agg["n_vessels"] >= C.PORT_MIN_VESSELS].reset_index()
    ports.to_csv(C.FEAT_DIR / "port_cells.csv", index=False)
    print(f"공용 항구 격자: {len(ports)}칸")
    tree = BallTree(np.radians(ports[["lat", "lon"]].to_numpy()), metric="haversine")
    d, _ = tree.query(np.radians(H[["lat_c", "lon_c"]].to_numpy()), k=1)
    H["port_km"] = (d[:, 0] * R_KM).astype(np.float32)
    H["in_port"] = (st & (H["port_km"].to_numpy() < 0.5)).astype(np.float32)

    # 모항: 선박별 정박 1km 격자 최빈
    s["cell1k"] = grid(s["lon"], s["lat"], C.HOME_GRID_M)
    cnt = s.groupby(["rid", "cell1k"]).size().rename("n").reset_index()
    top = cnt.sort_values(["rid", "n"], ascending=[True, False]).drop_duplicates("rid")
    home = s.merge(top[["rid", "cell1k"]], on=["rid", "cell1k"]).groupby("rid")[["lon", "lat"]].median()
    home.columns = ["home_lon", "home_lat"]
    H = H.merge(home, left_on="rfid_id", right_index=True, how="left")
    # 정박 기록이 없는 배는 첫 기록 위치를 모항으로 근사
    first = H.groupby("rfid_id")[["lon_s", "lat_s"]].transform("first")
    H["home_lon"] = H["home_lon"].fillna(first["lon_s"])
    H["home_lat"] = H["home_lat"].fillna(first["lat_s"])
    H["home_km"] = hav_km(H["lon_c"], H["lat_c"], H["home_lon"], H["home_lat"]).astype(np.float32)

    port_stay = st & ((H["home_km"].to_numpy() < C.HOME_RADIUS_KM) | (H["in_port"].to_numpy() == 1))
    v = H["ks_speed_med"].fillna(H["sp_p50"]).fillna(0).to_numpy()
    hr = H["ks_heading_R"].fillna(0).to_numpy()
    state = np.select(
        [port_stay, st | (v < 0.3), v < 2, v < 5, hr >= 0.8],
        [PORT, SEA_STOP, DRIFT, FISH, TRANSIT], FISH).astype(np.int8)
    H["state"] = state
    H["port_stay"] = port_stay.astype(np.float32)
    # 이동 시간대 (나연): 정지 80% 미만 & 평활거리 200m 이상 & 항구 정박 아님
    H["moving"] = ((H["frac_stop"].fillna(1) < 0.8) & (H["dist_m"] >= 200) & ~port_stay).astype(np.float32)
    print("상태 비율:", pd.Series(state).value_counts(normalize=True).rename(
        {PORT: "PORT", SEA_STOP: "SEA_STOP", DRIFT: "DRIFT", FISH: "FISH", TRANSIT: "TRANSIT"}).round(3).to_dict())
    print(f"이동 시간대 비율: {H['moving'].mean():.3f}")
    return H


# ─────────────────────────────────────────────
# 4. 선박-일 집계
# ─────────────────────────────────────────────
MV_FEATS = ["sp_mean", "sp_p90", "sp_std", "sp_change_mean", "frac_slow", "frac_mid", "frac_fast",
            "heading_R", "bearing_std", "turn_mean", "turn_std", "uturn_ratio", "turn90_per_km",
            "large_turn_ratio", "loops", "straightness", "path_over_rg", "path_over_diag",
            "n_stop_move_switch", "rg_m", "rg_med_m", "bbox_diag_m", "bbox_aspect", "hull_area_m2",
            "uniq_pos_ratio", "pca_elong", "seg_sinuosity_mean", "ks_speed_med", "ks_speed_p90",
            "ks_heading_R", "ks_turn_med", "ks_straight_max_m", "slow_run_max_min", "speed_entropy",
            "n_stays", "stay_frac", "spec_entropy", "lowfreq_ratio", "step_cv", "spike_ratio",
            "raw_clean_ratio", "n_segments", "n_points", "bin_dispersion_m"]


def day_revisit(H):
    """가설 A: 이동 시간대의 10분 평활 위치를 200m 격자로 → 같은 칸을 2시간 이상 떠났다 다시 온 횟수 등"""
    mv = H[H["moving"] == 1]
    lon = mv[[f"b_lon{i}" for i in range(C.N_BINS)]].to_numpy()
    lat = mv[[f"b_lat{i}" for i in range(C.N_BINS)]].to_numpy()
    rep = lambda a: np.repeat(a.to_numpy(), C.N_BINS)
    P = pd.DataFrame({"rfid_id": rep(mv["rfid_id"]), "part_dt": rep(mv["part_dt"]), "hour": rep(mv["hour"]),
                      "lon": lon.ravel(), "lat": lat.ravel()}).dropna()
    P["cell"] = grid(P["lon"], P["lat"], C.REVISIT_GRID_M)
    P["cell1k"] = grid(P["lon"], P["lat"], C.SITE_GRID_M)
    key = ["rfid_id", "part_dt"]

    u = P[key + ["cell", "hour"]].drop_duplicates().sort_values(key + ["cell", "hour"])
    same = (u["cell"].to_numpy()[1:] == u["cell"].to_numpy()[:-1]) & \
           (u["part_dt"].to_numpy()[1:] == u["part_dt"].to_numpy()[:-1]) & \
           (u["rfid_id"].to_numpy()[1:] == u["rfid_id"].to_numpy()[:-1])
    ret = np.r_[False, same & (np.diff(u["hour"].to_numpy()) >= C.REVISIT_ABSENCE_H)]
    u["ret"] = ret
    nh = u.groupby(key + ["cell"]).size().rename("nh").reset_index()
    out = pd.DataFrame({
        "return_after_absence": u.groupby(key)["ret"].sum(),
        "revisit_cell_ratio": nh.assign(r=nh["nh"] > 1).groupby(key)["r"].mean(),
        "n_cells_200m": nh.groupby(key).size(),
    })
    cc = P.groupby(key + ["cell"]).size().rename("n").reset_index()
    tot = cc.groupby(key)["n"].sum()
    top3 = cc.sort_values(key + ["n"], ascending=[True, True, False]).groupby(key).head(3).groupby(key)["n"].sum()
    out["top3_cell_share"] = top3 / tot
    cells1k = P[key + ["cell1k"]].drop_duplicates()
    return out.reset_index(), cells1k


def build_shipday(H):
    key = ["rfid_id", "part_dt"]
    rid, pdt, hr = H["rfid_id"].to_numpy(), H["part_dt"].to_numpy(), H["hour"].to_numpy().astype(int)
    newday = np.r_[True, (rid[1:] != rid[:-1]) | (pdt[1:] != pdt[:-1])]
    mov = H["moving"].to_numpy() == 1
    state = H["state"].to_numpy()

    prev_state = np.r_[-1, state[:-1]]
    H["_trans"] = (~newday & (state != prev_state)).astype(np.int8)
    H["_depart"] = (~newday & (prev_state == PORT) & (state != PORT)).astype(np.int8)
    consec = np.r_[False, ~newday[1:] & (hr[1:] == hr[:-1] + 1)]
    jump = np.full(len(H), np.nan)
    j = np.flatnonzero(consec)
    jump[j] = hav_km(H["lon_e"].to_numpy()[j - 1], H["lat_e"].to_numpy()[j - 1],
                     H["lon_s"].to_numpy()[j], H["lat_s"].to_numpy()[j])
    H["_jump"] = jump
    # 이동 블록 (가설 E 하루 리듬)
    prev_mov = np.r_[False, mov[:-1]]
    bstart = mov & (newday | ~prev_mov | ~consec)
    H["_bstart"] = bstart.astype(np.int8)
    H["_bid"] = np.where(mov, np.cumsum(bstart), -1)
    for s, nm in [(SEA_STOP, "sea_stop"), (DRIFT, "drift"), (FISH, "fish"), (TRANSIT, "transit")]:
        H[f"_s_{nm}"] = (state == s).astype(np.float32)
    H["_night_mv"] = np.where(mov, np.isin(hr, list(C.NIGHT_HOURS)), np.nan)
    for nm, (a, b) in {"dawn": (0, 5), "morning": (6, 11), "afternoon": (12, 17), "night": (18, 23)}.items():
        H[f"_mv_{nm}"] = np.where(mov, (hr >= a) & (hr <= b), np.nan)
    H["_hr_mv"] = np.where(mov, hr, np.nan)

    g = H.groupby(key, sort=False)
    D = g.agg(
        fishing_code=("fishing_code", "first"),
        n_hours=("hour", "size"), first_hour=("hour", "min"), last_hour=("hour", "max"),
        n_moving_hours=("moving", "sum"), stationary_ratio=("is_stationary", "mean"),
        n_points_day=("n_points", "sum"), n_segments_mean=("n_segments", "mean"), frac_stop_mean=("frac_stop", "mean"),
        dist_sum_m=("dist_m", "sum"), dist_raw_sum_m=("dist_raw_m", "sum"), n_jumps_day=("n_jumps", "sum"),
        home_km_max=("home_km", "max"), home_km_mean=("home_km", "mean"),
        port_km_mean=("port_km", "mean"), in_port_ratio=("port_stay", "mean"),
        sea_stop_share=("_s_sea_stop", "mean"), drift_share=("_s_drift", "mean"),
        fish_share=("_s_fish", "mean"), transit_share=("_s_transit", "mean"),
        n_transitions=("_trans", "sum"), departures=("_depart", "sum"),
        inter_hour_jump_med_km=("_jump", "median"), active_block_count=("_bstart", "sum"),
        night_ratio=("_night_mv", "mean"), mv_share_dawn=("_mv_dawn", "mean"),
        mv_share_morning=("_mv_morning", "mean"), mv_share_afternoon=("_mv_afternoon", "mean"),
        mv_share_night=("_mv_night", "mean"), mv_first_hour=("_hr_mv", "min"), mv_last_hour=("_hr_mv", "max"),
        lon_c=("lon_c", "median"), lat_c=("lat_c", "median"),
        lon_start=("lon_s", "first"), lat_start=("lat_s", "first"),
        lon_end=("lon_e", "last"), lat_end=("lat_e", "last"),
        home_km_start=("home_km", "first"), home_km_end=("home_km", "last"),
        state_end=("state", "last"), month=("part_dt", "first"),
    )
    D["moving_ratio"] = D["n_moving_hours"] / D["n_hours"]
    D["active_span_h"] = D["last_hour"] - D["first_hour"] + 1
    D["mv_span_h"] = D["mv_last_hour"] - D["mv_first_hour"] + 1
    D["dist_day_km"] = D["dist_sum_m"] / 1000
    D["dist_raw_day_km"] = D["dist_raw_sum_m"] / 1000
    D["jump_share_day"] = 1 - D["dist_sum_m"] / D["dist_raw_sum_m"].clip(lower=1)
    D["day_net_disp_km"] = hav_km(D["lon_start"], D["lat_start"], D["lon_end"], D["lat_end"])
    D["day_straightness"] = (D["day_net_disp_km"] / D["dist_day_km"].where(D["dist_day_km"] > 0.2)).clip(upper=1)
    D["returned_to_port"] = ((D["home_km_end"] < C.HOME_RADIUS_KM) | (D["state_end"] == PORT)).astype(float)
    D["month"] = D["month"] // 100 % 100
    D = D.drop(columns=["dist_sum_m", "dist_raw_sum_m", "state_end"])

    # 시간 중심들의 퍼짐 / 첫 위치에서 95% 이격 (최댓값 대신 분위수 = 지터 강건)
    cx = H["lon_c"].to_numpy() * 111.32 * REF_COS
    cy = H["lat_c"].to_numpy() * 110.574
    tmp = pd.DataFrame({"rfid_id": rid, "part_dt": pdt, "cx": cx, "cy": cy})
    gm = tmp.groupby(key)[["cx", "cy"]].transform("mean")
    tmp["r2"] = (tmp["cx"] - gm["cx"]) ** 2 + (tmp["cy"] - gm["cy"]) ** 2
    f0 = tmp.groupby(key)[["cx", "cy"]].transform("first")
    tmp["d0"] = np.hypot(tmp["cx"] - f0["cx"], tmp["cy"] - f0["cy"])
    gg = tmp.groupby(key)
    D["day_rg_km"] = np.sqrt(gg["r2"].mean())
    D["day_bbox_diag_km"] = np.hypot(gg["cx"].max() - gg["cx"].min(), gg["cy"].max() - gg["cy"].min())
    D["day_range_q95_km"] = gg["d0"].quantile(0.95)

    # 최장 연속 이동 / 블록 간 최대 휴식
    B = H.loc[H["_bid"] >= 0, key + ["_bid", "hour"]]
    bl = B.groupby("_bid").agg(rfid_id=("rfid_id", "first"), part_dt=("part_dt", "first"),
                               h0=("hour", "min"), h1=("hour", "max"), n=("hour", "size"))
    bl["gap"] = bl["h0"] - bl.groupby(key)["h1"].shift() - 1
    D = D.join(bl.groupby(key).agg(longest_streak=("n", "max"), max_block_gap_h=("gap", "max")))
    D["longest_streak"] = D["longest_streak"].fillna(0)

    # 이동 시간대 행동 평균 (mv_*) + 이동 시간대 대표 위치
    M = H[H["moving"] == 1]
    mvf = M.groupby(key)[MV_FEATS].mean().add_prefix("mv_")
    mvf["mv_lon_c"] = M.groupby(key)["lon_c"].median()
    mvf["mv_lat_c"] = M.groupby(key)["lat_c"].median()
    D = D.join(mvf)

    rv, cells1k = day_revisit(H)
    D = D.join(rv.set_index(key))
    D = D.reset_index()
    H.drop(columns=[c for c in H.columns if c.startswith("_")], inplace=True)
    print(f"shipday {len(D):,}행 × {D.shape[1]}열")
    return D, cells1k


# ─────────────────────────────────────────────
# 5. 선박 집계
# ─────────────────────────────────────────────
DAY_STAT = ["n_hours", "n_moving_hours", "moving_ratio", "active_span_h", "first_hour", "last_hour",
            "dist_day_km", "jump_share_day", "day_net_disp_km", "day_straightness", "day_rg_km",
            "day_bbox_diag_km", "day_range_q95_km", "home_km_max", "home_km_mean", "in_port_ratio",
            "sea_stop_share", "drift_share", "fish_share", "transit_share", "n_transitions", "departures",
            "longest_streak", "active_block_count", "max_block_gap_h", "mv_first_hour", "mv_last_hour",
            "mv_span_h", "mv_share_dawn", "mv_share_morning", "mv_share_afternoon", "mv_share_night",
            "night_ratio", "inter_hour_jump_med_km", "return_after_absence", "revisit_cell_ratio",
            "top3_cell_share", "n_cells_200m", "stationary_ratio", "n_segments_mean"] + \
           ["mv_" + c for c in MV_FEATS]
HOUR_POOL = ["ks_speed_med", "ks_speed_p90", "rg_med_m", "ks_heading_R", "ks_turn_med", "n_stays",
             "ks_straight_max_m", "slow_run_max_min", "sp_p50", "sp_p90", "turn_mean", "straightness",
             "pca_elong", "frac_slow", "frac_mid", "path_over_rg", "speed_entropy", "stay_frac",
             "uniq_pos_ratio", "dist_m"]
DAY_TYPE_FEATS = ["n_moving_hours", "dist_day_km", "day_range_q95_km", "home_km_max", "drift_share",
                  "fish_share", "transit_share", "sea_stop_share", "mv_first_hour", "mv_span_h",
                  "revisit_cell_ratio", "mv_ks_speed_med", "mv_ks_heading_R", "mv_turn_mean"]
LOC_COLS = ["lon_c", "lat_c", "home_lon", "home_lat", "home_port_km", "fishing_lon", "fishing_lat"]


def build_ship(H, D, cells1k):
    rk = "rfid_id"
    lab = H.groupby(rk)["fishing_code"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
    S = pd.DataFrame(index=lab.index)
    S["fishing_code"] = lab

    # 활동 / 시간 사용 구성 (민성 A, H4: 항구 정박 시간 제외 비율도 같이)
    dg = D.groupby(rk)
    S["n_days"] = dg.size()
    S["n_months"] = dg["month"].nunique()
    Dm = D[D["n_moving_hours"] >= 1]
    S["n_moving_days"] = Dm.groupby(rk).size().reindex(S.index).fillna(0)
    S["moving_days_ratio"] = S["n_moving_days"] / S["n_days"]
    S["moving_days_per_month"] = S["n_moving_days"] / S["n_months"]
    S["hours_per_day"] = dg["n_hours"].mean()
    hg = H.groupby(rk)
    S["port_share"] = hg["port_stay"].mean()
    for s, nm in [(SEA_STOP, "sea_stationary"), (DRIFT, "drift"), (FISH, "fishing_speed"), (TRANSIT, "transit")]:
        S[f"{nm}_share"] = (H["state"] == s).groupby(H[rk]).mean()
        nonport = H["state"] != PORT
        S[f"{nm}_share_sea"] = (H.loc[nonport, "state"] == s).groupby(H.loc[nonport, rk]).mean()
    S["moving_hour_share"] = hg["moving"].mean()
    S["returned_ratio"] = Dm.groupby(rk)["returned_to_port"].mean()
    S["transitions_per_day"] = Dm.groupby(rk)["n_transitions"].mean()
    S["trips_per_day"] = Dm.groupby(rk)["departures"].mean()

    # 출어일 통계 (나연 3단위 사전: mean/median/std, 일부 p90)
    dmg = Dm.groupby(rk)[DAY_STAT]
    for stat in ["mean", "median", "std"]:
        S = S.join(dmg.agg(stat).add_suffix(f"_{stat}"))
    S = S.join(Dm.groupby(rk)[["dist_day_km", "home_km_max", "n_moving_hours", "day_range_q95_km"]]
               .quantile(0.9).add_suffix("_p90"))
    S["trip_hours_q50"] = S["mv_span_h_median"]
    S["trip_max_dist_q50"] = S["home_km_max_median"]
    S["departure_hour_q50"] = S["mv_first_hour_median"]

    # 조업 시간대 풀링 분위수 (민성 C: 정박·항해 제외 후 배 단위 통계)
    Fh = H[(H["moving"] == 1) & (H["state"] != TRANSIT)]
    fg = Fh.groupby(rk)[HOUR_POOL]
    S = S.join(fg.median().add_prefix("fh_").add_suffix("_q50"))
    S = S.join(fg.quantile(0.9).add_prefix("fh_").add_suffix("_q90"))
    S["fishing_night_share"] = Fh["hour"].isin(list(C.NIGHT_HOURS)).groupby(Fh[rk]).mean()
    dr = H[H["state"] == DRIFT]
    S["drift_heading_R_q50"] = dr.groupby(rk)["ks_heading_R"].median()

    # 어장 반복 (민성 B cross_day_revisit, 가설 A day_site_jaccard)
    c = cells1k.merge(Dm[[rk, "part_dt"]], on=[rk, "part_dt"])
    nd = c.groupby([rk, "cell1k"])["part_dt"].nunique()
    S["cross_day_revisit"] = (nd >= 2).groupby(level=0).mean()
    jac = {}
    for r, sub in c.groupby(rk):
        days = sub.groupby("part_dt")["cell1k"].apply(set).sort_index()
        if len(days) >= 2:
            vals = list(days.values)
            jac[r] = np.mean([len(a & b) / max(len(a | b), 1) for a, b in zip(vals[:-1], vals[1:])])
    S["site_jaccard_mean"] = pd.Series(jac)

    # 조업 해역 넓이·퍼짐 (나연 area_range_km, 민성 fishing_area_radius) ─ 크기라서 위치 shortcut 아님
    fx = Dm["mv_lon_c"] * 111.32 * REF_COS
    fy = Dm["mv_lat_c"] * 110.574
    q = pd.DataFrame({rk: Dm[rk], "fx": fx, "fy": fy}).dropna().groupby(rk)
    S["area_range_km"] = np.hypot(q["fx"].quantile(0.95) - q["fx"].quantile(0.05),
                                  q["fy"].quantile(0.95) - q["fy"].quantile(0.05))
    med = q[["fx", "fy"]].transform("median")
    tmp = pd.DataFrame({rk: Dm.loc[med.index, rk], "d": np.hypot(fx.loc[med.index] - med["fx"],
                                                                 fy.loc[med.index] - med["fy"])})
    S["fishing_area_radius"] = tmp.groupby(rk)["d"].median()

    # 계절 (가설 H): 월별 출어일 비중
    mm = Dm.groupby([rk, "month"]).size().unstack(fill_value=0)
    mm = mm.div(mm.sum(axis=1), axis=0).reindex(columns=range(1, 7), fill_value=0)
    mm.columns = [f"season_m{m}_share" for m in mm.columns]
    S = S.join(mm)

    # bag-of-days (가설 G): 출어일 유형 군집 비율 + 엔트로피. 라벨 미사용(비지도) → 전체 선박으로 적합
    X = Dm[DAY_TYPE_FEATS].astype(float)
    X = X.fillna(X.median())
    Z = StandardScaler().fit_transform(X.clip(X.quantile(0.01), X.quantile(0.99), axis=1))
    km = KMeans(C.N_DAY_TYPES, random_state=C.RANDOM_STATE, n_init=4).fit(Z)
    bag = pd.crosstab(Dm[rk].to_numpy(), km.labels_, normalize="index")
    bag.columns = [f"daytype_{i}_share" for i in bag.columns]
    S = S.join(bag)
    p = S[bag.columns].fillna(0).to_numpy()
    S["day_type_entropy"] = -(p * np.log(p + 1e-9)).sum(axis=1)
    S.loc[S["n_moving_days"] == 0, "day_type_entropy"] = np.nan

    # 위치 (LOC, shortcut 위험 → Model B 에서만)
    S["lon_c"] = hg["lon_c"].median()
    S["lat_c"] = hg["lat_c"].median()
    S["home_lon"] = hg["home_lon"].first()
    S["home_lat"] = hg["home_lat"].first()
    S["fishing_lon"] = Dm.groupby(rk)["mv_lon_c"].median()
    S["fishing_lat"] = Dm.groupby(rk)["mv_lat_c"].median()
    ports = pd.read_csv(C.FEAT_DIR / "port_cells.csv")
    tree = BallTree(np.radians(ports[["lat", "lon"]].to_numpy()), metric="haversine")
    d, _ = tree.query(np.radians(S[["home_lat", "home_lon"]].to_numpy()), k=1)
    S["home_port_km"] = d[:, 0] * R_KM

    S = S.reset_index().rename(columns={"index": rk})
    print(f"ship_features {len(S):,}척 × {S.shape[1]}열")
    return S


def main():
    t = time.time()
    H = load_hourly()
    H = add_port_state(H)
    D, cells1k = build_shipday(H)
    D.to_parquet(C.FEAT_DIR / "shipday.parquet", index=False)
    S = build_ship(H, D, cells1k)
    S.to_parquet(C.FEAT_DIR / "ship_features.parquet", index=False)
    print(S["fishing_code"].value_counts())
    print(f"✅ 완료 {(time.time() - t) / 60:.1f}분")


if __name__ == "__main__":
    main()
