"""
[STEP9-6] 매칭 안 된 사고 활용 ─ 사고 선박이 사고 지점에 안 보여도 남는 간접 흔적 두 가지를 위약 검정으로 확인
  (1) 동행 쌍(예인·구조 후보): 사고 후 시간대에 바다에서 두 배가 같은 10분 구간 위치 150m 안으로 3구간(30분) 이상 붙어서 이동
      범위: 사고 위치 15km 안, 사고 시각 −3 ~ +12시간. 이동 = 1시간 이동거리 1km 이상, 속력 7노트 미만
  (2) 장기 휴항: 사고 위치 5km 안(±3시간)에 있던 같은 업종 배 중, 그 날 이후 다음 출어까지 7일 이상 + 그 배 평소 간격 상위 5%
  위약 검정: 사고 시각을 ±7·14일 옮겨 같은 계산. 실제가 옮긴 시각보다 많아야 "사고 흔적"
  출력: results/step9_rematch.csv
실행: python step9_rematch.py
"""
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C
import step9_labels as L9

SHIFTS = [-14, -7, 0, 7, 14]
BIN_LON = [f"b_lon{j}" for j in range(6)]
BIN_LAT = [f"b_lat{j}" for j in range(6)]
PAIR_M, PAIR_BINS, TOW_KM, LAY_KM = 150.0, 3, 15.0, 5.0


def load(acc):
    la0, la1 = acc["위도(변환)"].min() - 0.3, acc["위도(변환)"].max() + 0.3
    lo0, lo1 = acc["경도(변환)"].min() - 0.3, acc["경도(변환)"].max() + 0.3
    out = []
    for m in range(1, 7):
        P = pd.read_parquet(C.FEAT_DIR / f"hourly_{m:02d}.parquet", columns=["rfid_id", "part_dt", "hour"] + BIN_LON + BIN_LAT)
        P = P[P["b_lat0"].between(la0, la1) & P["b_lon0"].between(lo0, lo1)].copy()
        P["ts"] = pd.to_datetime(P["part_dt"].astype(str)) + pd.to_timedelta(P["hour"].astype(int), unit="h")
        out.append(P.drop(columns=["part_dt", "hour"]))
    P = pd.concat(out, ignore_index=True)
    P["rid"] = P["rfid_id"].astype(str)
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "ts", "at_sea", "dist_m", "ks_speed_med"])
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    return P.merge(H[["rid", "ts", "at_sea", "dist_m", "ks_speed_med"]], on=["rid", "ts"], how="left"), H


def tow_pairs(W):
    """W: 한 시간대의 이동 중 바다 선박들. 붙어서 움직인 쌍 수"""
    if len(W) < 2:
        return set()
    lon = W[BIN_LON].to_numpy()
    lat = W[BIN_LAT].to_numpy()
    k = np.cos(np.radians(np.nanmean(lat))) * 111320.0
    ids = W["rid"].to_numpy()
    pairs = set()
    for i in range(len(W) - 1):
        d = np.hypot((lon[i + 1:] - lon[i]) * k, (lat[i + 1:] - lat[i]) * 110540.0)
        hit = np.nansum(d <= PAIR_M, axis=1) >= PAIR_BINS
        for j in np.flatnonzero(hit):
            pairs.add(tuple(sorted((ids[i], ids[i + 1 + j]))))
    return pairs


def main():
    t0 = time.time()
    acc = pd.read_excel(L9.ACC)
    P, H = load(acc)
    Mv = P[(P["at_sea"] == 1) & (P["dist_m"] >= 1000) & (P["ks_speed_med"].fillna(0) < 7)]
    by_hour = {t: g for t, g in Mv.groupby("ts")}
    print(f"범위 안 시간대 {len(P):,}행 / 이동 중 {len(Mv):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code"])
    gear = dict(zip(S["rfid_id"].astype(str), S["fishing_code"]))
    D = H[H["at_sea"] == 1].assign(day=lambda d: d["ts"].dt.normalize())[["rid", "day"]].drop_duplicates().sort_values(["rid", "day"])
    days = {r: g["day"].to_numpy() for r, g in D.groupby("rid")}

    def layoff(rid, t):
        d = days.get(rid)
        if d is None or len(d) < 5:
            return False
        nxt = d[d > t]
        if len(nxt) == 0:
            return bool((np.datetime64("2026-06-30") - t) / np.timedelta64(1, "D") >= 7)
        after = (nxt[0] - t) / np.timedelta64(1, "D")
        return bool(after >= 7 and (np.diff(d) / np.timedelta64(1, "D") < after).mean() >= 0.95)

    rows = []
    for s in SHIFTS:
        for i, a in acc.iterrows():
            t = (a["발생일시"] + pd.Timedelta(days=s)).floor("h")
            if not (pd.Timestamp("2026-01-01") <= t - pd.Timedelta(hours=3) and t + pd.Timedelta(hours=12) <= pd.Timestamp("2026-06-30 23:00")):
                continue
            pairs = set()
            for h in pd.date_range(t - pd.Timedelta(hours=3), t + pd.Timedelta(hours=12), freq="h"):
                W = by_hour.get(h)
                if W is None:
                    continue
                near = A.hav_km(W[BIN_LON[2]], W[BIN_LAT[2]], a["경도(변환)"], a["위도(변환)"]) <= TOW_KM
                pairs |= tow_pairs(W[near.fillna(False)])
            g = L9.GEAR.get(a["어업면허"], a["어업면허"])
            W = P[(P["ts"] >= t - pd.Timedelta(hours=3)) & (P["ts"] <= t + pd.Timedelta(hours=3))]
            d = A.hav_km(W[BIN_LON[2]], W[BIN_LAT[2]], a["경도(변환)"], a["위도(변환)"])
            cand = W.loc[d <= LAY_KM, "rid"].unique()
            cand = [r for r in cand if gear.get(r) == g]
            lay = [r for r in cand if layoff(r, np.datetime64(t.normalize()))]
            rows.append(dict(shift=s, acc_idx=i, group=L9.group(a), n_pairs=len(pairs), any_pair=len(pairs) > 0,
                             n_cand=len(cand), n_layoff=len(lay), any_layoff=len(lay) > 0))
        print(f"  shift {s:+d} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    R = pd.DataFrame(rows)
    R["kind"] = np.where(R["shift"] == 0, "실제 시각", "옮긴 시각")
    T = R.groupby(["group", "kind"]).agg(n=("acc_idx", "size"), any_pair=("any_pair", "mean"), n_pairs=("n_pairs", "mean"),
                                          any_layoff=("any_layoff", "mean"), n_layoff=("n_layoff", "mean"),
                                          n_cand=("n_cand", "mean")).reset_index()
    T.round(4).to_csv(C.RESULT_DIR / "step9_rematch.csv", index=False, encoding="utf-8-sig")
    print(T.round(3).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
