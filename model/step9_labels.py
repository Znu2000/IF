"""
[STEP9-1] 사고 라벨 v2 ─ 10분 평활 위치(시간당 6점) 거리 + 사고 유형별 항적 흔적으로 매칭
  v1(anomaly_labels.py) 한계: 1시간 대표점 3개로만 거리 계산, 사고 유형 정보 미사용 → 확실(A) 35건
  거리   : 사고 시각 ±90분 안의 10분 평활 점과 사고 위치의 최소 거리 (같은 업종 선박만)
  흔적   : 사고 시각 기준 시간대 상태로 유형별 판정 (anom_hourly 의 상태: 0 항구, 1 정지, 2 표류, 3 조업, 4 이동)
           기동 불능(기관손상·부유물감김·운항저해·추진기·키 손상·표류·방향상실):
               사고 무렵 바다에 있고, 직후 정지·표류 + (직전 이동·조업 또는 이후 저속 직진 귀항(예인) 또는 이후 귀항)
           기타 바다 사고(충돌·화재·침수·좌초 등): 사고 무렵 바다 + (이후 귀항 또는 신호 끊김 또는 정지·표류)
           항구 안 사고: 흔적 판정 없음 (원래 서 있어서 구분 불가)
  등급   : A  = 가장 가까운 배가 1.0km 안 + (두 번째보다 2배 이상 가깝거나 두 번째가 3km 밖)
           S  = (바다 사고) 2km 안 후보 중 흔적이 있는 배가 정확히 1척
           AS = A 와 S 가 같은 배 (가장 확실)
           B  = 가장 가까운 배 3km 안 (애매), none = 그 외
  출력 : results/anom_labels_v2.csv (실제 rfid_id 포함 → 깃헙 업로드 금지)
실행: python step9_labels.py
"""
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C

ACC = C.PROJECT_ROOT / "문제 데이터_수정_1103" / "26년 1~6월 어선 사고현황(테스트 데이터).xlsx"
GEAR = {"연승": "채낚기연승낚시"}
DISABLED = {"기관손상", "부유물감김", "운항저해", "추진기손상", "키손상", "표류", "방향상실"}
NEAR_MIN = 90
SIG_KM = 2.0


def group(a):
    if a["발생해역"] == "항계 내":
        return "port"
    return "disabled" if a["발생유형"] in DISABLED else "sea_other"


def load_points(acc):
    """사고 시각 ±2시간 시간대의 10분 평활 점 (선박·시각·6점)"""
    hours = set()
    for t in acc["발생일시"]:
        hours |= set(pd.date_range(t.floor("h") - pd.Timedelta(hours=2), t.floor("h") + pd.Timedelta(hours=2), freq="h"))
    hours = pd.DatetimeIndex(sorted(hours))
    cols = ["rfid_id", "part_dt", "hour"] + [f"b_lon{j}" for j in range(6)] + [f"b_lat{j}" for j in range(6)]
    out = []
    for m in sorted(set(hours.month)):
        P = pd.read_parquet(C.FEAT_DIR / f"hourly_{m:02d}.parquet", columns=cols)
        P["ts"] = pd.to_datetime(P["part_dt"].astype(str)) + pd.to_timedelta(P["hour"].astype(int), unit="h")
        out.append(P[P["ts"].isin(hours)].drop(columns=["part_dt", "hour"]))
    P = pd.concat(out, ignore_index=True)
    P["rid"] = P["rfid_id"].astype(str)
    return P


def signature(Hv, t, grp):
    """Hv: 한 선박의 시간대 기록 (ts 정렬), t: 사고 시각 정시 내림"""
    rel = (Hv["ts"] - t) / pd.Timedelta(hours=1)
    sea = Hv["at_sea"].fillna(0).to_numpy() == 1
    st = Hv["state"].to_numpy()
    sp = Hv["ks_speed_med"].fillna(0).to_numpy()
    hr = Hv["ks_heading_R"].fillna(0).to_numpy()
    w = lambda a, b: ((rel >= a) & (rel <= b)).to_numpy()
    near_sea = (w(-1, 1) & sea).any()
    stop = (w(-1, 3) & sea & np.isin(st, (1, 2))).any()
    pre_move = (w(-4, 0) & sea & np.isin(st, (3, 4))).any()
    tow = (w(1, 12) & sea & (sp >= 2) & (sp < 6) & (hr >= 0.8)).any()
    ret = (w(1, 12) & (st == 0)).any()
    gap = (w(-1, 1) & (Hv["gap_next_h"].fillna(0).to_numpy() >= 3)).any()
    if grp == "disabled":
        sig = near_sea and stop and (pre_move or tow or ret)
    elif grp == "sea_other":
        sig = near_sea and (ret or gap or stop)
    else:
        sig = False
    return dict(sig=bool(sig), near_sea=bool(near_sea), stop=bool(stop), pre_move=bool(pre_move), tow=bool(tow),
                ret=bool(ret), gap=bool(gap))


def main():
    t0 = time.time()
    acc = pd.read_excel(ACC)
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code"])
    gear_of = dict(zip(S["rfid_id"].astype(str), S["fishing_code"]))
    P = load_points(acc)
    P["gear"] = P["rid"].map(gear_of)
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet",
                        columns=["rfid_id", "ts", "at_sea", "state", "ks_speed_med", "ks_heading_R", "gap_next_h"])
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    print(f"평활 점 {len(P):,}행 / 시간대 기록 {len(H):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    rows = []
    for i, a in acc.iterrows():
        t = a["발생일시"]
        grp = group(a)
        gear = GEAR.get(a["어업면허"], a["어업면허"])
        base = dict(acc_idx=i, acc_time=t, acc_type=a["발생유형"], area=a["발생해역"], group=grp, gear=gear,
                    urgent=a["긴급/비긴급"], lat=a["위도(변환)"], lon=a["경도(변환)"])
        W = P[(P["gear"] == gear) & (P["ts"] >= t.floor("h") - pd.Timedelta(hours=2)) & (P["ts"] <= t.floor("h") + pd.Timedelta(hours=2))]
        if W.empty:
            rows.append(base | dict(tier="none", n_cand=0))
            continue
        lon = W[[f"b_lon{j}" for j in range(6)]].to_numpy()
        lat = W[[f"b_lat{j}" for j in range(6)]].to_numpy()
        tm = (W["ts"].to_numpy()[:, None] + (np.arange(6) * 10 + 5).astype("timedelta64[m]"))
        ok = np.abs((tm - np.datetime64(t)) / np.timedelta64(1, "m")) <= NEAR_MIN
        d = np.where(ok & ~np.isnan(lon), A.hav_km(lon, lat, a["경도(변환)"], a["위도(변환)"]), np.inf)
        per = pd.Series(d.min(1), index=W["rid"].to_numpy()).groupby(level=0).min().sort_values()
        per = per[np.isfinite(per)]
        if per.empty:
            rows.append(base | dict(tier="none", n_cand=0))
            continue
        d1, d2 = per.iloc[0], (per.iloc[1] if len(per) > 1 else np.inf)
        a_ok = d1 <= 1.0 and (d2 >= 2 * d1 or d2 > 3)
        near = per[per <= SIG_KM].index.tolist()
        sigs = {}
        if grp != "port":
            Hn = H[H["rid"].isin(near) & (H["ts"] >= t.floor("h") - pd.Timedelta(hours=4))
                   & (H["ts"] <= t.floor("h") + pd.Timedelta(hours=12))].sort_values("ts")
            sigs = {r: signature(g, t.floor("h"), grp) for r, g in Hn.groupby("rid")}
        sig_ids = [r for r in near if sigs.get(r, {}).get("sig")]
        s_ok = len(sig_ids) == 1
        if a_ok and s_ok and sig_ids[0] == per.index[0]:
            tier, rid = "AS", per.index[0]
        elif a_ok:
            tier, rid = "A", per.index[0]
        elif s_ok:
            tier, rid = "S", sig_ids[0]
        elif d1 <= 3:
            tier, rid = "B", per.index[0]
        else:
            tier, rid = "none", None
        rows.append(base | dict(tier=tier, rfid_id=rid, d1_km=round(d1, 3), d2_km=round(d2, 3),
                                d_match_km=round(per.get(rid, np.nan), 3) if rid else np.nan,
                                n_cand=len(per), n_within2=len(near), n_sig=len(sig_ids),
                                **{f"sig_{k}": v for k, v in sigs.get(rid, {}).items()} if rid else {}))
    L = pd.DataFrame(rows)
    L["label_ts"] = L["acc_time"].dt.floor("h")
    L.to_csv(C.RESULT_DIR / "anom_labels_v2.csv", index=False, encoding="utf-8-sig")

    print(pd.crosstab(L["group"], L["tier"], margins=True).to_string())
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    m = L.merge(v1[["acc_idx", "tier", "rfid_id"]], on="acc_idx", suffixes=("", "_v1"))
    rid = lambda s: pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    both = m[m["tier"].isin(["A", "AS", "S"]) & m["tier_v1"].eq("A")]
    print(f"v1 A {int((m['tier_v1'] == 'A').sum())}건 / v2 확실(AS·A·S) {int(m['tier'].isin(['AS', 'A', 'S']).sum())}건 / "
          f"둘 다 확실 {len(both)}건 중 같은 배 {(rid(both['rfid_id']) == rid(both['rfid_id_v1'])).sum()}건")
    print(L[L["tier"].isin(["AS", "A", "S"])].groupby(["group", "acc_type"]).size().to_string())
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
