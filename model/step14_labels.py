"""
[STEP14-1] 사고 라벨 v3 후보표 ─ 사고 시점 위치 + 다른 날 어장 지문 + 모항 지역으로 후보를 매기고, 답을 아는 가짜 사고로 규칙 정확도 측정
  후보 (같은 업종, 사고 뒤 항적은 쓰지 않음)
    d_now    : 사고 시각 ±90분 10분 평활 점과 최소 거리. 없으면 inf
    days_near: 사고 ±3일을 뺀 날 중 사고 위치 2km 안에서 일한(바다 정지·표류·조업) 날 수
    fp       : 그 작업 시간 / (그 배 작업 시간 + 20)
    in_jur   : 모항이 그 해경서 사고 범위 ±0.2도 안
    MOVE_KM(15km) 넘게 떨어진 기록이 있는 배는 사고 시각에 다른 곳에 있었으므로 지문 규칙 후보에서 제외
  규칙
    A      : 가장 가까운 배 1km 안 + (둘째의 2배 이상 또는 둘째 3km 밖) (v2 와 같음)
    A+관할 : 관할 안 후보로만 A
    지문   : 관할 안, days_near >= 3 인 배 중 fp 1위가 2위의 2배 이상
    결합   : 관할 안, log(fp) - d_now^2 / (2*1.5^2) 1위가 2위보다 log(4) 이상 (기록 없으면 거리 항 -2)
  가짜 사고: 지역 안 바다 시간 임의 추출, 업종은 실제 바다 사고 분포, 위치 오차 표준편차 --noise km(기본 1), 신고 시각 -1~+3시간 어긋남
    기록 있음 : 그대로 / 기록 없음 : 그 배의 ±90분 점 제거 / 자료에 없는 배 : 그 배를 후보에서 제거 (뭘 고르든 오답)
  출력: results/step14_rules_<시각>.csv (rfid_id 없음), results/anom_labels_v3_cand.parquet (rfid_id 포함 → 깃헙 업로드 금지)
실행: python step14_labels.py [--noise 1.0]
"""
import argparse
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C
import step13_stage2 as M
import step9_labels as L9

STAMP = time.strftime("%m%d_%H%M")
N_SIM = 300
R_FP = 2.0
EXCL_DAYS = 3
MOVE_KM = 15.0
JUR_PAD = 0.2
SIG_KM = 1.5
MIN_DAYS = 3
NO_REC = -2.0
GAP_FP, GAP_COMBO = np.log(2), np.log(4)
SHIFTS = [-28, -21, -14, -7, 7, 14, 21, 28]
RULES_FP = "지문"


def load():
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "ts", "lat_c", "lon_c", "state", "at_sea"])
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.drop(columns="rfid_id").sort_values("ts", kind="stable").reset_index(drop=True)
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code", "home_lat", "home_lon"])
    S["rid"] = S["rfid_id"].astype(str)
    H = H[H["rid"].isin(set(S["rid"]))].reset_index(drop=True)
    acc = pd.read_excel(L9.ACC)
    acc["group"] = acc.apply(L9.group, axis=1)
    acc["gear"] = acc["어업면허"].map(lambda g: L9.GEAR.get(g, g))
    return H, S.set_index("rid"), acc


def jur_boxes(acc):
    g = acc.groupby("관할해경서")
    return pd.DataFrame({"lat0": g["위도(변환)"].min() - JUR_PAD, "lat1": g["위도(변환)"].max() + JUR_PAD,
                         "lon0": g["경도(변환)"].min() - JUR_PAD, "lon1": g["경도(변환)"].max() + JUR_PAD})


def make_sims(H, S, acc, rng, NOISE):
    sea_acc = acc[acc["group"] != "port"]
    home_ok = (S["home_lat"].between(acc["위도(변환)"].min() - JUR_PAD, acc["위도(변환)"].max() + JUR_PAD)
               & S["home_lon"].between(acc["경도(변환)"].min() - JUR_PAD, acc["경도(변환)"].max() + JUR_PAD))
    gear_of = S["fishing_code"]
    Hs = H[(H["at_sea"] == 1) & H["rid"].isin(set(S.index[home_ok])) & H["lat_c"].notna()]
    gears = Hs["rid"].map(gear_of).to_numpy()
    freq = sea_acc["gear"].value_counts(normalize=True)
    rows = []
    for j, g in enumerate(rng.choice(freq.index.to_numpy(), N_SIM, p=freq.to_numpy())):
        idx = np.flatnonzero(gears == g)
        r = Hs.iloc[rng.choice(idx)]
        lat = r["lat_c"] + rng.normal(0, NOISE) / 111.0
        lon = r["lon_c"] + rng.normal(0, NOISE) / (111.0 * np.cos(np.radians(r["lat_c"])))
        t = r["ts"] + pd.Timedelta(minutes=float(rng.uniform(-60, 180)))
        t = min(max(t, pd.Timestamp("2026-01-01 02:00")), pd.Timestamp("2026-06-30 21:00"))
        near = ((sea_acc["위도(변환)"] - lat) ** 2 + (sea_acc["경도(변환)"] - lon) ** 2).to_numpy().argmin()
        rows.append(dict(eid=f"sim{j}", t=t, lat=lat, lon=lon, gear=g, group="sim", jur=sea_acc["관할해경서"].iloc[near], truth=r["rid"]))
    return pd.DataFrame(rows)


def candidates(E, H, S, P, boxes):
    """사건별 후보 표"""
    gear_of = S["fishing_code"]
    P = P.sort_values("ts", kind="stable")
    pts = P["ts"].to_numpy()
    p_lon = P[[f"b_lon{j}" for j in range(6)]].to_numpy()
    p_lat = P[[f"b_lat{j}" for j in range(6)]].to_numpy()
    p_rid, p_gear = P["rid"].to_numpy(), P["rid"].map(gear_of).to_numpy()
    work = (H["at_sea"].to_numpy() == 1) & H["state"].isin([1, 2, 3]).to_numpy() & H["lat_c"].notna().to_numpy()
    W = H[work]
    w_lat, w_lon, w_rid, w_day = W["lat_c"].to_numpy(), W["lon_c"].to_numpy(), W["rid"].to_numpy(), W["ts"].dt.floor("D").to_numpy()
    w_gear = W["rid"].map(gear_of).to_numpy()
    n_work_all = W.groupby("rid").size()
    hts = H["ts"].to_numpy()
    h_rid, h_sea, h_port = H["rid"].to_numpy(), H["at_sea"].to_numpy() == 1, H["state"].to_numpy() == 0
    rows = []
    for e in E.itertuples():
        t = np.datetime64(e.t)
        t0 = np.datetime64(pd.Timestamp(e.t).floor("h"))
        a, b = np.searchsorted(pts, t0 - np.timedelta64(2, "h")), np.searchsorted(pts, t0 + np.timedelta64(2, "h"), side="right")
        m = p_gear[a:b] == e.gear
        tm = pts[a:b][m][:, None] + (np.arange(6) * 10 + 5).astype("timedelta64[m]")
        ok = np.abs((tm - t) / np.timedelta64(1, "m")) <= L9.NEAR_MIN
        d = np.where(ok & ~np.isnan(p_lon[a:b][m]), A.hav_km(p_lon[a:b][m], p_lat[a:b][m], e.lon, e.lat), np.inf)
        d_now = pd.Series(d.min(1), index=p_rid[a:b][m]).groupby(level=0).min() if m.any() else pd.Series(dtype=float)
        d_now = d_now[np.isfinite(d_now)]

        box = (np.abs(w_lat - e.lat) < 0.03) & (np.abs(w_lon - e.lon) < 0.04) & (w_gear == e.gear)
        far = np.abs((w_day - np.datetime64(pd.Timestamp(e.t).floor("D"))) / np.timedelta64(1, "D")) > EXCL_DAYS
        i = np.flatnonzero(box & far)
        i = i[A.hav_km(w_lon[i], w_lat[i], e.lon, e.lat) <= R_FP]
        near = pd.DataFrame({"rid": w_rid[i], "day": w_day[i]})
        n_near = near.groupby("rid").size()
        days = near.groupby("rid")["day"].nunique()

        pool = sorted(set(d_now.index[d_now <= MOVE_KM]) | set(n_near.index) | ({e.truth} if isinstance(e.truth, str) else set()))
        if not pool:
            continue
        c, d_ = np.searchsorted(hts, t0 - np.timedelta64(1, "h")), np.searchsorted(hts, t0 + np.timedelta64(1, "h"), side="right")
        now = pd.DataFrame({"rid": h_rid[c:d_], "sea": h_sea[c:d_], "port": h_port[c:d_]}).groupby("rid").max()
        R = pd.DataFrame({"rid": pool})
        R["d_now"] = R["rid"].map(d_now).fillna(np.inf)
        R["n_near"] = R["rid"].map(n_near).fillna(0)
        R["days_near"] = R["rid"].map(days).fillna(0)
        R["fp"] = R["n_near"] / (R["rid"].map(n_work_all).fillna(0) + 20)
        R["sea_now"] = R["rid"].map(now["sea"]).eq(True)
        R["port_now"] = R["rid"].map(now["port"]).eq(True)
        bx = boxes.loc[e.jur]
        hl, hn = S.loc[R["rid"], "home_lat"].to_numpy(), S.loc[R["rid"], "home_lon"].to_numpy()
        R["in_jur"] = (hl >= bx["lat0"]) & (hl <= bx["lat1"]) & (hn >= bx["lon0"]) & (hn <= bx["lon1"])
        R["home_km"] = A.hav_km(hn, hl, e.lon, e.lat)
        R["eid"] = e.eid
        rows.append(R)
    return pd.concat(rows, ignore_index=True)


def rule_a(R):
    R = R[np.isfinite(R["d_now"])].sort_values("d_now")
    if R.empty:
        return None
    d1, d2 = R["d_now"].iloc[0], (R["d_now"].iloc[1] if len(R) > 1 else np.inf)
    return R["rid"].iloc[0] if d1 <= 1.0 and (d2 >= 2 * d1 or d2 > 3) else None


def top_gap(R, s, gap):
    if R.empty:
        return None
    o = np.argsort(-s)
    if len(o) == 1 or s[o[0]] - s[o[1]] >= gap:
        return R["rid"].iloc[o[0]]
    return None


def rule_fp(R):
    R = R[R["in_jur"] & (R["days_near"] >= MIN_DAYS) & (~np.isfinite(R["d_now"]) | (R["d_now"] <= MOVE_KM))]
    return top_gap(R, np.log(R["fp"].to_numpy() + 1e-6), GAP_FP)


def rule_combo(R):
    R = R[R["in_jur"] & (~np.isfinite(R["d_now"]) | (R["d_now"] <= MOVE_KM))]
    dd = R["d_now"].to_numpy()
    s = np.log(R["fp"].to_numpy() + 1e-3) + np.where(np.isfinite(dd), -dd ** 2 / (2 * SIG_KM ** 2), NO_REC)
    return top_gap(R, s, GAP_COMBO)


RULES = {"A": rule_a, "A+관할": lambda R: rule_a(R[R["in_jur"]]), "지문": rule_fp, "결합": rule_combo}


def sim_eval(Cn, sims):
    rows = []
    groups = dict(tuple(Cn.groupby("eid")))
    truth = sims.set_index("eid")["truth"]
    for mode in ["기록 있음", "기록 없음", "자료에 없는 배"]:
        res = {k: [] for k in RULES}
        for eid, tr in truth.items():
            R = groups.get(eid)
            if R is None:
                for k in RULES:
                    res[k].append((False, False))
                continue
            R = R.copy()
            if mode == "기록 없음":
                R.loc[R["rid"] == tr, "d_now"] = np.inf
            elif mode == "자료에 없는 배":
                R = R[R["rid"] != tr]
            for k, f in RULES.items():
                pick = f(R)
                res[k].append((pick is not None, pick == tr))
        for k, v in res.items():
            a = np.array(v)
            rows.append(dict(상황=mode, 규칙=k, 결정비율=a[:, 0].mean(), 정답비율=a[:, 1].mean(),
                             정밀도=a[:, 1].sum() / max(a[:, 0].sum(), 1), 결정수=int(a[:, 0].sum())))
    return pd.DataFrame(rows)


def main(a):
    t0 = time.time()
    rng = np.random.default_rng(0)
    H, S, acc = load()
    boxes = jur_boxes(acc)
    sims = make_sims(H, S, acc, rng, a.noise)
    real = pd.DataFrame(dict(eid=[f"acc{i}" for i in acc.index], t=acc["발생일시"], lat=acc["위도(변환)"], lon=acc["경도(변환)"],
                             gear=acc["gear"], group=acc["group"], jur=acc["관할해경서"], truth=None))
    E = pd.concat([real, sims], ignore_index=True)
    P = L9.load_points(E.rename(columns={"t": "발생일시"}))
    print(f"사건 {len(E)} / 평활 점 {len(P):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    Cn = candidates(E, H, S, P, boxes)
    print(f"후보 {len(Cn):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    Sm = sim_eval(Cn[Cn["eid"].str.startswith("sim")], sims)
    Sm.round(4).to_csv(C.RESULT_DIR / f"step14_rules_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Sm.round(3).to_string(index=False), flush=True)

    Cr = Cn[Cn["eid"].str.startswith("acc")].copy()
    Cr.to_parquet(C.RESULT_DIR / "anom_labels_v3_cand.parquet", index=False)
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv")
    L["eid"] = "acc" + L["acc_idx"].astype(str)
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v2 = L.set_index("eid")
    picks = []
    for eid, R in Cr.groupby("eid"):
        picks.append(dict(eid=eid, **{k: f(R) for k, f in RULES.items()}))
    Pk = pd.DataFrame(picks).set_index("eid").reindex(v2.index)
    Pk["group"], Pk["tier"], Pk["rid2"] = v2["group"], v2["tier"], v2["rid"]
    rows = []
    for k in RULES:
        for grp, g in [("바다", Pk[Pk["group"] != "port"]), ("항구", Pk[Pk["group"] == "port"])]:
            sure = g[g["tier"].isin(["A", "AS"])]
            rows.append(dict(규칙=k, 묶음=grp, 사고수=len(g), 결정=int(g[k].notna().sum()),
                             v2_none에서_결정=int(g.loc[g["tier"] == "none", k].notna().sum()),
                             v2_AAS_유지=int((sure[k] == sure["rid2"]).sum()), v2_AAS수=len(sure)))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    sure_sea = Pk[(Pk["group"] != "port") & Pk["tier"].isin(["A", "AS"])]
    sea_l = L[L["tier"].isin(["A", "AS"])].merge(Cr, on=["eid", "rid"], how="left")
    print("v2 A·AS 라벨의 관할 안 비율:", sea_l.groupby("group")["in_jur"].mean().round(3).to_dict())
    print("v2 A·AS 바다 라벨, 사고 ±1h 항구 기록 있음:", int(sea_l.loc[sea_l["group"] != "port", "port_now"].sum()),
          "/ 항구 라벨 항구 기록 있음:", int(sea_l.loc[sea_l["group"] == "port", "port_now"].sum()), flush=True)

    # 자료 포함 비율 w 추정: 실제 결정률 = w * 기록 있음 결정률 + (1 - w) * 자료에 없는 배 결정률
    Ms = Sm.set_index(["상황", "규칙"])
    real_rate = {k: Pk.loc[Pk["group"] != "port", k].notna().mean() for k in RULES}
    rows = []
    for k in RULES:
        d1, c1 = Ms.loc[("기록 있음", k), "결정비율"], Ms.loc[("기록 있음", k), "정답비율"]
        d0 = Ms.loc[("자료에 없는 배", k), "결정비율"]
        w = float(np.clip((real_rate[k] - d0) / max(d1 - d0, 1e-9), 0, 1))
        rows.append(dict(규칙=k, 실제결정률=real_rate[k], 추정w=w, 추정정밀도=w * c1 / max(w * d1 + (1 - w) * d0, 1e-9)))
    Wt = pd.DataFrame(rows)
    Wt.round(4).to_csv(C.RESULT_DIR / f"step14_mix_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Wt.round(3).to_string(index=False), flush=True)

    # 일치도: 가짜 사고에서 A 와 지문이 같은 배를 고르는 비율 vs 실제
    Cs = Cn[Cn["eid"].str.startswith("sim")]
    pa = {e: (rule_a(R), rule_fp(R)) for e, R in Cs.groupby("eid")}
    a_dec = [v for v in pa.values() if v[0] is not None]
    print(f"A 결정 중 지문이 같은 배: 가짜 {sum(v[0] == v[1] for v in a_dec)}/{len(a_dec)} / "
          f"실제 바다 {int((sure_sea[RULES_FP] == sure_sea['rid2']).sum())}/{len(sure_sea)}", flush=True)

    # 지문 검증: 시간을 안 쓰고 고른 배가 사고 시각에 사고 위치 3km 안에 있었나 vs 시각을 옮긴 위약
    H_ = H.set_index("rid").sort_index()
    sea_e = real[real["group"] != "port"].set_index("eid")
    rows, per = [], {}
    for name, min_days, gap, ranks in [("지문 1위·2배 차이", MIN_DAYS, GAP_FP, [0]), ("지문 1위", MIN_DAYS, 0.0, [0]), ("대조: 지문 2~6위", MIN_DAYS, 0.0, [1, 2, 3, 4, 5])]:
        for eid, R in Cr[Cr["eid"].isin(sea_e.index)].groupby("eid"):
            Q = R[R["in_jur"] & (R["days_near"] >= min_days)]
            if top_gap(Q, np.log(Q["fp"].to_numpy() + 1e-6), gap) is None:
                continue
            Q = Q.sort_values("fp", ascending=False)
            e = sea_e.loc[eid]
            for pick in Q["rid"].to_numpy()[[k for k in ranks if k < len(Q)]]:
                Hv = H_.loc[[pick]]
                for s in [0] + SHIFTS:
                    ts = e["t"] + pd.Timedelta(days=s)
                    if not pd.Timestamp("2026-01-01") <= ts <= pd.Timestamp("2026-06-30 23:59"):
                        continue
                    W_ = Hv[((Hv["ts"] - ts).abs() <= pd.Timedelta(hours=1)) & Hv["lat_c"].notna()]
                    d = A.hav_km(W_["lon_c"].to_numpy(), W_["lat_c"].to_numpy(), e["lon"], e["lat"]) if len(W_) else np.array([np.inf])
                    per.setdefault(name, []).append(dict(eid=eid, s=s, sea=bool((W_["at_sea"] == 1).any()), near=bool(d.min() <= 3)))
    rng2 = np.random.default_rng(0)
    for name, v in per.items():
        v = pd.DataFrame(v)
        for col in ["sea", "near"]:
            real_ = v[v["s"] == 0].groupby("eid")[col].mean()
            plc = v[v["s"] != 0].groupby("eid")[col].mean().reindex(real_.index)
            d = (real_ - plc).dropna().to_numpy()
            rows.append(dict(선택=name, 항목={"sea": "바다에 있음", "near": "사고 3km 안"}[col], 사고수=len(d), 실제=real_.mean(), 위약=plc.mean(), 차이=d.mean(), p=M.sign_flip(d, rng2)))
    Ck = pd.DataFrame(rows)
    Ck.round(4).to_csv(C.RESULT_DIR / f"step14_fpcheck_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Ck.round(3).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=1.0, help="가짜 사고 위치 오차 표준편차 km")
    main(ap.parse_args())
