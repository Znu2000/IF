"""
[STEP13] 분류 → 이상탐지 2단계 ─ "이 업종의 평소"를 기준으로 시간대 이상 점수를 내고, 업종 정보의 값어치를 잼
  1단계 입력: step11_oof_all_rp.npy (그 배를 학습에 안 쓴 fold 모델의 업종 확률, 2026 만)
  업종 기준표 (다른 fold 의 사고 무관 선박으로 계산, 업종 × 시간대)
    trans : 직전 상태 → 이번 상태 전이 확률        run : 바다 정지·표류가 지금까지 L 시간 이상 이어질 확률
    loc   : 그 업종·월에 이 칸(0.1도)만큼 드문 곳에 있을 확률
    whole : (사후) 정지·표류 전체 길이             gap : (사후) 바다에서 k 시간 이상 끊김         trip : (사후) 항차가 이만큼 짧을 확률
  점수 = 각 확률의 -log2 합. rt = trans + run + loc (시각 t 까지만 사용), post = rt + whole + gap + trip
       자기 평소가 아니라 같은 업종 다른 배들과 비교 → 동일 선박 프로필을 만들지 않음. 가중은 라벨로 고르지 않음
  업종을 넣는 방식 4가지
    none 전체 공통 기준 / oracle 실제 업종(상한) / hard 1단계 예측 1개(클래스 가중 적용) / soft 1단계 확률로 기준표 혼합
  평가 (바다 사고)
    시각 : 라벨 A·AS(거리로만 고름) 와 S 포함(흔적으로 고른 순환 라벨) 따로. 1위 시각 ±6h, 사고 시각의 배 내부 순위
    묶음 : 사고 3km 안 같은 업종 배 전체, 사고 시각 vs ±7·14일 옮긴 위약. 항구 사고는 음성 대조
    선박 : 최고 점수·경보율의 활동량 보정 AUC
    경보 : 사고 무관 선박 바다 시간 상위 0.1% 를 경보로 둘 때 업종별 경보율 (업종 쏠림)
  출력: results/step13_{time,bag,vessel,alert}_<시각>.csv (rfid_id 없음), features/step13_bag_cand.parquet (후보 캐시)
실행: python step13_stage2.py [--region]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import cv_common as V
import fe_aggregate as A
import fe_config as C
import step9_detect as D
import step9_labels as L9

STAMP = time.strftime("%m%d_%H%M")
GEARS = list(C.CLASSES)
NG = len(GEARS)                 # 기준표의 마지막 칸(NG) = 업종 무시 공통
W_HARD = np.array([1.35, 1.0, 1.492, 1.105])   # step11 all_rp 클래스 가중
CELL = 0.1
LAT0, LON0, NLAT, NLON = 30.0, 122.0, 100, 120
MAXRUN, MAXGAP, MAXTRIP = 72, 48, 240
ALERT_Q = 0.999
SHIFTS = [-14, -7, 7, 14]
BAG_KM = 3.0
N_PERM = 10000
PRE = ["trans", "run", "loc"]
POST = ["whole", "gap", "trip"]
METHODS = ["none", "oracle", "hard", "soft"]


def load():
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=D.COLS + ["lat_c", "lon_c"])
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.sort_values(["rid", "ts"], kind="stable").reset_index(drop=True)
    S, y = V.load_ship()
    S["rid"] = S["rfid_id"].astype(str)
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    fold = np.empty(len(y), int)
    for f, v in enumerate(R["fold_val_idx"]):
        fold[np.array(v)] = f
    H = H[H["rid"].isin(set(S["rid"]))].reset_index(drop=True)
    return H, S, y, fold, np.load(C.RESULT_DIR / "step11_oof_all_rp.npy")


def context(H):
    """시간대 배열 사전: 상태·연속 길이·항차"""
    v = H["rid"].to_numpy()
    ts = H["ts"].to_numpy()
    st = H["state"].fillna(4).to_numpy().astype(int).clip(0, 4)
    sea = H["at_sea"].fillna(0).to_numpy() == 1
    n = len(H)
    gap = np.r_[np.inf, (ts[1:] - ts[:-1]) / np.timedelta64(1, "h")]
    same = np.r_[False, v[1:] == v[:-1]]
    near = same & (gap <= 4)
    prev = np.where(near, np.r_[0, st[:-1]], 5)
    stop = sea & np.isin(st, (1, 2))
    # 정지·표류 연속 (2시간 넘게 끊기면 새 구간)
    cont = np.r_[False, stop[:-1]] & stop & same & (gap <= 2)
    run_id = np.cumsum(~cont)
    run_now = (pd.Series(1, index=range(n)).groupby(run_id).cumsum().to_numpy() * stop).clip(0, MAXRUN)
    run_all = (pd.Series(1, index=range(n)).groupby(run_id).transform("sum").to_numpy() * stop).clip(0, MAXRUN)
    run_start = stop & ~cont
    # 항차: 바다 시간 연속 (6시간 넘게 끊기면 새 항차)
    tcont = np.r_[False, sea[:-1]] & sea & same & (gap <= 6)
    trip_id = np.cumsum(~tcont)
    tt = pd.Series(ts).groupby(trip_id)
    dur = ((tt.transform("max") - tt.transform("min")) / pd.Timedelta(hours=1)).to_numpy()
    trip_len = (np.round(dur).astype(int) * sea).clip(0, MAXTRIP)
    trip_start = sea & ~tcont
    gn = H["gap_next_h"].fillna(1).to_numpy()
    gap_h = np.where(sea & (gn >= 3) & (gn < 999), np.minimum(gn, MAXGAP), 0).astype(int)
    cell = (np.floor((H["lat_c"].fillna(0).to_numpy() - LAT0) / CELL).astype(int).clip(0, NLAT - 1) * NLON
            + np.floor((H["lon_c"].fillna(0).to_numpy() - LON0) / CELL).astype(int).clip(0, NLON - 1))
    return dict(v=v, ts=ts, st=st, sea=sea, prev=prev, hod=H["ts"].dt.hour.to_numpy(), mon=H["ts"].dt.month.to_numpy() - 1,
                run_now=run_now, run_all=run_all, run_start=run_start, trip_len=trip_len, trip_start=trip_start,
                gap_h=gap_h, cell=cell)


def add_pool(c):
    """gear 축 첫 번째, 마지막 칸에 전체 합"""
    c[NG] = c[:NG].sum(0)
    return c


def surv(c):
    """마지막 축 길이별 개수 → P(길이 ≥ L), 0 칸은 1"""
    tot = c.sum(-1, keepdims=True)
    s = (np.cumsum(c[..., ::-1], -1)[..., ::-1] + 0.5) / (tot + 1)
    s[..., 0] = 1.0
    return s


def tables(X, ref, g):
    """ref: 기준 행 마스크, g: 행별 실제 업종 번호"""
    r = np.flatnonzero(ref)
    gr, hod, sea = g[r], X["hod"][r], X["sea"][r]
    T = {}
    c = np.zeros((NG + 1, 24, 6, 5))
    np.add.at(c, (gr, hod, X["prev"][r], X["st"][r]), 1)
    add_pool(c)
    T["trans"] = (c + 0.5) / (c.sum(-1, keepdims=True) + 2.5)

    c = np.zeros((NG + 1, 24, MAXRUN + 1))
    np.add.at(c, (gr[sea], hod[sea], X["run_now"][r][sea]), 1)
    T["run"] = surv(add_pool(c))

    c = np.zeros((NG + 1, 6, NLAT * NLON))
    np.add.at(c, (gr[sea], X["mon"][r][sea], X["cell"][r][sea]), 1)
    add_pool(c)
    tot = c.sum(-1, keepdims=True).clip(1)
    order = np.argsort(c, -1)
    tail = np.empty_like(c)
    np.put_along_axis(tail, order, np.cumsum(np.take_along_axis(c, order, -1), -1) / tot, -1)
    T["loc"] = np.maximum(tail, 0.5 / tot)

    rs = X["run_start"][r]
    c = np.zeros((NG + 1, 24, MAXRUN + 1))
    np.add.at(c, (gr[rs], hod[rs], X["run_all"][r][rs]), 1)
    T["whole"] = surv(add_pool(c))

    c = np.zeros((NG + 1, MAXGAP + 1))
    np.add.at(c, (gr[sea], X["gap_h"][r][sea]), 1)
    T["gap"] = surv(add_pool(c))

    ts_ = X["trip_start"][r]
    c = np.zeros((NG + 1, MAXTRIP + 1))
    np.add.at(c, (gr[ts_], X["trip_len"][r][ts_]), 1)
    add_pool(c)
    T["trip"] = (np.cumsum(c, -1) + 0.5) / (c.sum(-1, keepdims=True) + 1)
    return T


def lookup(T, X, i):
    """행 i 의 업종별 확률 (행 × 업종+공통). 사건이 아닌 행은 1"""
    hod, sea = X["hod"][i], X["sea"][i]
    out = {"trans": T["trans"][:, hod, X["prev"][i], X["st"][i]].T,
           "run": T["run"][:, hod, X["run_now"][i]].T,
           "loc": T["loc"][:, X["mon"][i], X["cell"][i]].T,
           "whole": T["whole"][:, hod, X["run_all"][i]].T,
           "gap": T["gap"][:, X["gap_h"][i]].T,
           "trip": T["trip"][:, X["trip_len"][i]].T}
    out["whole"] = np.where(X["run_start"][i][:, None], out["whole"], 1.0)
    out["trip"] = np.where(sea[:, None], out["trip"], 1.0)
    return out


def bits_all(X, g, P_row, fold_row, ref_ok):
    n = len(X["v"])
    b = {m: {k: np.zeros(n, np.float32) for k in PRE + POST} for m in METHODS}
    hard = (P_row * W_HARD).argmax(1)
    pi = P_row / P_row.sum(1, keepdims=True)
    for f in np.unique(fold_row):
        i = np.flatnonzero(fold_row == f)
        T = tables(X, (fold_row != f) & ref_ok, g)
        L = lookup(T, X, i)
        for k, p in L.items():
            b["none"][k][i] = -np.log2(p[:, NG])
            b["oracle"][k][i] = -np.log2(p[np.arange(len(i)), g[i]])
            b["hard"][k][i] = -np.log2(p[np.arange(len(i)), hard[i]])
            b["soft"][k][i] = -np.log2((p[:, :NG] * pi[i]).sum(1))
    return b


def score_sets(b, X, base):
    S_ = {"기준: 급정지": base}
    for m in METHODS:
        rt = sum(b[m][k] for k in PRE)
        S_[f"{m} rt"] = rt * X["sea"]
        S_[f"{m} post"] = (rt + sum(b[m][k] for k in POST)) * X["sea"]
    for m in ("none", "oracle"):
        for k in PRE + POST:
            S_[f"{m} [{k}]"] = b[m][k] * X["sea"]
    return S_


def slices(v):
    """정렬된 선박 배열 → rid: (시작, 끝)"""
    u, st = np.unique(v, return_index=True)
    return dict(zip(u, zip(st, np.r_[st[1:], len(v)])))


def time_eval(S_, X, pos, sl):
    rows = []
    for name, s in S_.items():
        for lab, g in pos.items():
            r = []
            for p in g.itertuples():
                a, e = sl[p.rid]
                m = X["sea"][a:e]
                ss = s[a:e][m]
                if len(ss) == 0:
                    continue
                rel = np.abs((X["ts"][a:e][m] - np.datetime64(p.label_ts)) / np.timedelta64(1, "h"))
                near = rel <= 1
                r.append(dict(hit6=rel[ss.argmax()] <= 6, rand6=(rel <= 6).mean(),
                              pct=(ss < ss[near].max()).mean() if near.any() else np.nan))
            r = pd.DataFrame(r)
            rows.append(dict(점수=name, 라벨=lab, n=len(r), 적중6h_건=int(r["hit6"].sum()), 무작위기대_건=r["rand6"].sum(),
                             순위백분위_중앙=r["pct"].median(), 순위백분위_평균=r["pct"].mean()))
    return pd.DataFrame(rows)


def bag_candidates():
    path = C.FEAT_DIR / "step13_bag_cand.parquet"
    if path.exists():
        return pd.read_parquet(path)
    acc = pd.read_excel(L9.ACC)
    acc["group"] = acc.apply(L9.group, axis=1)
    acc["gear"] = acc["어업면허"].map(lambda g: L9.GEAR.get(g, g))
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code"])
    gear_of = dict(zip(S["rfid_id"].astype(str), S["fishing_code"]))
    shifted = pd.concat([acc.assign(발생일시=acc["발생일시"] + pd.Timedelta(days=s)) for s in [0] + SHIFTS])
    shifted = shifted[shifted["발생일시"].between("2026-01-01", "2026-06-30 23:59")]
    P = L9.load_points(shifted)
    P["gear"] = P["rid"].map(gear_of)
    rows = []
    for s in [0] + SHIFTS:
        for i, a in acc.iterrows():
            t = a["발생일시"] + pd.Timedelta(days=s)
            if not pd.Timestamp("2026-01-01") <= t <= pd.Timestamp("2026-06-30 23:59"):
                continue
            W = P[(P["gear"] == a["gear"]) & (P["ts"] >= t.floor("h") - pd.Timedelta(hours=2)) & (P["ts"] <= t.floor("h") + pd.Timedelta(hours=2))]
            if W.empty:
                continue
            lon = W[[f"b_lon{j}" for j in range(6)]].to_numpy()
            lat = W[[f"b_lat{j}" for j in range(6)]].to_numpy()
            tm = W["ts"].to_numpy()[:, None] + (np.arange(6) * 10 + 5).astype("timedelta64[m]")
            ok = np.abs((tm - np.datetime64(t)) / np.timedelta64(1, "m")) <= L9.NEAR_MIN
            d = np.where(ok & ~np.isnan(lon), A.hav_km(lon, lat, a["경도(변환)"], a["위도(변환)"]), np.inf)
            per = pd.Series(d.min(1), index=W["rid"].to_numpy()).groupby(level=0).min()
            per = per[per <= BAG_KM]
            rows += [dict(acc_idx=i, shift=s, group=a["group"], rid=r, t=t.floor("h")) for r in per.index]
    B = pd.DataFrame(rows)
    B.to_parquet(path, index=False)
    return B


def sign_flip(d, rng):
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return ((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (N_PERM + 1)


def bag_eval(S_, X, B, sl, rng):
    per = {}
    for name, s in S_.items():
        vals = np.zeros(len(B))
        for j, c in enumerate(B.itertuples()):
            if c.rid not in sl:
                continue
            a, e = sl[c.rid]
            m = X["sea"][a:e]
            ss = s[a:e][m]
            if len(ss) == 0:
                continue
            w = np.abs((X["ts"][a:e][m] - np.datetime64(c.t)) / np.timedelta64(1, "h")) <= 6
            vals[j] = (ss < ss[w].max()).mean() if w.any() else 0.0
        per[name] = B.assign(v=vals).groupby(["group2", "acc_idx", "shift"])["v"].mean()
    rows, diffs = [], {}
    for name, v in per.items():
        for grp in ["sea", "port"]:
            g = v.xs(grp, level="group2")
            real = g.xs(0, level="shift")
            plc = g[g.index.get_level_values("shift") != 0].groupby("acc_idx").mean()
            idx = real.index.intersection(plc.index)
            d = (real.loc[idx] - plc.loc[idx]).to_numpy()
            diffs[name, grp] = pd.Series(d, index=idx)
            base = diffs.get(("none " + name.split()[-1], grp)) if name.split()[0] in METHODS and name.split()[-1] in ("rt", "post") else None
            rows.append(dict(점수=name, 묶음=grp, 사고수=len(idx), 실제=real.loc[idx].mean(), 위약=plc.loc[idx].mean(), 차이=d.mean(),
                             p=sign_flip(d, rng),
                             차이_대비none=(d - base.reindex(idx).to_numpy()).mean() if base is not None else np.nan,
                             p_대비none=sign_flip(d - base.reindex(idx).to_numpy(), rng) if base is not None and name.split()[0] != "none" else np.nan))
    return pd.DataFrame(rows)


def vessel_eval(S_, X, pos, neg, sl):
    rows = []
    sea_h = pd.Series(X["sea"]).groupby(X["v"]).sum()
    neg_mask = np.isin(X["v"], neg) & X["sea"]
    for name, s in S_.items():
        thr = np.quantile(s[neg_mask], ALERT_Q)
        df = pd.DataFrame({"v": X["v"], "s": s, "al": (s > thr) & X["sea"]})
        vmax = df.groupby("v")["s"].max()
        rate = df.groupby("v")["al"].sum() / sea_h.clip(lower=1)
        for lab, g in pos.items():
            vid = np.r_[g["rid"].to_numpy(), neg]
            y = np.r_[np.ones(len(g)), np.zeros(len(neg))]
            a = sea_h.reindex(vid).fillna(0).to_numpy()
            rows.append(dict(점수=name, 라벨=lab, n=len(g),
                             최고점_보정AUC=D.strat_auc(y, vmax.reindex(vid).fillna(0).to_numpy(), a, nbin=5),
                             경보율_보정AUC=D.strat_auc(y, rate.reindex(vid).fillna(0).to_numpy(), a, nbin=5)))
    return pd.DataFrame(rows)


def alert_eval(S_, X, g, neg, pos, sl):
    rows = []
    neg_mask = np.isin(X["v"], neg) & X["sea"]
    for name, s in S_.items():
        if name.split()[-1] not in ("rt", "post") and name != "기준: 급정지":
            continue
        thr = np.quantile(s[neg_mask], ALERT_Q)
        rec = dict(점수=name, 임계값=thr)
        rates = []
        for k, gear in enumerate(GEARS):
            m = neg_mask & (g == k)
            rates.append((s[m] > thr).mean() * 1000)
            rec[f"경보_천시간당_{gear}"] = rates[-1]
        rec["업종_최대최소비"] = max(rates) / max(min(rates), 1e-3)
        hit = 0
        for p in pos.itertuples():
            a, e = sl[p.rid]
            m = X["sea"][a:e]
            rel = np.abs((X["ts"][a:e][m] - np.datetime64(p.label_ts)) / np.timedelta64(1, "h")) <= 6
            hit += bool((s[a:e][m][rel] > thr).any())
        rec["±6h안_경보_사고수"] = hit
        rec["사고수"] = len(pos)
        rows.append(rec)
    return pd.DataFrame(rows)


def main(a):
    t0 = time.time()
    H, S, y, fold, P = load()
    X = context(H)
    row_of = pd.Series(np.arange(len(S)), index=S["rid"]).reindex(X["v"]).to_numpy()
    g, fold_row, P_row = y[row_of], fold[row_of], P[row_of]

    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    ref_ok = ~np.isin(X["v"], list(touched))
    sl = slices(X["v"])
    sea_l = L[(L["group"] != "port") & L["rid"].isin(sl)]
    pos = {"A·AS(거리)": sea_l[sea_l["tier"].isin(["A", "AS"])].drop_duplicates("rid"),
           "A·AS·S(순환 포함)": sea_l[sea_l["tier"].isin(["A", "AS", "S"])].drop_duplicates("rid")}
    neg = np.array(sorted(set(sl) - touched))
    acc = (P.argmax(1) == y).mean()
    print(f"시간대 {len(H):,}행 / 선박 {len(sl):,}척 / 1단계 cv {acc:.4f} / 라벨 "
          + ", ".join(f"{k} {len(v)}" for k, v in pos.items()) + f" ({(time.time() - t0) / 60:.1f}분)", flush=True)

    b = bits_all(X, g, P_row, fold_row, ref_ok)
    F = D.build(H)
    base = np.where(F["onset"], 1 + F["f_drift"] / 6, 0.0)
    S_ = score_sets(b, X, base)
    print(f"점수 계산 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    if a.region:
        # 사고현황은 5개 해경서 관할만 있음 → 사고 위치 범위 안 모항 선박끼리 비교 (지역 효과 제거)
        acc = pd.read_excel(L9.ACC)
        H_ = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "home_lat", "home_lon"])
        box = (H_["home_lat"].between(acc["위도(변환)"].min() - 0.2, acc["위도(변환)"].max() + 0.2)
               & H_["home_lon"].between(acc["경도(변환)"].min() - 0.2, acc["경도(변환)"].max() + 0.2))
        neg_r = np.array(sorted(set(neg) & set(H_.loc[box, "rfid_id"].astype(str))))
        print(f"지역 안 정상 선박 {len(neg_r):,}척", flush=True)
        Vr = vessel_eval(S_, X, pos, neg_r, sl)
        Vr.round(4).to_csv(C.RESULT_DIR / f"step13_vessel_region_{STAMP}.csv", index=False, encoding="utf-8-sig")
        print(Vr.round(3).to_string(index=False), flush=True)
        print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")
        return

    Tm = time_eval(S_, X, pos, sl)
    Tm.round(4).to_csv(C.RESULT_DIR / f"step13_time_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Tm.round(3).to_string(index=False), flush=True)

    Al = alert_eval(S_, X, g, neg, pos["A·AS·S(순환 포함)"], sl)
    Al.round(4).to_csv(C.RESULT_DIR / f"step13_alert_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Al.round(3).to_string(index=False), flush=True)

    Ve = vessel_eval(S_, X, pos, neg, sl)
    Ve.round(4).to_csv(C.RESULT_DIR / f"step13_vessel_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Ve.round(3).to_string(index=False), flush=True)

    B = bag_candidates()
    B["group2"] = np.where(B["group"] == "port", "port", "sea")
    print(f"묶음 후보 {len(B):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    Bg = bag_eval(S_, X, B, sl, np.random.default_rng(0))
    Bg.round(4).to_csv(C.RESULT_DIR / f"step13_bag_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Bg.round(3).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", action="store_true", help="선박 판별만 지역 한정 정상 선박으로 다시 계산")
    main(ap.parse_args())
