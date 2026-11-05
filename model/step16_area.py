"""
[STEP16-1] 해역 단위 경보 ─ "어느 배가 사고 났나" 대신 "이 해역에서 지금 평소보다 이상한 배가 많은가"
  근거: step9_bag_robust·step14_recall 의 묶음 결과 (사고 3km 안 배들 급정지↑, 항구 사고엔 없음, 초과가 사고당 상한을 넘어 주변 배 공통 요인 의심)
  배를 특정하지 않으므로 사고 배가 자료에 없어도 평가 가능, 업종 제한도 풂
  [A] 사고 단위 검정 (실제 시각 vs 같은 자리 ±7·14·21·28일 위약, 사고 단위 짝 비교)
      지표: frac = 반경 안 바다 배 중 경보 배 비율. n_active 를 나누므로 "사고는 배 많은 곳에서 난다" 효과를 제외
      미리 정한 주 검정: 바다 사고, 급정지, 3km, ±6h, frac. 나머지는 해석용 (Holm 보정 표시)
      대조: 항구 사고(음성), 10~20km 고리(지역 공통 요인이면 고리도 오름), 사고 전 -6~-1h / 후 0~+6h, 같은 업종 / 다른 업종
            다른 업종은 앞선 실험에서 본 적 없는 표본
  [B] 운영 평가 (사고 범위 박스, 2km 격자, 3×3칸·6시간 합. 시각 t 까지 자료만)
      규칙: 경보 배-시간 ≥ k, 또는 포아송 놀람 상위 q (기대값 = 해역 활동량 × 시각별 경보율)
      재현 = 사고 시각~+6h 사이 사고 칸에 해역 경보. 위약 재현과의 차이가 진짜 잡은 몫
      정밀도 = 초과 적중 / 경보 건수 (6km·6시간 안 연속 경보는 1건). 손익 분기 = 1/(C+1)
  라벨로 임계값을 고르지 않음. 출력: results/step16_area_{test,ops}_<시각>.csv (rfid_id 없음)
  점수 캐시: features/step16_scores.npz (step13 soft rt·post, 급정지)
실행: python step16_area.py
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson

import fe_aggregate as A
import fe_config as C
import step13_stage2 as M
import step9_detect as D
import step9_labels as L9

STAMP = time.strftime("%m%d_%H%M")
SHIFTS = [-28, -21, -14, -7, 7, 14, 21, 28]
RINGS = {"2km": (0, 2), "3km": (0, 3), "5km": (0, 5), "고리10-20km": (10, 20)}
WINS = {"±6h": (-6, 6), "전-6~-1h": (-6, -1), "후0~+6h": (0, 6)}
CELL_KM, BOX_T = 2.0, 6
KS = [1, 2, 3, 5]
QS = [0.1, 0.05, 0.01, 0.001]
COSTS = [30, 100]
CACHE = C.FEAT_DIR / "step16_scores.npz"


def scores(H, S, y, fold, P):
    """step14 와 같은 방법으로 soft rt·post·급정지 계산 후 캐시 (행 순서 = M.load 결과)"""
    if CACHE.exists():
        Z = np.load(CACHE)
        if len(Z["soft_rt"]) == len(H):
            return dict(Z)
    X = M.context(H)
    row_of = pd.Series(np.arange(len(S)), index=S["rid"]).reindex(X["v"]).to_numpy()
    b = M.bits_all(X, y[row_of], P[row_of], fold[row_of], ~np.isin(X["v"], list(touched())))
    F = D.build(H)
    Z = dict(soft_rt=sum(b["soft"][k] for k in M.PRE).astype(np.float32),
             soft_post=(sum(b["soft"][k] for k in M.PRE + M.POST)).astype(np.float32),
             none_rt=sum(b["none"][k] for k in M.PRE).astype(np.float32),
             onset=F["onset"].to_numpy())
    np.savez_compressed(CACHE, **Z)
    return Z


def touched():
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv")
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    f = lambda d: set(pd.to_numeric(d["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    return f(L) | f(v1)


def load_acc():
    acc = pd.read_excel(L9.ACC)
    acc["group"] = acc.apply(L9.group, axis=1)
    acc["sea"] = acc["group"] != "port"
    acc["gear_i"] = acc["어업면허"].map(lambda g: L9.GEAR.get(g, g)).map({g: i for i, g in enumerate(M.GEARS)}).fillna(-1).astype(int)
    acc["t0"] = acc["발생일시"].dt.floor("h")
    return acc.reset_index(drop=True)


def in_range(t):
    return pd.Timestamp("2026-01-01") <= t <= pd.Timestamp("2026-06-30 23:00")


def accident_table(acc, ts, lon, lat, v, g, alerts):
    """사고·위약 × 반경 × 창 × 배 묶음 × 경보별 n_active, n_alert"""
    o = np.argsort(ts, kind="stable")
    ts_s, lon_s, lat_s, v_s, g_s = ts[o], lon[o], lat[o], v[o], g[o]
    al_s = {k: a[o] for k, a in alerts.items()}
    rows = []
    for i, a in acc.iterrows():
        for s in [0] + SHIFTS:
            t = a["t0"] + pd.Timedelta(days=s)
            if not in_range(t):
                continue
            t_ = np.datetime64(t)
            lo = np.searchsorted(ts_s, t_ - np.timedelta64(6, "h"))
            hi = np.searchsorted(ts_s, t_ + np.timedelta64(6, "h"), side="right")
            d = A.hav_km(lon_s[lo:hi], lat_s[lo:hi], a["경도(변환)"], a["위도(변환)"])
            rel = ((ts_s[lo:hi] - t_) / np.timedelta64(1, "h")).astype(int)
            near = d <= 20
            d, rel, vv, gg = d[near], rel[near], v_s[lo:hi][near], g_s[lo:hi][near]
            al = {k: x[lo:hi][near] for k, x in al_s.items()}
            for rn, (r0, r1) in RINGS.items():
                mr = (d >= r0) & (d <= r1)
                for wn, (w0, w1) in WINS.items():
                    mw = mr & (rel >= w0) & (rel <= w1)
                    for sn, ms in (("전체", mw), ("같은업종", mw & (gg == a["gear_i"])), ("다른업종", mw & (gg != a["gear_i"]))):
                        na = np.unique(vv[ms]).size
                        rec = dict(acc=i, shift=s, sea=a["sea"], 반경=rn, 창=wn, 묶음=sn, n_active=na)
                        for k, x in al.items():
                            rows.append(rec | dict(경보=k, n_alert=np.unique(vv[ms & x]).size))
    return pd.DataFrame(rows)


def test_table(T, rng):
    """사고 단위 짝 비교: frac 와 n_alert, any"""
    T = T.assign(frac=np.where(T["n_active"] > 0, T["n_alert"] / T["n_active"].clip(lower=1), np.nan), any_=(T["n_alert"] > 0).astype(float))
    out = []
    for key, G in T.groupby(["sea", "경보", "반경", "창", "묶음"]):
        real = G[G["shift"] == 0].set_index("acc")
        pl = G[G["shift"] != 0].groupby("acc")
        rec = dict(zip(["바다", "경보", "반경", "창", "묶음"], key))
        fp = pl["frac"].mean()
        idx = real.index[real["frac"].notna()].intersection(fp.dropna().index)
        d = (real.loc[idx, "frac"] - fp.loc[idx]).to_numpy()
        rec |= dict(사고=len(idx), 실제frac=real.loc[idx, "frac"].mean(), 위약frac=fp.loc[idx].mean(), 차이=d.mean(),
                    p=M.sign_flip(d, rng) if len(d) > 1 else np.nan, 실제활동=real.loc[idx, "n_active"].mean(), 위약활동=pl["n_active"].mean().loc[idx].mean(),
                    초과척=(real.loc[idx, "n_alert"] - pl["n_alert"].mean().loc[idx]).mean())
        out.append(rec)
    return pd.DataFrame(out)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def ops_eval(acc, ts, lon, lat, hod, alerts, rng):
    """박스 2km 격자 × 시간 밀집 배열로 해역 경보 계산"""
    lat0, lat1 = acc["위도(변환)"].min() - 0.15, acc["위도(변환)"].max() + 0.15
    lon0, lon1 = acc["경도(변환)"].min() - 0.15, acc["경도(변환)"].max() + 0.15
    dlat, dlon = CELL_KM / 110.574, CELL_KM / (111.32 * np.cos(np.radians(35.0)))
    ny, nx = int((lat1 - lat0) / dlat) + 1, int((lon1 - lon0) / dlon) + 1
    t00 = np.datetime64("2026-01-01T00")
    nt = int((np.datetime64("2026-07-01T00") - t00) / np.timedelta64(1, "h"))
    mi = (lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1)
    iy, ix = ((lat[mi] - lat0) / dlat).astype(int), ((lon[mi] - lon0) / dlon).astype(int)
    it = ((ts[mi] - t00) / np.timedelta64(1, "h")).astype(int)
    ok = (it >= 0) & (it < nt)
    idx = (it[ok], iy[ok], ix[ok])
    print(f"격자 {ny}×{nx}×{nt}, 바다 시간 {ok.sum():,}", flush=True)

    def box(x):
        """3×3칸, 현재 포함 지난 6시간 합 (시각 t 까지만)"""
        c = np.cumsum(x, 0, dtype=np.int32)
        c[BOX_T:] = c[BOX_T:] - c[:-BOX_T]
        p = np.pad(c, ((0, 0), (1, 1), (1, 1)))
        return p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:] + p[:, 1:-1, :-2] + p[:, 1:-1, 1:-1] + p[:, 1:-1, 2:] + p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:]

    N0 = np.zeros((nt, ny, nx), np.int16)
    np.add.at(N0, idx, 1)
    N = box(N0)
    del N0
    hod_b = hod[mi][ok]
    acc_c = [(int((a["위도(변환)"] - lat0) / dlat), int((a["경도(변환)"] - lon0) / dlon)) for _, a in acc.iterrows()]
    rows = []
    for name, al in alerts.items():
        x = al[mi][ok]
        Ax0 = np.zeros((nt, ny, nx), np.int16)
        np.add.at(Ax0, (idx[0][x], idx[1][x], idx[2][x]), 1)
        Ax = box(Ax0)
        del Ax0
        rate_h = np.bincount(hod_b[x], minlength=24) / np.bincount(hod_b, minlength=24).clip(1)
        # 포아송 놀람: 활동 칸만 계산. 기대값은 해역 활동량 × 최근 6시간 평균 경보율 (시각별)
        t_i, y_i, x_i = np.nonzero(N)
        r6 = np.convolve(np.r_[rate_h, rate_h], np.ones(BOX_T) / BOX_T, "full")[BOX_T - 1:BOX_T + 23]  # hod h: h-5..h 평균, 0~4시는 전날 포함
        h_now = (t_i % 24).astype(int)
        lam = N[t_i, y_i, x_i] * r6[h_now]
        ka = Ax[t_i, y_i, x_i]
        sur = -poisson.logsf(ka - 1, lam.clip(1e-9))
        sur[ka == 0] = 0.0
        rules = {f"배-시간≥{k}": ka >= k for k in KS}
        pos = sur[ka > 0]
        for q in QS:
            rules[f"놀람 상위{q:g}"] = (ka > 0) & (sur >= np.quantile(sur, 1 - q)) if len(pos) else ka < 0
        for rn, on in rules.items():
            hit, plc = [], []
            al_set = set(np.ravel_multi_index((t_i[on], y_i[on], x_i[on]), (nt, ny, nx)).tolist())
            for (cy, cx), (_, a) in zip(acc_c, acc.iterrows()):
                def any_hit(t):
                    h = int((np.datetime64(t) - t00) / np.timedelta64(1, "h"))
                    return any(np.ravel_multi_index((hh, cy, cx), (nt, ny, nx)) in al_set for hh in range(h, min(h + 7, nt)))
                hit.append(any_hit(a["t0"]))
                pv = [any_hit(a["t0"] + pd.Timedelta(days=s)) for s in SHIFTS if in_range(a["t0"] + pd.Timedelta(days=s))]
                plc.append(np.mean(pv) if pv else np.nan)
            hit, plc = np.array(hit, float), np.array(plc, float)
            cs = acc["sea"].to_numpy()
            d = (hit - plc)[cs & ~np.isnan(plc)]
            dp = (hit - plc)[~cs & ~np.isnan(plc)]
            # 경보 건수: 6km 칸·6시간 안 연속은 1건
            E = pd.DataFrame({"t": t_i[on], "c": (y_i[on] // 3) * 100000 + x_i[on] // 3}).drop_duplicates().sort_values(["c", "t"])
            new = np.r_[True, (E["c"].to_numpy()[1:] != E["c"].to_numpy()[:-1]) | (np.diff(E["t"].to_numpy()) > 6)] if len(E) else np.array([], bool)
            ev = int(new.sum())
            ex = d.mean() * len(d)
            rec = dict(경보=name, 규칙=rn, 활동칸시간비율=on.mean(), 경보건=ev, 일평균=ev / 181, 바다재현=hit[cs].mean(), 바다위약=np.nanmean(plc[cs]), 차이=d.mean(), p=M.sign_flip(d, rng) if len(d) else np.nan,
                       항구재현=hit[~cs].mean(), 항구위약=np.nanmean(plc[~cs]), 항구차이=dp.mean() if len(dp) else np.nan, 초과적중=ex, 초과정밀도=ex / max(ev, 1))
            for c in COSTS:
                rec[f"순이득C{c}"] = c * ex - ev
            rows.append(rec)
        print(f"{name} 운영 평가 끝", flush=True)
        del Ax
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    H, S, y, fold, P = M.load()
    Z = scores(H, S, y, fold, P)
    print(f"점수 준비 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    row_of = pd.Series(np.arange(len(S)), index=S["rid"]).reindex(H["rid"].to_numpy()).to_numpy()
    sea = H["at_sea"].fillna(0).to_numpy() == 1
    st = H["state"].fillna(4).to_numpy().astype(int)
    tch = touched()
    H_ = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "home_lat", "home_lon"])
    acc = load_acc()
    box = (H_["home_lat"].between(acc["위도(변환)"].min() - 0.2, acc["위도(변환)"].max() + 0.2)
           & H_["home_lon"].between(acc["경도(변환)"].min() - 0.2, acc["경도(변환)"].max() + 0.2))
    neg = set(H_.loc[box, "rfid_id"].astype(str)) - tch
    ref = np.isin(H["rid"].to_numpy(), list(neg)) & sea
    s = Z["soft_rt"]
    alerts = {"급정지": Z["onset"].astype(bool) & sea,
              "soft rt 1%": (s > np.quantile(s[ref], 0.99)) & sea,
              "soft rt 5%": (s > np.quantile(s[ref], 0.95)) & sea,
              "정지표류(대조)": sea & np.isin(st, (1, 2))}
    for k, a in alerts.items():
        print(f"{k}: 바다 시간 중 {a[sea].mean():.3%}", flush=True)
    print(f"사고 바다 {acc['sea'].sum()} / 항구 {(~acc['sea']).sum()}", flush=True)

    ts = H["ts"].to_numpy()
    lon, lat = H["lon_c"].to_numpy(), H["lat_c"].to_numpy()
    good = sea & ~np.isnan(lon) & ~np.isnan(lat)
    g = y[row_of]
    v = pd.factorize(H["rid"])[0]
    T = accident_table(acc, ts[good], lon[good], lat[good], v[good], g[good], {k: a[good] for k, a in alerts.items()})
    print(f"사고 표 {len(T):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    rng = np.random.default_rng(0)
    R = test_table(T, rng)
    R.round(4).to_csv(C.RESULT_DIR / f"step16_area_test_{STAMP}.csv", index=False, encoding="utf-8-sig")
    show = R[(R["창"] == "±6h") & (R["묶음"] == "전체")].sort_values(["경보", "바다", "반경"])
    print(show.round(4).to_string(index=False), flush=True)
    print("\n[주 검정 표본 세부: 바다·3km·전체 / 같은·다른 업종·창별]", flush=True)
    print(R[R["바다"] & (R["반경"] == "3km")].sort_values(["경보", "묶음", "창"]).round(4).to_string(index=False), flush=True)
    fam = R[R["바다"] & (R["창"] == "±6h") & (R["묶음"] == "전체") & (R["반경"].isin(["2km", "3km", "5km"])) & (R["경보"] != "정지표류(대조)")].copy()
    fam["p_holm"] = holm(fam["p"].fillna(1))
    print("\n[Holm 보정, 경보 3 × 반경 3]\n" + fam[["경보", "반경", "사고", "실제frac", "위약frac", "차이", "p", "p_holm"]].round(4).to_string(index=False), flush=True)

    O = ops_eval(acc, ts[good], lon[good], lat[good], H["ts"].dt.hour.to_numpy()[good], {k: a[good] for k, a in alerts.items() if k != "정지표류(대조)"}, rng)
    O.round(4).to_csv(C.RESULT_DIR / f"step16_area_ops_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(O.round(4).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
