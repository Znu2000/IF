"""
[STEP12] 조업행태 문서(조업형태분석/조업행태_이론vs실제)의 지식으로 만든 피처 ─ 기준 모델 all_rp (2025 정보 없는 재학습)
  피처 묶음 (선박 1행, 각자의 항적만 사용, 정답·다른 연도 사용 없음)
    trip     : 자정 넘김·다일 항차 비율, 출항·귀항 시각 분위수 (채낚기 자정 넘김 62.8%, 문어단지 04~05시 출항, 4톤급 12~16시 귀항)
    shift    : 월별 어장 이동 거리, 이웃 달·첫 달↔마지막 달 어장 겹침 (채낚기 이웃 달 상관 0.25~0.52, 연안 0.68 이상)
    soak     : 같은 칸 재방문 간격 8~15일, 16~30일, 30일 초과 비율 (통발 침지 3~30일, 약 15일 양승. rv_ 는 7일 초과를 한 칸으로 묶음)
    active   : 1~2월 대 3~6월 출어율, 월별 출어 변동, 쉬는 기간 (통발은 쉬는 달이 적음)
    calendar : 공휴일·설 연휴·화요일·주말 출어 (연도별 달력)
    ban      : 금어기 경계 전후 14일 출어율 변화 (살오징어 4/1, 주꾸미 5/11, 참문어 5/16, 대게·낙지 6/1, 꽃게 6/21) + 대구 금어기 출어율
    season_prior : 1~4월, 5~6월 조업 위치 각각의 지역 사전확률 (채낚기 1~4월 제주 → 5~6월 동해처럼 계절 위치가 업종을 가르는지)
  선별 : LightGBM 단독 5-fold cv (2026 만). 기준 대비 묶음별 추가 효과, fold 부호 확인. 최종 판단은 3모델 평균으로 다시 함
  출력 : features/theory_feats{,_test}.parquet (실제 rfid_id → 깃헙 금지), results/step12_screen_<시각>.csv
실행: python step12_theory.py --build
      python step12_theory.py --screen
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import cv_common as V
import fe_aggregate as A
import fe_config as C
import step1_experiments as E1
import train_classifiers as TC
from step4_behavior import EP_GAP_H, FISH_STATES
from step4_final import attach

HOLIDAYS = {2025: ["01-01", "01-27", "01-28", "01-29", "01-30", "03-03", "05-05", "05-06", "06-03", "06-06"],
            2026: ["01-01", "02-16", "02-17", "02-18", "03-02", "05-05", "05-25", "06-03"]}
SEOLLAL = {2025: ["01-28", "01-29", "01-30"], 2026: ["02-16", "02-17", "02-18"]}
BAN = {"squid": "04-01", "webfoot": "05-11", "octopus": "05-16", "crab_snow": "06-01", "crab_blue": "06-21"}
N_DAYS = 181  # 두 해 모두 평년, 1/1 ~ 6/30
BASE_PRIORS = dict(E1.PRIOR_SETS)
SEASON_PRIORS = {"early": ["th_early_lat", "th_early_lon"], "late": ["th_late_lat", "th_late_lon"]}
GROUPS = {"trip": "th_trip_", "shift": "th_shift_", "soak": "th_soak_", "active": "th_active_",
          "calendar": "th_cal_", "ban": "th_ban_"}
CONFIGS = ["base", "trip", "shift", "soak", "active", "calendar", "ban", "season_prior", "all"]


def doy(year, mmdd):
    return (pd.Timestamp(f"{year}-{mmdd}") - pd.Timestamp(f"{year}-01-01")).days


def trip_feats(H):
    H = H.sort_values(["rfid_id", "t"], kind="stable")
    sea = H.at_sea.to_numpy() == 1
    same = H.rfid_id.eq(H.rfid_id.shift()).to_numpy()
    prev_sea = np.r_[False, sea[:-1]]
    gap = np.r_[0, np.diff(H.t.to_numpy())]
    new = sea & ~(same & prev_sea & (gap <= EP_GAP_H))
    tid = np.cumsum(new)
    P = H.loc[sea, ["rfid_id", "t", "hour", "doy"]].assign(tid=tid[sea])
    g = P.groupby("tid")
    T = pd.DataFrame({"rfid_id": g.rfid_id.first(), "dep_h": g.hour.first(), "ret_h": g.hour.last(), "d0": g.doy.first(),
                      "d1": g.doy.last(), "len_h": g.t.last() - g.t.first() + 1})
    v = T.groupby("rfid_id")
    out = pd.DataFrame({
        "th_trip_n": v.size(), "th_trip_overnight": (T.d1 > T.d0).groupby(T.rfid_id).mean(),
        "th_trip_multiday": (T.len_h > 24).groupby(T.rfid_id).mean(), "th_trip_short_lt4": (T.len_h < 4).groupby(T.rfid_id).mean(),
        "th_trip_len_q25": v.len_h.quantile(0.25), "th_trip_len_q75": v.len_h.quantile(0.75),
        "th_trip_dep_q25": v.dep_h.quantile(0.25), "th_trip_dep_q75": v.dep_h.quantile(0.75),
        "th_trip_ret_q25": v.ret_h.quantile(0.25), "th_trip_ret_q50": v.ret_h.quantile(0.5), "th_trip_ret_q75": v.ret_h.quantile(0.75),
        "th_trip_dep_0309": T.dep_h.between(3, 9).groupby(T.rfid_id).mean(), "th_trip_dep_1520": T.dep_h.between(15, 20).groupby(T.rfid_id).mean(),
    })
    out["th_trip_midnight_hours"] = P.hour.isin([22, 23, 0, 1, 2, 3]).groupby(P.rfid_id).mean()
    return out


def shift_feats(F):
    F = F.assign(cell=np.floor(F.lon_c / 0.02).astype(np.int64) * 100000 + np.floor(F.lat_c / 0.02).astype(np.int64))
    M = F.groupby(["rfid_id", "month"])[["lon_c", "lat_c"]].median()
    rows = {}
    cells = F.groupby(["rfid_id", "month"])["cell"].agg(set)
    for v, c in cells.groupby(level=0):
        c = c.droplevel(0)
        ms = c.index.to_numpy()
        pts = M.loc[v].loc[ms]
        d = A.hav_km(pts.lon_c.to_numpy()[1:], pts.lat_c.to_numpy()[1:], pts.lon_c.to_numpy()[:-1], pts.lat_c.to_numpy()[:-1]) if len(ms) > 1 else np.array([])
        jac = [len(a & b) / len(a | b) for a, b in zip(c.values[:-1], c.values[1:])]
        r = dict(th_shift_months=len(ms))
        if len(ms) > 1:
            r.update(th_shift_km_mean=float(d.mean()), th_shift_km_max=float(d.max()),
                     th_shift_first_last_km=float(A.hav_km(pts.lon_c.iloc[0], pts.lat_c.iloc[0], pts.lon_c.iloc[-1], pts.lat_c.iloc[-1])),
                     th_shift_jac_mean=float(np.mean(jac)), th_shift_jac_min=float(np.min(jac)),
                     th_shift_jac_first_last=len(c.values[0] & c.values[-1]) / len(c.values[0] | c.values[-1]))
        rows[v] = r
    out = pd.DataFrame.from_dict(rows, orient="index")
    for name, ms in [("early", [1, 2, 3, 4]), ("late", [5, 6])]:
        L = F[F.month.isin(ms)].groupby("rfid_id")[["lon_c", "lat_c"]].median()
        out[f"th_{name}_lon"], out[f"th_{name}_lat"] = L.lon_c, L.lat_c
    out["th_shift_early_late_km"] = A.hav_km(out.th_early_lon, out.th_early_lat, out.th_late_lon, out.th_late_lat)
    return out


def soak_feats(F):
    c = F.assign(cx=np.floor(F.lon_c / 0.005).astype(np.int64), cy=np.floor(F.lat_c / 0.005).astype(np.int64))
    c = c.sort_values(["rfid_id", "cx", "cy", "t"])
    same = (c.rfid_id.eq(c.rfid_id.shift()) & c.cx.eq(c.cx.shift()) & c.cy.eq(c.cy.shift())).to_numpy()
    ep = c[~same | (c.t.diff().to_numpy() > EP_GAP_H)].copy()
    ep_same = (ep.rfid_id.eq(ep.rfid_id.shift()) & ep.cx.eq(ep.cx.shift()) & ep.cy.eq(ep.cy.shift())).to_numpy()
    ep["iv"] = np.where(ep_same, ep.t.diff(), np.nan) / 24.0
    iv = ep.dropna(subset=["iv"])
    out = {}
    for name, lo, hi in [("d8_15", 7.5, 15.5), ("d16_30", 15.5, 30.5), ("gt30", 30.5, 1e9), ("d13_17", 12.5, 17.5)]:
        out[f"th_soak_{name}"] = iv.iv.between(lo, hi, inclusive="left").groupby(iv.rfid_id).mean()
    ge2 = iv[iv.iv >= 1.5]
    out["th_soak_med_ge2d"] = ge2.groupby("rfid_id").iv.median()
    return pd.DataFrame(out)


def calendar_feats(H, year):
    S = H[H.at_sea == 1][["rfid_id", "doy"]].drop_duplicates()
    ids, vi = np.unique(S.rfid_id.to_numpy(), return_inverse=True)
    M = np.zeros((len(ids), N_DAYS), bool)
    M[vi, S.doy.to_numpy()] = True
    days = pd.date_range(f"{year}-01-01", periods=N_DAYS)
    rate = lambda mask: M[:, mask].mean(1) if mask.any() else np.full(len(ids), np.nan)
    hol = np.isin(days.strftime("%m-%d"), HOLIDAYS[year])
    seol = np.isin(days.strftime("%m-%d"), SEOLLAL[year])
    wd = days.weekday.to_numpy()
    all_rate = M.mean(1)
    safe = lambda x: np.divide(x, all_rate, out=np.full(len(ids), np.nan), where=all_rate > 0)
    month = days.month.to_numpy()
    mon = np.stack([rate(month == m) for m in range(1, 7)], 1)
    out = {"th_active_winter": mon[:, :2].mean(1), "th_active_spring": mon[:, 2:].mean(1),
           "th_active_winter_ratio": np.divide(mon[:, :2].mean(1), mon[:, 2:].mean(1), out=np.full(len(ids), np.nan), where=mon[:, 2:].mean(1) > 0),
           "th_active_month_cv": np.divide(mon.std(1), mon.mean(1), out=np.full(len(ids), np.nan), where=mon.mean(1) > 0),
           "th_active_min_month": mon.min(1), "th_active_months_ge20": (mon >= 0.2).sum(1).astype(float),
           "th_cal_holiday": safe(rate(hol)), "th_cal_seollal": rate(seol),
           "th_cal_tue_nohol": safe(rate((wd == 1) & ~hol)), "th_cal_weekend_nohol": safe(rate((wd >= 5) & ~hol))}
    rest_max, rest_ge7 = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    for i in range(len(ids)):
        d = np.flatnonzero(M[i])
        if len(d) > 1:
            g = np.diff(d) - 1
            rest_max[i], rest_ge7[i] = g.max(), (g >= 7).sum()
    out["th_active_rest_max"], out["th_active_rest_ge7"] = rest_max, rest_ge7
    for name, mmdd in BAN.items():
        b = doy(year, mmdd)
        out[f"th_ban_{name}"] = M[:, b:min(b + 14, N_DAYS)].mean(1) - M[:, max(b - 14, 0):b].mean(1)
    cod = (days >= f"{year}-01-16") & (days <= f"{year}-02-15")
    out["th_ban_cod"] = safe(rate(cod))
    return pd.DataFrame(out, index=ids)


def build(tag):
    t0 = time.time()
    H = pd.read_parquet(C.FEAT_DIR / f"anom_hourly{tag}.parquet", columns=["rfid_id", "ts", "lon_c", "lat_c", "at_sea", "state"])
    ts = pd.to_datetime(H.ts)
    year = int(ts.dt.year.mode().iat[0])
    H["t"] = (ts.astype("datetime64[ns]").astype("int64") // 3_600_000_000_000).astype(np.int64)
    H["hour"], H["month"], H["doy"] = ts.dt.hour.to_numpy(), ts.dt.month.to_numpy(), (ts.dt.dayofyear - 1).to_numpy()
    F = H[(H.at_sea == 1) & H.state.isin(FISH_STATES)].dropna(subset=["lon_c", "lat_c"])
    parts = []
    for name, fn in [("trip", lambda: trip_feats(H)), ("shift", lambda: shift_feats(F)), ("soak", lambda: soak_feats(F)),
                     ("calendar", lambda: calendar_feats(H, year))]:
        parts.append(fn())
        print(f"  [{tag or 'train'} {year}] {name} {parts[-1].shape} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    B = pd.concat(parts, axis=1)
    B.index.name = "rfid_id"
    B = B.reset_index()
    B.to_parquet(C.FEAT_DIR / f"theory_feats{tag}.parquet", index=False)
    print(f"✅ theory_feats{tag}.parquet {B.shape}", flush=True)


def attach_theory(D, tag):
    B = pd.read_parquet(C.FEAT_DIR / f"theory_feats{tag}.parquet")
    B["rfid_id"] = B["rfid_id"].astype(str)
    m = D[["rfid_id"]].astype(str).merge(B, on="rfid_id", how="left").drop(columns="rfid_id")
    for c in m:
        D[c] = m[c].to_numpy()


def columns_for(cfg, S):
    if cfg == "all":
        prefixes = tuple(GROUPS.values())
    elif cfg in GROUPS:
        prefixes = (GROUPS[cfg],)
    else:
        return []
    return [c for c in S.columns if c.startswith(prefixes)]


def screen():
    t0 = time.time()
    S, y = V.load_ship()
    pool_cols = V.feature_pool(S, "B")
    attach(S, "")
    attach_theory(S, "")
    add = [c for c in S.columns if c.startswith(("rv_", "pd_"))]
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    stamp = time.strftime("%m%d_%H%M")
    rows, base_acc, oofs = [], None, {}
    for cfg in CONFIGS:
        E1.PRIOR_SETS.clear()
        E1.PRIOR_SETS.update(BASE_PRIORS | (SEASON_PRIORS if cfg in ("season_prior", "all") else {}))
        cols = pool_cols + add + columns_for(cfg, S)
        oof = np.zeros((len(y), len(C.CLASSES)))
        acc = []
        for va in folds:
            tr = np.setdiff1d(np.arange(len(y)), va)
            Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
            Xtr = pd.concat([Str[cols], E1.region_prior(Str, y[tr], Str, self_in_ref=True)], axis=1)
            Xva = pd.concat([Sva[cols], E1.region_prior(Str, y[tr], Sva)], axis=1)
            oof[va] = TC.m_lgbm_base(Xtr, y[tr], Xva)
            acc.append(accuracy_score(y[va], oof[va].argmax(1)))
        acc = np.array(acc)
        oofs[cfg] = oof
        if cfg == "base":
            base_acc = acc
        pred = oof.argmax(1)
        pc = f1_score(y, pred, average=None, labels=range(len(C.CLASSES)))
        rows.append(dict(config=cfg, n_new=len(cols) - len(pool_cols) - len(add), cv=accuracy_score(y, pred), macro=V.macro_f1(y, pred),
                         d_cv=accuracy_score(y, pred) - accuracy_score(y, oofs["base"].argmax(1)),
                         folds_up=int((acc > base_acc).sum()), **{f"f1_{cl}": v for cl, v in zip(C.CLASSES, pc)}))
        print(pd.DataFrame(rows).round(4).to_string(index=False), f"\n({(time.time() - t0) / 60:.1f}분)", flush=True)
        pd.DataFrame(rows).round(4).to_csv(C.RESULT_DIR / f"step12_screen_{stamp}.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--screen", action="store_true")
    a = ap.parse_args()
    if a.build:
        build("")
        build("_test")
    if a.screen:
        screen()
