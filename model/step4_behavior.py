"""
[STEP4] 아직 비어 있던 행동 관점 피처 4종 (학습·테스트 같은 방식, 정답 사용 없음)
  revisit : 같은 500m 칸을 다시 찾는 간격 (자망=다음 날, 통발=며칠 뒤, 연안복합=불규칙)
  dbscan  : 배마다 조업 지점을 밀도 군집(DBSCAN, 반경 500m)으로 묶은 어장 수·크기·모양·간격
  period  : 출어 주기 (바다에 나간 날 간격, 연속 출어, 요일·24시간 자기상관)
  congest : 조업할 때 같은 시각 같은 칸(1km, 5km)에 다른 배가 몇 척 있었나 (학습=2026 선단, 테스트=2025 선단 안에서)
  ※ 5번째 '행동 유사 kNN'은 정답 비율이라 fold 별로 step4_eval.py 에서 계산
입력: features/anom_hourly{,_test}.parquet   출력: features/behav_feats{,_test}.parquet
"""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import fe_config as C

FISH_STATES = (1, 2, 3)  # 정지·표류·조업 (바다)
EP_GAP_H = 3             # 같은 칸에서 3시간 넘게 비면 다른 방문


def load(tag):
    H = pd.read_parquet(C.FEAT_DIR / f"anom_hourly{tag}.parquet",
                        columns=["rfid_id", "ts", "lon_c", "lat_c", "at_sea", "state"])
    H["t"] = (pd.to_datetime(H["ts"]).astype("int64") // 3_600_000_000_000).astype(np.int64)
    return H


def revisit(F):
    c = F.assign(cx=np.floor(F.lon_c / 0.005).astype(np.int64), cy=np.floor(F.lat_c / 0.005).astype(np.int64))
    c = c.sort_values(["rfid_id", "cx", "cy", "t"])
    same = (c.rfid_id.eq(c.rfid_id.shift()) & c.cx.eq(c.cx.shift()) & c.cy.eq(c.cy.shift())).to_numpy()
    gap = c.t.diff().to_numpy()
    new_ep = ~same | (gap > EP_GAP_H)
    ep = c[new_ep].copy()                                    # 방문(에피소드) 시작
    ep_same = (ep.rfid_id.eq(ep.rfid_id.shift()) & ep.cx.eq(ep.cx.shift()) & ep.cy.eq(ep.cy.shift())).to_numpy()
    ep["iv"] = np.where(ep_same, ep.t.diff(), np.nan)        # 같은 칸 재방문 간격 (시간)
    iv = ep.dropna(subset=["iv"])
    g = iv.groupby("rfid_id")["iv"]
    out = pd.DataFrame({
        "rv_n_per_ep": iv.groupby("rfid_id").size() / ep.groupby("rfid_id").size(),
        "rv_iv_med_h": g.median(), "rv_iv_q25_h": g.quantile(0.25), "rv_iv_q75_h": g.quantile(0.75),
        "rv_iv_log_std": np.log1p(iv.iv).groupby(iv.rfid_id).std(),
    })
    for name, lo, hi in [("lt12", 0, 12), ("d1", 12, 36), ("d2_3", 36, 84), ("d4_7", 84, 180), ("gt7d", 180, 1e9)]:
        out[f"rv_iv_{name}"] = iv.assign(b=iv.iv.between(lo, hi, inclusive="left")).groupby("rfid_id")["b"].mean()
    cells = ep.groupby(["rfid_id", "cx", "cy"]).size()
    out["rv_cell_revisit_share"] = (cells > 1).groupby("rfid_id").mean()
    out["rv_ep_per_cell"] = cells.groupby("rfid_id").mean()
    return out


def _pca_elong(xy):
    if len(xy) < 3:
        return np.nan
    ev = np.linalg.eigvalsh(np.cov(xy.T))
    return float(np.sqrt(ev[-1] / max(ev[0], 1e-9)))


def dbscan(F):
    rows = {}
    for v, h in F.groupby("rfid_id"):
        lat0 = np.radians(h.lat_c.mean())
        xy = np.c_[h.lon_c.to_numpy() * 111.32 * np.cos(lat0), h.lat_c.to_numpy() * 110.57]  # km
        days = (h.t.to_numpy() // 24)
        r = dict(db_n_pts=len(xy))
        if len(xy) >= 5:
            lab = DBSCAN(eps=0.5, min_samples=5).fit_predict(xy)
            ks = [k for k in np.unique(lab) if k >= 0]
            r["db_noise"] = (lab < 0).mean()
            r["db_n_cl"] = len(ks)
            r["db_n_cl_per100h"] = len(ks) / len(xy) * 100
            if ks:
                sz = np.array([(lab == k).sum() for k in ks])
                cen = np.array([xy[lab == k].mean(0) for k in ks])
                rad = np.array([np.sqrt(((xy[lab == k] - cen[i]) ** 2).sum(1)).mean() for i, k in enumerate(ks)])
                el = np.array([_pca_elong(xy[lab == k]) for k in ks])
                nd = np.array([len(np.unique(days[lab == k])) for k in ks])
                r.update(db_top1=sz.max() / len(xy), db_top3=np.sort(sz)[-3:].sum() / len(xy),
                         db_rad_km=np.average(rad, weights=sz), db_elong=np.nanmedian(el),
                         db_days_per_cl=np.average(nd, weights=sz), db_multiday_cl=(nd >= 3).mean(),
                         db_span_km=np.sqrt(((cen - cen.mean(0)) ** 2).sum(1)).max() if len(ks) > 1 else 0.0)
                if len(ks) > 1:
                    d = np.sqrt(((cen[:, None] - cen[None]) ** 2).sum(2)); np.fill_diagonal(d, np.inf)
                    r["db_nn_km"] = np.median(d.min(1))
                    pe = np.linalg.eigvalsh(np.cov(cen.T)) if len(ks) > 2 else None
                    r["db_cl_line"] = float(np.sqrt(pe[-1] / max(pe[0], 1e-9))) if pe is not None else np.nan
        rows[v] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def _ac(x, lag):
    if len(x) <= lag + 2 or x.std() == 0:
        return np.nan
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def period(H):
    rows = {}
    t0 = H.t.min()
    span_h = H.t.max() - t0 + 1
    for v, h in H.groupby("rfid_id"):
        sea = np.zeros(span_h, np.float32)
        sea[(h.t[h.at_sea == 1] - t0).to_numpy()] = 1
        obs = np.zeros(span_h, bool); obs[(h.t - t0).to_numpy()] = True
        first, last = np.flatnonzero(obs)[[0, -1]]
        s = sea[first:last + 1]
        d = s[: len(s) // 24 * 24].reshape(-1, 24).max(1)       # 바다에 나간 날
        sd = np.flatnonzero(d)
        r = dict(pd_sea_day_share=d.mean() if len(d) else np.nan,
                 pd_ac24=_ac(s, 24), pd_ac48=_ac(s, 48), pd_ac72=_ac(s, 72), pd_ac168=_ac(s, 168),
                 pd_day_ac1=_ac(d, 1), pd_day_ac7=_ac(d, 7))
        if len(sd) > 1:
            gp = np.diff(sd)
            r.update(pd_gap_med=np.median(gp), pd_gap_cv=gp.std() / gp.mean(), pd_gap1_share=(gp == 1).mean())
            e = np.flatnonzero(np.diff(np.r_[0, d, 0]) != 0)
            runs = e[1::2] - e[0::2]                                 # 연속 출어 일수
            r.update(pd_run_max=np.max(runs), pd_run_med=np.median(runs))
            wd = pd.to_datetime((sd + (t0 + first) // 24) * 86400, unit="s").dayofweek
            cnt = np.bincount(wd, minlength=7) / len(sd)
            r.update(pd_wd_std=cnt.std(), pd_sun_share=cnt[6])
        rows[v] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def congest(F):
    out = {}
    for res, nm in [(0.01, "1km"), (0.05, "5km")]:
        k = (np.floor(F.lon_c / res).astype(np.int64) * 100_000 + np.floor(F.lat_c / res).astype(np.int64)) * 10_000_000 + F.t
        u = pd.DataFrame({"k": k.to_numpy(), "v": F.rfid_id.to_numpy()}).drop_duplicates()
        u["n"] = u.groupby("k")["v"].transform("size") - 1
        g = u.groupby("v")["n"]
        out[f"cg_{nm}_mean"] = np.log1p(u.n).groupby(u.v).mean()
        out[f"cg_{nm}_any"] = (u.n > 0).groupby(u.v).mean()
        out[f"cg_{nm}_q90"] = g.quantile(0.9)
    return pd.DataFrame(out)


def build(tag):
    t0 = time.time()
    H = load(tag)
    F = H[(H.at_sea == 1) & H.state.isin(FISH_STATES)].dropna(subset=["lon_c", "lat_c"])
    parts = []
    for name, fn, src in [("revisit", revisit, F), ("congest", congest, F), ("period", period, H), ("dbscan", dbscan, F)]:
        parts.append(fn(src))
        print(f"  [{tag or 'train'}] {name} {parts[-1].shape} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    B = pd.concat(parts, axis=1)
    B.index.name = "rfid_id"
    B = B.reset_index()
    B.to_parquet(C.FEAT_DIR / f"behav_feats{tag}.parquet", index=False)
    return B


if __name__ == "__main__":
    for tag in ["", "_test"]:
        b = build(tag)
        print(f"✅ behav_feats{tag}.parquet {b.shape}", flush=True)
