"""
[STEP9-4] 원본 1분 점으로 급정지 순간 보기 ─ 1시간 집계에서 흐려진 "달리다 갑자기 멈춤"을 분 단위로 잡을 수 있나
  대상 : 라벨 v2 확실 등급 사고 선박 38척 + 바다 시간 수 10분위를 맞춘 정상 선박 (사고 선박 1척당 10척), 6개월 전체
  추출 : 원본 CSV(2026 학습) 를 월별 병렬로 훑어 대상 선박 행만 파싱. 점별 시각이 없어 1시간 안 균등 간격 가정
         튐 제거(fe_hourly.remove_spikes) 후 3점 간격 속력(노트)
  분 단위 특징 (시간마다)
    m_decel     : 직전 5분 평균 속력 − 직후 10분 평균 속력 의 최댓값 (직후 평균 < 0.5노트일 때만) = 급정지 세기
    m_decel_min : 그 급정지가 일어난 분 (시각 보정용)
    m_stop_min  : 0.5노트 미만 분 수,  m_vmax : 최고 속력,  m_switch : 정지↔이동 전환 횟수
  점수 : minute       = m_decel
         minute_drift = m_decel × (1 + 이후 12시간 정지·표류 시간 / 6)   (시간 단위 이후 흐름 결합)
  평가 : step9_detect 와 같은 지표 (±1·3·6h 적중, 순위 백분위, 선박 판별 AUC, 활동량 보정 AUC)
         급정지 분으로 시각을 보정한 경우도 따로 봄
  출력 : features/minute_feats.parquet (실제 rfid_id → 깃헙 금지), results/step9_minute_eval.csv
실행: python step9_minute.py --extract   (원본 스캔, 시간 걸림)
      python step9_minute.py             (평가)
"""
import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fe_config as C
from fe_hourly import parse_parts, remove_spikes
from step9_detect import strat_auc
from step9_testb_sim import fwd_any

SURE = ["AS", "A", "S"]
SRC = C.PROJECT_ROOT / "문제 데이터_수정_1103" / "vpass_26년(1월_6월) 데이터"
R_KM = 6371.0


def targets():
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    pos = L[L["tier"].isin(SURE)].drop_duplicates("rid")[["rid", "label_ts", "group"]].reset_index(drop=True)
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "at_sea"])
    act = H.assign(rid=H["rfid_id"].astype(str)).groupby("rid")["at_sea"].sum()
    q = pd.qcut(act, 10, labels=False, duplicates="drop")
    rng = np.random.default_rng(0)
    neg = []
    for b in q.reindex(pos["rid"]).dropna().astype(int):
        pool = [r for r in q.index[q == b] if r not in touched and r not in neg]
        neg += list(rng.choice(pool, min(10, len(pool)), replace=False))
    return pos, sorted(set(neg)), touched


def hour_feats(geom):
    parts = parse_parts(geom)
    if not parts:
        return None
    p = np.vstack(parts)
    if len(p) < 8:
        return None
    lat0 = np.radians(p[:, 1].mean())
    q = np.c_[np.radians(p[:, 0]) * np.cos(lat0), np.radians(p[:, 1])] * R_KM * 1000   # 미터
    q, _ = remove_spikes(q)
    n = len(q)
    if n < 8:
        return None
    x, y = q[:, 0], q[:, 1]
    k = 3
    dt_min = 60.0 / n
    d = np.hypot(x[k:] - x[:-k], y[k:] - y[:-k])
    v = d / (k * dt_min * 60) * 1.94384                       # 노트
    minute = (np.arange(len(v)) + k / 2) * dt_min
    w5, w10 = max(1, int(round(5 / dt_min))), max(1, int(round(10 / dt_min)))
    ok = v <= 25                                               # 어선이 낼 수 없는 속력(점프)은 제외
    if ok.sum() < 8:
        return None
    best, best_m = 0.0, np.nan
    cs = np.r_[0, np.cumsum(np.where(ok, v, 0))]
    cn = np.r_[0, np.cumsum(ok)]
    for i in range(w5, len(v) - w10 + 1):
        n_pre, n_post = cn[i] - cn[i - w5], cn[i + w10] - cn[i]
        if n_pre < w5 / 2 or n_post < w10 / 2:
            continue
        pre = (cs[i] - cs[i - w5]) / n_pre
        post = (cs[i + w10] - cs[i]) / n_post
        if post < 0.5 and pre - post > best:
            best, best_m = pre - post, minute[i]
    vv = v[ok]
    stop = vv < 0.5
    return dict(m_decel=best, m_decel_min=best_m, m_stop_min=float(stop.mean() * 60), m_vmax=float(vv.max()),
                m_switch=int((np.diff(stop.astype(int)) != 0).sum()))


def scan(path, ids):
    t0 = time.time()
    ids = set(ids)
    out = []
    for raw in pd.read_csv(path, chunksize=200_000, dtype=str, encoding="utf-8-sig",
                           usecols=["part_dt", "rfid_id", "rfid_time", "geom"]):
        raw = raw[raw["rfid_id"].isin(ids)]
        for r in raw.itertuples(index=False):
            f = hour_feats(r.geom)
            if f is not None:
                out.append(dict(rid=r.rfid_id, part_dt=r.part_dt, hour=r.rfid_time) | f)
    print(f"  {path.name}: {len(out):,}시간 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    return out


def extract():
    pos, neg, _ = targets()
    ids = list(pos["rid"]) + neg
    files = sorted(SRC.glob("**/vpass_*.csv"))
    print(f"대상 {len(ids)}척 (사고 {len(pos)}, 정상 {len(neg)}) / 파일 {len(files)}개", flush=True)
    with ProcessPoolExecutor(min(6, C.N_JOBS)) as ex:
        res = list(ex.map(scan, files, [ids] * len(files)))
    M = pd.DataFrame([r for part in res for r in part])
    M["hour"] = pd.to_numeric(M["hour"], errors="coerce")
    M = M[M["hour"].between(0, 23)]
    M["ts"] = pd.to_datetime(M["part_dt"]) + pd.to_timedelta(M["hour"].astype(int), unit="h")
    M.drop(columns=["part_dt", "hour"]).to_parquet(C.FEAT_DIR / "minute_feats.parquet", index=False)
    print(f"✅ {len(M):,}시간 저장")


def time_eval(G, pos, col, refine):
    out = []
    for _, p in pos.iterrows():
        g = G.get(p["rid"])
        if g is None or g[col].max() <= 0:
            out.append(dict(group=p["group"], h1=0.0, h3=0.0, h6=0.0, pct=np.nan))
            continue
        tt = g["ts"] + (pd.to_timedelta(g["m_decel_min"].fillna(30), unit="m") if refine else pd.Timedelta(0))
        err = np.abs((tt - p["label_ts"]) / pd.Timedelta(hours=1)).to_numpy()
        s = g[col].to_numpy()
        top = s.argmax()
        near = np.abs((g["ts"] - p["label_ts"]) / pd.Timedelta(hours=1)).to_numpy() <= 1
        pct = (s < s[near].max()).mean() if near.any() else np.nan
        out.append(dict(group=p["group"], h1=float(err[top] <= 1), h3=float(err[top] <= 3), h6=float(err[top] <= 6), pct=pct))
    return pd.DataFrame(out)


def evaluate():
    t0 = time.time()
    pos, neg, _ = targets()
    M = pd.read_parquet(C.FEAT_DIR / "minute_feats.parquet")
    M["rid"] = M["rid"].astype(str)
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "ts", "at_sea", "state"])
    H["rid"] = H["rfid_id"].astype(str)
    H = H[H["rid"].isin(set(M["rid"]))]
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.sort_values(["rid", "ts"], kind="stable").reset_index(drop=True)
    sea = H["at_sea"].fillna(0).to_numpy() == 1
    stop = sea & np.isin(H["state"].to_numpy(), (1, 2))
    H["drift12"] = np.minimum(fwd_any(H["rid"].to_numpy(), stop, H["ts"].to_numpy(), 12), 12)
    M = M.merge(H[["rid", "ts", "at_sea", "drift12"]], on=["rid", "ts"], how="left")
    M = M[M["at_sea"].fillna(0) == 1].copy()                     # 바다 시간만 후보
    M["minute"] = M["m_decel"]
    M["minute_drift"] = M["m_decel"] * (1 + M["drift12"].fillna(0) / 6)
    G = {r: g.reset_index(drop=True) for r, g in M.groupby("rid")}
    act = M.groupby("rid").size()
    vid = np.r_[pos["rid"].to_numpy(), np.array(neg)]
    yv = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    rows = []
    for col in ["minute", "minute_drift"]:
        for refine in (False, True):
            T = time_eval(G, pos, col, refine)
            vs = M.groupby("rid")[col].max().reindex(vid).fillna(0).to_numpy()
            a = act.reindex(vid).fillna(0).to_numpy()
            for grp, Tg in [("all", T), ("sea", T[T["group"] != "port"]), ("port", T[T["group"] == "port"])]:
                rows.append(dict(detector=col + (" +분 보정" if refine else ""), labels=grp, n=len(Tg),
                                 hit1h=Tg["h1"].mean(), hit3h=Tg["h3"].mean(), hit6h=Tg["h6"].mean(),
                                 rank_pct_median=Tg["pct"].median(),
                                 vessel_auc=roc_auc_score(yv, vs) if grp == "all" else np.nan,
                                 vessel_auc_activity_matched=strat_auc(yv, vs, a, nbin=5) if grp == "all" else np.nan))
    # 무작위 시각 기준: 그 배 바다 시간 중 아무 시각
    rng = np.random.default_rng(0)
    hits = []
    for _, p in pos.iterrows():
        g = G.get(p["rid"])
        if g is None:
            continue
        err = np.abs((g["ts"] - p["label_ts"]) / pd.Timedelta(hours=1)).to_numpy()
        hits.append([(err[rng.integers(len(err), size=2000)] <= h).mean() for h in (1, 3, 6)])
    h = np.mean(hits, 0)
    rows.append(dict(detector="무작위 바다 시각", labels="all", hit1h=h[0], hit3h=h[1], hit6h=h[2]))
    R = pd.DataFrame(rows)
    R.round(4).to_csv(C.RESULT_DIR / "step9_minute_eval.csv", index=False, encoding="utf-8-sig")
    print(f"바다 시간 {len(M):,}개 / 사고 {len(pos)}척, 정상 {len(neg)}척")
    print(R.round(3).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    extract() if ap.parse_args().extract else evaluate()
