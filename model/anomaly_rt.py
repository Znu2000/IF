"""
[ANOM-7] 실시간 이상 점수 + 업종별 경보 ─ 시각 t 점수는 t 까지의 기록만으로 계산 (현장 배치 가능한 형태)
  기존(anomaly_model.py)에서 미래 정보를 쓰던 부분 교체
    gap_next_h(다음 기록까지 공백)  → 제거. 대신 바다에서 오래 끊겼다 다시 잡힌 시각(gap_prev_h)을 사용
    forward_max([t, t+k] 최대)     → 제거
    own_dev 선박 전체 기간 중앙값   → 그 배의 "직전까지" 지수가중 평균·절대편차 (EWM, 과거 기록 shift)
                                      기록이 적은 초반(MIN_HIST 미만)은 같은 업종 학습 선박 전체 분포로 대체
    백분위(pct) 전체 데이터 기준    → 학습 데이터로 고정한 분위 표 (테스트·실시간에 그대로 적용)
    GRU                           → anomaly_gru.py --rt 결과 (gap_next_h 제외 학습)
  업종 연계 : 업종별 Isolation Forest + 업종별 경보 임계값 (학습 선박의 그 업종 점수 상위 q 분위)
              테스트 선박 업종 = 과제1 예측 (results/test_pred_all.csv)
  평가 (라벨 v1 / 원본 점 재매칭 v2 각각, anomaly_eval 과 같은 지표 + 경보 지표)
    AUROC / AUPRC / top-1 적중 : 기존과 비교
    사고 탐지율   : 사고 [-3h, +1h] 안에 경보가 1번이라도 울린 비율
    선행 시간     : 사고 시각 - 그 구간 첫 경보 시각 (양수 = 사고 전에 울림)
    오경보        : 사고 구간 밖 바다 위 100 선박-시간당 경보 수
  출력 : results/anom_rt_eval.csv, results/anom_rt_alerts.csv(업종별 임계값),
         results/anom_scores_test_rt.parquet, results/submission_rt.xlsx (테스트B 를 실시간 점수로 채운 사본)
실행: python anomaly_rt.py
"""
import json
import time

import numpy as np
import openpyxl
import pandas as pd
from sklearn.ensemble import IsolationForest

import fe_config as C
from anomaly_eval import labels, metrics
from anomaly_model import FEATS, OWN, components as components_old, fit_if as fit_if_old

RT_FEATS = [f for f in FEATS if f != "gap_next_h"]
HALF_LIFE, MIN_HIST = 72, 24          # 지수가중 반감기(기록 수), 개인 기준선 최소 기록 수
W = {"if_type": 2, "own_dev": 1, "gru": 2}   # 기존 제출 조합 그대로 (새로 고르지 않음 → 선택 편향 없음, 규칙 점수는 기존에도 0)
QS = [0.99, 0.995, 0.999]             # 경보 임계 분위 (학습 선박 시간의 상위 1% / 0.5% / 0.1%)
PRE_H, POST_H = 3, 1                  # 사고 탐지 인정 구간 [사고-3h, 사고+1h]


def prep_rt(H):
    X = H[RT_FEATS].astype(np.float32).copy()
    X["gap_prev_h"] = np.log1p(X["gap_prev_h"])
    X["hour_sin"], X["hour_cos"] = np.sin(X["hour"] * np.pi / 12), np.cos(X["hour"] * np.pi / 12)
    return X.drop(columns="hour").fillna(0)


class RefPct:
    """학습 점수 분포로 고정한 백분위 변환 (실시간에 새 점수 1개도 바로 변환 가능)"""

    def __init__(self, ref):
        self.q = np.quantile(np.asarray(ref, float)[~np.isnan(ref)], np.linspace(0, 1, 1001))

    def __call__(self, x):
        x = np.asarray(x, float)
        out = np.searchsorted(self.q, x, side="right") / len(self.q)
        return np.where(np.isnan(x), 0.5, out)


def fit_if(Htr):
    models = {}
    for t in C.CLASSES:
        X = prep_rt(Htr[Htr["fishing_code"] == t])
        X = X.sample(min(len(X), 300_000), random_state=C.RANDOM_STATE)
        models[t] = IsolationForest(n_estimators=200, max_samples=4096, random_state=C.RANDOM_STATE,
                                    n_jobs=C.N_JOBS).fit(X)
    return models


def score_if(H, models, typ):
    s = np.zeros(len(H), np.float32)
    for t, m in models.items():
        idx = np.flatnonzero(typ == t)
        if len(idx):
            s[idx] = -m.score_samples(prep_rt(H.iloc[idx]))
    return s


def type_baseline(Htr):
    """업종별 학습 선박 전체 중앙값·MAD (개인 기록이 적은 초반용 기준선)"""
    g = Htr.groupby("fishing_code", observed=True)[OWN]
    med = g.median()
    mad = (Htr[OWN] - med.reindex(Htr["fishing_code"]).to_numpy()).abs().groupby(Htr["fishing_code"],
                                                                                  observed=True).median()
    return med, mad, Htr[OWN].std()


def own_dev_rt(H, typ, base):
    """그 배의 직전까지 기록으로 만든 기준선 대비 이탈 (현재 기록은 기준선에 안 들어감) ─ 큰 3개 |z| 평균"""
    z = np.abs(own_z_rt(H, typ, base))
    return np.sort(z, axis=1)[:, -3:].mean(axis=1)


def own_z_rt(H, typ, base):
    """피처별 부호 있는 z (OWN 순서, ±20 클리핑) ─ own_dev_rt 와 변화점 탐지(anomaly_combo)가 같이 씀"""
    med, mad, sd = base
    X = H[OWN].astype(float)
    rid = H["rfid_id"].to_numpy()
    prev = X.groupby(rid).shift()                                   # t-1 까지만
    mu = prev.groupby(rid).transform(lambda s: s.ewm(halflife=HALF_LIFE, ignore_na=True).mean())
    dev_prev = (prev - mu.groupby(rid).shift()).abs()
    sc = dev_prev.groupby(rid).transform(lambda s: s.ewm(halflife=HALF_LIFE, ignore_na=True).mean()) * 1.2533
    n_hist = H.groupby("rfid_id").cumcount().to_numpy()
    cold = n_hist < MIN_HIST                                         # 기록이 적으면 업종 기준선
    tmed, tmad = med.reindex(typ).to_numpy(), mad.reindex(typ).to_numpy() * 1.4826
    mu = np.where(cold[:, None], tmed, mu.to_numpy())
    sc = np.where(cold[:, None], tmad, sc.to_numpy())
    sc = np.maximum(np.nan_to_num(sc, nan=0.0), 0.1 * sd.to_numpy())
    z = np.clip((X.to_numpy() - mu) / sc, -20, 20)
    return np.nan_to_num(z, nan=0.0)


def raw_components(H, models, typ, base):
    D = pd.DataFrame(index=H.index)
    D["if_type"] = score_if(H, models, typ)
    D["own_dev"] = own_dev_rt(H, typ, base)
    D["gru"] = H["gru_rt"].to_numpy() if "gru_rt" in H else np.nan
    return D


def fuse(D, ref):
    w = {c: v for c, v in W.items() if c in ref}
    s = sum(ref[c](D[c].to_numpy()) * v for c, v in w.items()) / sum(w.values())
    return s, w


def alert_eval(H, s, thr_by_type, typ, L):
    """사고 탐지율·선행 시간·오경보율"""
    alert = s >= pd.Series(typ).map(thr_by_type).to_numpy()
    rid, ts = H["rfid_id"].to_numpy(), H["ts"]
    in_win = np.zeros(len(H), bool)
    hits, leads = 0, []
    for _, l in L.iterrows():
        m = (rid == l["rfid_id"]) & (ts >= l["label_ts"] - pd.Timedelta(hours=PRE_H)).to_numpy() & \
            (ts <= l["label_ts"] + pd.Timedelta(hours=POST_H)).to_numpy()
        in_win |= m
        a = m & alert
        if a.any():
            hits += 1
            leads.append((l["label_ts"] - ts[a].min()).total_seconds() / 3600)
    sea = (H["at_sea"].to_numpy() == 1) & ~in_win
    rate = alert[~in_win].mean()
    return dict(detect=hits / len(L), lead_med_h=np.median(leads) if leads else np.nan,
                false_per100_sea_h=100 * alert[sea].mean(), alert_rate=rate,
                detect_random=1 - (1 - rate) ** (PRE_H + POST_H + 1))


def load_labels(name):
    p = C.RESULT_DIR / name
    if not p.exists():
        return None
    L = pd.read_csv(p, parse_dates=["label_ts"])
    return L[L["tier"].isin(["A", "B"]) & L["label_ts"].notna()]


def main(a):
    t0 = time.time()
    Htr = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet")
    Hte = pd.read_parquet(C.FEAT_DIR / "anom_hourly_test.parquet")
    for H, tag in [(Htr, ""), (Hte, "_test")]:
        for col, f in [("gru_rt", f"anom_gru_rt{tag}.parquet"), ("gru_err", f"anom_gru{tag}.parquet")]:
            if (C.FEAT_DIR / f).exists():
                H[col] = pd.read_parquet(C.FEAT_DIR / f)["gru_err"].to_numpy()
    if a.smoke:  # 동작 확인용: 라벨 선박 + 무작위 1,500척 / 테스트 300척
        keep = set(pd.read_csv(C.RESULT_DIR / "anom_labels.csv")["rfid_id"].dropna().astype("int64"))
        keep |= set(np.random.default_rng(0).choice(Htr["rfid_id"].unique(), 1500, replace=False))
        Htr = Htr[Htr["rfid_id"].isin(keep)].reset_index(drop=True)
        Hte = Hte[Hte["rfid_id"].isin(Hte["rfid_id"].unique()[:300])].reset_index(drop=True)
    print(f"GRU 실시간 오차: {'gru_rt' in Htr} / 기존 GRU: {'gru_err' in Htr}", flush=True)

    models, base = fit_if(Htr), type_baseline(Htr)
    print(f"업종별 IF 학습 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    # 기준 분포·업종별 경보 임계값: 라벨 선박을 뺀 학습 선박으로 고정
    labs = {n: load_labels(n) for n in ["anom_labels.csv", "anom_labels_raw.csv"]}
    lab_ids = set().union(*[set(L["rfid_id"]) for L in labs.values() if L is not None])
    rng = np.random.default_rng(C.RANDOM_STATE)
    pool = np.setdiff1d(Htr["rfid_id"].unique(), list(lab_ids))
    ref_ids = rng.choice(pool, min(3000, len(pool)), replace=False)
    Hr = Htr[Htr["rfid_id"].isin(ref_ids)].reset_index(drop=True)
    Dr = raw_components(Hr, models, Hr["fishing_code"].to_numpy(), base)
    ref = {c: RefPct(Dr[c].to_numpy()) for c in W if not Dr[c].isna().all()}
    s_ref, w = fuse(Dr, ref)
    thr = {q: Hr.assign(s=s_ref).groupby("fishing_code", observed=True)["s"].quantile(q).to_dict() for q in QS}
    pd.DataFrame(thr).rename_axis("업종").round(4).to_csv(C.RESULT_DIR / "anom_rt_alerts.csv", encoding="utf-8-sig")
    print(f"기준 선박 {len(ref_ids)}척으로 분위표·임계값 고정, 가중 {w} ({(time.time() - t0) / 60:.1f}분)", flush=True)

    # 평가
    rows = []
    old_models = fit_if_old(Htr, list(C.CLASSES)) if "gru_err" in Htr else None
    for name, L in labs.items():
        if L is None:
            continue
        Hv = Htr[Htr["rfid_id"].isin(L["rfid_id"].unique())].reset_index(drop=True)
        typ = Hv["fishing_code"].to_numpy()
        y, y3 = labels(Hv, L, 1), labels(Hv, L, 3)
        Dv = raw_components(Hv, models, typ, base)
        s, _ = fuse(Dv, ref)
        tag = dict(labels=name, n_acc=len(L), n_vessel=Hv["rfid_id"].nunique())
        rng2 = np.random.default_rng(0)
        rows.append(tag | dict(method="무작위") | metrics(Hv, rng2.random(len(Hv)), y, y3))
        if old_models is not None:  # 기존(미래 정보 포함) 제출 조합과 같은 라벨로 비교
            Do = components_old(Hv, old_models, typ)
            so = (Do[["if_type", "own_dev", "gru"]].to_numpy() * [2, 1, 2]).sum(1) / 5
            rows.append(tag | dict(method="기존 IF2:z1:GRU2 (미래 정보 포함)") | metrics(Hv, so, y, y3))
        for c in ref:
            rows.append(tag | dict(method=f"실시간 {c}") | metrics(Hv, ref[c](Dv[c].to_numpy()), y, y3))
        rows.append(tag | dict(method=f"실시간 융합 {w}") | metrics(Hv, s, y, y3))
        for q in QS:
            rows.append(tag | dict(method=f"실시간 경보 q={q}") | alert_eval(Hv, s, thr[q], typ, L))
        print(f"[{name}] 평가 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    E = pd.DataFrame(rows)
    E.round(4).to_csv(C.RESULT_DIR / "anom_rt_eval.csv", index=False, encoding="utf-8-sig")
    cols = [c for c in ["labels", "method", "auroc", "auroc_vessel_mean", "auprc", "base_rate", "hit1_3h",
                        "detect", "detect_random", "lead_med_h", "false_per100_sea_h"] if c in E]
    print(E[cols].round(3).to_string(index=False))

    # 테스트: 업종 = 과제1 예측
    pred = pd.read_csv(C.RESULT_DIR / "test_pred_all.csv", dtype={"rfid_id": str})
    typ = Hte["rfid_id"].astype(str).map(pred.set_index("rfid_id")["pred"]).fillna("연안복합").to_numpy()
    Dt = raw_components(Hte, models, typ, base)
    st, _ = fuse(Dt, ref)
    Hte["score"] = st.astype(np.float32)
    Hte["fishing_pred"] = typ
    Hte["alert_q995"] = st >= pd.Series(typ).map(thr[0.995]).to_numpy()
    Hte[["rfid_id", "ts", "fishing_pred", "score", "alert_q995"]].to_parquet(
        C.RESULT_DIR / "anom_scores_test_rt.parquet", index=False)
    if not a.smoke:
        write_submission_rt(Hte)
    (C.RESULT_DIR / "anom_rt_config.json").write_text(json.dumps(dict(
        weights=w, half_life=HALF_LIFE, min_hist=MIN_HIST, thresholds={str(q): thr[q] for q in QS}),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분 → anom_rt_eval.csv, anom_scores_test_rt.parquet, submission_rt.xlsx")


def write_submission_rt(Hte):
    """submission.xlsx 를 복사해 테스트B 만 실시간 점수로 채움 (사고시각 = 1위 시각, 시각1~3 = 직전 24h 안 기록)"""
    import itertools
    wb = openpyxl.load_workbook(C.RESULT_DIR / "submission.xlsx")
    ws = wb.worksheets[1]
    G = {k: v.set_index("ts")["score"] for k, v in Hte.assign(rid=Hte["rfid_id"].astype(str)).groupby("rid")}
    fmt = lambda t: t.strftime("%Y%m%d %H")
    n = 0
    for row in ws.iter_rows(min_row=2):
        rid = str(row[0].value).strip()
        if rid not in G:
            continue
        s = G[rid]
        t0 = s.idxmax()
        vals = [fmt(t0), round(float(s.max()), 4)]
        prev = s[(s.index < t0) & (s.index >= t0 - pd.Timedelta(hours=24))].sort_index().tail(3)
        for t, v in prev.items():
            vals += [fmt(t), round(float(v), 4)]
        for cell, v in itertools.zip_longest(row[1:9], vals):
            cell.value = v
        n += 1
    wb.save(C.RESULT_DIR / "submission_rt.xlsx")
    print(f"테스트B {n}척 → submission_rt.xlsx (기존 submission.xlsx 는 그대로)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="일부 선박만으로 동작 확인")
    main(ap.parse_args())
