"""
[STEP9-3] 실제 이상 판별 · 사고 시각 적중 ─ 리더보드가 아니라 탐지 능력 자체를 정직하게 재고 개선
  지표
    선박 판별   : 사고 선박 vs 사고 없는 선박. 바쁜 배일수록 사고·이벤트가 많아 생기는 착시를 빼려고
                  바다 시간 수 10분위 안에서만 비교한 AUC 의 평균(활동량 보정 AUC)을 같이 봄
    시각 적중   : 그 배의 바다 시간들 중 점수 1위 시각이 사고 ±1·3·6h 안인 비율, 사고 시각의 순위 백분위(1 = 맨 위)
  탐지기 (시간대 점수, 앞뒤 기록 사용 = 사후 판정)
    onset       : 바다에서 이동·조업하다 급정지·표류 시작 (step9_testb_sim 과 같음, 비교 기준)
    unusual     : 급정지 사건마다 특징을 뽑고 "그 배의 다른 급정지들" 대비 얼마나 튀는지(robust z)로 점수
                  특징: 이후 정지·표류 시간, 평소 안 가는 칸, 이후 12h 안 귀항, 이후 저속 직진(예인), 출항 후 경과 시간이 짧음,
                        기록 끊김, 직전 속력
    ranker      : 같은 특징으로 로지스틱 순위 모델. 라벨 선박 하나씩 빼고 학습(선박 단위 leave-one-out)
                  양성 = 사고 ±3h 안 급정지, 음성 = 같은 배의 나머지 급정지
  라벨: results/anom_labels_v2.csv 확실 등급(AS·A·S). 바다 사고 / 항구 사고 따로 보고
  출력: results/step9_detect_eval.csv
실행: python step9_detect.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import fe_config as C
from step9_testb_sim import bwd_any, fwd_any

SURE = ["AS", "A", "S"]
COLS = ["rfid_id", "ts", "at_sea", "state", "ks_speed_med", "ks_heading_R", "cell_rarity", "gap_next_h", "home_km"]
FEATS = ["f_drift", "f_rare", "f_ret", "f_tow", "f_trip_short", "f_gap", "f_pre_speed"]


def build(H):
    v = H["rid"].to_numpy()
    ts = H["ts"].to_numpy()
    sea = H["at_sea"].fillna(0).to_numpy() == 1
    st = H["state"].to_numpy()
    sp = H["ks_speed_med"].fillna(0).to_numpy()
    hr = H["ks_heading_R"].fillna(0).to_numpy()
    stop = sea & np.isin(st, (1, 2))
    move = sea & np.isin(st, (3, 4))
    prev_stop = np.r_[False, (v[1:] == v[:-1]) & stop[:-1]]
    onset = stop & ~prev_stop & bwd_any(v, move, ts, 4)
    # 출항 후 경과 시간: 마지막 항구 기록 이후 시간
    port = st == 0
    last_port = pd.Series(np.where(port, ts, np.datetime64("NaT")), dtype="datetime64[ns]").groupby(v).ffill().to_numpy()
    since = (ts - last_port) / np.timedelta64(1, "h")
    prev_sp = np.r_[0.0, np.where(v[1:] == v[:-1], sp[:-1], 0.0)]
    F = pd.DataFrame({
        "rid": v, "ts": ts, "sea": sea, "onset": onset,
        "f_drift": np.minimum(fwd_any(v, stop, ts, 12), 12).astype(float),
        "f_rare": H["cell_rarity"].fillna(0).to_numpy(),
        "f_ret": (fwd_any(v, port, ts, 12) > 0).astype(float),
        "f_tow": (fwd_any(v, sea & (sp >= 2) & (sp < 6) & (hr >= 0.8), ts, 12) > 0).astype(float),
        "f_trip_short": -np.nan_to_num(since, nan=48.0).clip(0, 48),   # 출항 직후일수록 큼
        "f_gap": (H["gap_next_h"].fillna(1).to_numpy() >= 3).astype(float),
        "f_pre_speed": prev_sp,
    })
    return F


def robust_z(E):
    g = E.groupby("rid")[FEATS]
    med = g.transform("median")
    iqr = (g.transform(lambda s: s.quantile(0.75)) - g.transform(lambda s: s.quantile(0.25))).replace(0, np.nan)
    sd = g.transform("std").replace(0, np.nan)
    scale = iqr.fillna(sd).fillna(1.0)
    return ((E[FEATS] - med) / scale).clip(-5, 5).fillna(0)


def strat_auc(y, s, act, nbin=10):
    q = pd.qcut(act, nbin, labels=False, duplicates="drop")
    aucs, ws = [], []
    for b in np.unique(q):
        m = q == b
        if y[m].sum() >= 1 and (1 - y[m]).sum() >= 1:
            aucs.append(roc_auc_score(y[m], s[m]))
            ws.append(y[m].sum())
    return float(np.average(aucs, weights=ws)) if aucs else np.nan


def time_metrics(F, score, pos):
    """pos: rid, label_ts, group. 그 배 바다 시간 중 점수 순위"""
    out = []
    G = dict(tuple(F.assign(s=score)[["rid", "ts", "sea", "s"]].groupby("rid")))
    for _, p in pos.iterrows():
        g = G.get(p["rid"])
        g = g[g["sea"] | (g["s"] > 0)]
        if g is None or g.empty or g["s"].max() <= 0:
            out.append(dict(group=p["group"], h1=0, h3=0, h6=0, pct=np.nan))
            continue
        err = np.abs((g["ts"] - p["label_ts"]) / pd.Timedelta(hours=1)).to_numpy()
        top = g["s"].to_numpy().argmax()
        near = err <= 1
        pct = (g["s"].to_numpy() < g["s"].to_numpy()[near].max()).mean() if near.any() else np.nan
        out.append(dict(group=p["group"], h1=float(err[top] <= 1), h3=float(err[top] <= 3), h6=float(err[top] <= 6), pct=pct))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=COLS)
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.sort_values(["rid", "ts"], kind="stable").reset_index(drop=True)
    F = build(H)
    E = F[F["onset"]].copy()
    E[[f"z_{c}" for c in FEATS]] = robust_z(E).to_numpy()
    print(f"급정지 사건 {len(E):,}건 / 선박 {E['rid'].nunique():,}척 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    pos = L[L["tier"].isin(SURE)].drop_duplicates("rid")[["rid", "label_ts", "group"]].reset_index(drop=True)

    # 사건 점수 3종
    E["s_onset"] = 1 + E["f_drift"] / 6
    zpos = ["z_f_drift", "z_f_rare", "z_f_ret", "z_f_tow", "z_f_trip_short", "z_f_gap"]
    E["s_unusual"] = E[zpos].clip(lower=0).sum(1) + 1e-3
    E["y"] = 0
    lab = pos.set_index("rid")["label_ts"]
    m = E["rid"].isin(lab.index)
    E.loc[m, "y"] = (np.abs((E.loc[m, "ts"] - E.loc[m, "rid"].map(lab)) / pd.Timedelta(hours=1)) <= 3).astype(int)
    X = E[[f"z_{c}" for c in FEATS] + FEATS].to_numpy()
    E["s_ranker"] = 0.0
    lab_ids = [r for r in pos["rid"] if E.loc[E["rid"] == r, "y"].any()]
    rng = np.random.default_rng(0)
    neg_pool = E.index[~E["rid"].isin(touched)].to_numpy()
    neg_bg = rng.choice(neg_pool, min(20000, len(neg_pool)), replace=False)  # 정상 선박 사건 일부를 음성으로 보탬
    for r in pos["rid"]:
        te = E.index[E["rid"] == r]
        if len(te) == 0:  # 급정지 사건이 없는 사고 선박
            continue
        tr = E.index[E["rid"].isin([x for x in lab_ids if x != r])].to_numpy()
        tr = np.r_[tr, neg_bg]
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(X[E.index.get_indexer(tr)], E.loc[tr, "y"])
        E.loc[te, "s_ranker"] = clf.predict_proba(X[E.index.get_indexer(te)])[:, 1]
    clf_all = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(
        X[E.index.get_indexer(np.r_[E.index[E["rid"].isin(lab_ids)], neg_bg])],
        E.loc[np.r_[E.index[E["rid"].isin(lab_ids)], neg_bg], "y"])
    rest = ~E["rid"].isin(pos["rid"])
    E.loc[rest, "s_ranker"] = clf_all.predict_proba(X[rest.to_numpy()])[:, 1]
    print("순위 모델 계수:", dict(zip([f"z_{c}" for c in FEATS] + FEATS, np.round(clf_all.coef_[0], 2))), flush=True)

    # 시간대 점수로 펼치기 (사건이 없는 시간 = 0)
    act = F.groupby("rid")["sea"].sum()
    neg_ids = np.array(sorted(set(F["rid"]) - touched))
    vid = np.r_[pos["rid"].to_numpy(), neg_ids]
    yv = np.r_[np.ones(len(pos)), np.zeros(len(neg_ids))]
    rows = []
    for name in ["s_onset", "s_unusual", "s_ranker"]:
        sc = np.zeros(len(F))
        sc[E.index] = E[name].to_numpy()
        T = time_metrics(F, sc, pos)
        vmax = pd.Series(sc).groupby(F["rid"].to_numpy()).max()
        vs = vmax.reindex(vid).fillna(0).to_numpy()
        a = act.reindex(vid).fillna(0).to_numpy()
        for grp, Tg in [("all", T), ("sea", T[T["group"] != "port"]), ("port", T[T["group"] == "port"])]:
            rows.append(dict(detector=name, labels=grp, n=len(Tg), hit1h=Tg["h1"].mean(), hit3h=Tg["h3"].mean(),
                             hit6h=Tg["h6"].mean(), rank_pct_median=Tg["pct"].median(),
                             vessel_auc=roc_auc_score(yv, vs) if grp == "all" else np.nan,
                             vessel_auc_activity_matched=strat_auc(yv, vs, a) if grp == "all" else np.nan))
    # 기준: 활동량만으로 판별
    a = act.reindex(vid).fillna(0).to_numpy()
    rows.append(dict(detector="바다 시간 수(활동량)만", labels="all", vessel_auc=roc_auc_score(yv, a),
                     vessel_auc_activity_matched=strat_auc(yv, a + rng.random(len(a)) * 1e-6, a)))
    R = pd.DataFrame(rows)
    R.round(4).to_csv(C.RESULT_DIR / "step9_detect_eval.csv", index=False, encoding="utf-8-sig")
    print(R.round(3).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
