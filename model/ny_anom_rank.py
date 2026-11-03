"""
[NY-2] 이상탐지 → '사고 시각 순위 학습' ─ 나연
  문제 : 테스트B는 배마다 사고시각 1개를 고르는 형식인데, 비지도 이상점수(IF·z-score·GRU)는 사고 시각을
         그 배의 1위로 거의 못 올림 (top-1 적중 ≈ 0~4%).
  방법 : 사고 매칭 라벨(anomaly_labels.py, 등급 A/B)을 정답으로, 배 안에서 사고 시각을 위로 올리는 모델 학습
    · 입력 = 기존 비지도 점수(if_type / own_dev / event) + 시간 피처 + 그 배 안 백분위 + 앞뒤 시간 변화(lead/lag)
    · 모델 = LightGBM lambdarank (배 = 그룹) / LightGBM 이진 / 로지스틱 / 기존 가중합
  검증 : 선박 단위 5-fold × 3 반복 (라벨 수가 적어 반복 평균). 지표는 제출 형식에 맞춘
    hit@1 ±1h·±3h (배마다 1위 시각이 사고 근처인가), top5·top24 (상위 k 안에 있는가), 백분위 중앙값
실행: python ny_anom_rank.py           (anomaly_prep.py, anomaly_labels.py 선행)
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fe_aggregate as A
import fe_config as C
from anomaly_model import components, fit_if

RAW = ["sp_max", "sp_p90", "ks_speed_med", "d_speed", "jump_prev_km", "gap_prev_h", "gap_next_h", "home_km",
       "home_km_delta", "port_km", "dist_m", "rg_m", "frac_stop", "turn_mean", "heading_R", "n_points",
       "spike_ratio", "ks_frac_drift", "hod_rarity", "cell_rarity", "at_sea"]
PCT = ["sp_max", "ks_speed_med", "d_speed", "jump_prev_km", "gap_next_h", "gap_prev_h", "home_km",
       "home_km_delta", "dist_m", "rg_m", "turn_mean", "ks_frac_drift"]
LEAD = ["ks_speed_med", "sp_max", "home_km_delta", "at_sea", "gap_next_h", "dist_m"]


def shift_within(H, col, k):
    """같은 배의 k 칸 뒤(k>0) / 앞(k<0) 기록 값 (연속 시간 아니어도 다음 기록)"""
    v = H[col].to_numpy(np.float32)
    rid = H["rfid_id"].to_numpy()
    out = np.full(len(H), np.nan, np.float32)
    if k > 0:
        ok = rid[k:] == rid[:-k]
        out[:-k][ok] = v[k:][ok]
    else:
        k = -k
        ok = rid[k:] == rid[:-k]
        out[k:][ok] = v[:-k][ok]
    return out


def trip_feats(H):
    """출어(trip) 구조: 항구 정박이 아닌 시간대를 3시간 넘는 공백으로 끊어 한 출어로 묶음.
    매칭 사고의 77%가 출어 중이고, 출어 끝까지 남은 시간 중앙값 2시간 (사고 뒤 귀항·예인)"""
    t = H["ts"].to_numpy().astype("datetime64[h]").astype(np.int64)
    rid = H["rfid_id"].to_numpy()
    act = H["state"].to_numpy() != A.PORT
    cols = ["h_to_trip_end", "h_from_trip_start", "trip_len_h", "trip_pos", "is_trip_last", "idle_after_h",
            "idle_before_h", "trip_max_home_km"]
    O = {c: np.full(len(H), np.nan) for c in cols}
    O["in_trip"] = act.astype(float)
    ia = np.flatnonzero(act)
    if len(ia):
        ta, ra = t[ia], rid[ia]
        new = np.r_[True, (ra[1:] != ra[:-1]) | (np.diff(ta) > 3)]
        trip = np.cumsum(new)
        T = pd.DataFrame({"trip": trip, "t": ta, "rid": ra, "hk": H["home_km"].to_numpy()[ia]})
        g = T.groupby("trip")
        s, e = g["t"].transform("min").to_numpy(), g["t"].transform("max").to_numpy()
        tr = g.agg(rid=("rid", "first"), s=("t", "min"), e=("t", "max"))
        nxt = tr.groupby("rid")["s"].shift(-1) - tr["e"]
        prv = tr["s"] - tr.groupby("rid")["e"].shift(1)
        O["h_to_trip_end"][ia] = e - ta
        O["h_from_trip_start"][ia] = ta - s
        O["trip_len_h"][ia] = e - s + 1
        O["trip_pos"][ia] = (ta - s) / np.maximum(e - s, 1)
        O["is_trip_last"][ia] = (ta == e)
        O["idle_after_h"][ia] = np.log1p(nxt.reindex(trip).to_numpy())
        O["idle_before_h"][ia] = np.log1p(prv.reindex(trip).to_numpy())
        O["trip_max_home_km"][ia] = g["hk"].transform("max").to_numpy()
    return pd.DataFrame(O, index=H.index).astype(np.float32)


def build(H, D):
    X = pd.DataFrame(index=H.index)
    for c in ["if_type", "own_dev", "event"]:
        X[c] = D[c].to_numpy()
    for c in RAW:
        X[c] = H[c].astype(np.float32).to_numpy()
    X["gap_prev_h"], X["gap_next_h"] = np.log1p(X["gap_prev_h"]), np.log1p(X["gap_next_h"])
    g = H.groupby("rfid_id")
    for c in PCT:
        X[f"vp_{c}"] = g[c].rank(pct=True).to_numpy(np.float32)
    for c in ["if_type", "own_dev", "event"]:
        X[f"vp_{c}"] = pd.Series(X[c].to_numpy()).groupby(H["rfid_id"].to_numpy()).rank(pct=True).to_numpy()
    for c in LEAD:
        for k in (1, 2, -1):
            X[f"{c}_{'n' if k > 0 else 'p'}{abs(k)}"] = shift_within(H, c, k)
    X["hour"] = H["hour"].to_numpy()
    X["hsin"], X["hcos"] = np.sin(X["hour"] * np.pi / 12), np.cos(X["hour"] * np.pi / 12)
    X["state"] = H["state"].to_numpy()
    X = X.join(trip_feats(H))
    X["is_first_of_day"] = (H.groupby(["rfid_id", "part_dt"]).cumcount() == 0).to_numpy().astype(np.float32)
    X["is_last_of_day"] = (H.groupby(["rfid_id", "part_dt"]).cumcount(ascending=False) == 0).to_numpy().astype(
        np.float32)
    return X


def labels(H, L, tol):
    y = np.zeros(len(H), bool)
    for _, l in L.iterrows():
        m = (H["rfid_id"] == l["rfid_id"]).to_numpy()
        y |= m & ((H["ts"] - l["label_ts"]).abs() <= pd.Timedelta(hours=tol)).to_numpy()
    return y


def vessel_metrics(H, s, L):
    r = pd.Series(s, index=H.index).groupby(H["rfid_id"]).rank(ascending=False, method="first")
    res = []
    top = pd.Series(s, index=H.index).groupby(H["rfid_id"]).idxmax()
    for _, l in L.iterrows():
        m = (H["rfid_id"] == l["rfid_id"]).to_numpy()
        if not m.any():
            continue
        dt = (H.loc[m, "ts"] - l["label_ts"]).abs()
        win1 = (dt <= pd.Timedelta(hours=1)).to_numpy()
        if not win1.any():
            continue
        best = r[m][win1].min()
        t1 = H.at[top[l["rfid_id"]], "ts"]
        d1 = abs(t1 - l["label_ts"])
        res.append(dict(hit1_1h=d1 <= pd.Timedelta(hours=1), hit1_3h=d1 <= pd.Timedelta(hours=3),
                        hit1_day=d1 <= pd.Timedelta(hours=12), top5=best <= 5, top24=best <= 24,
                        pct=1 - best / m.sum(), n=m.sum()))
    R = pd.DataFrame(res)
    return dict(n=len(R), hit1_1h=R["hit1_1h"].mean(), hit1_3h=R["hit1_3h"].mean(), hit1_12h=R["hit1_day"].mean(),
                top5=R["top5"].mean(), top24=R["top24"].mean(), pct_med=R["pct"].median())


def fit_predict(kind, Xtr, ytr, gtr, Xva):
    if kind == "rank":
        sizes = pd.Series(gtr).value_counts(sort=False).reindex(pd.unique(gtr)).to_numpy()
        m = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1,
                           n_jobs=C.N_JOBS, random_state=0, lambdarank_truncation_level=30)
        m.fit(Xtr, ytr.astype(int), group=sizes)
        return m.predict(Xva)
    if kind == "bin":
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5,
                               scale_pos_weight=50, verbose=-1, n_jobs=C.N_JOBS, random_state=0)
        return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    if kind == "logit":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000))
        return m.fit(Xtr.fillna(0), ytr).predict_proba(Xva.fillna(0))[:, 1]
    raise ValueError(kind)


def load_light(lab_ids, path=None, per_type=300_000):
    """메모리 절약: IF 학습용은 업종별 표본만, 검증용은 라벨 선박 전체 시간만 읽음"""
    from anomaly_model import FEATS
    path = path or C.FEAT_DIR / "anom_hourly.parquet"
    cols = list(dict.fromkeys(["rfid_id", "fishing_code", "ts", "part_dt", "hour", "state", "at_sea", "moving"]
                              + FEATS + RAW))
    H = pd.read_parquet(path, columns=cols)
    rng = np.random.default_rng(C.RANDOM_STATE)
    samp = [g.iloc[rng.choice(len(g), min(len(g), per_type), replace=False)] for _, g in H.groupby("fishing_code")]
    Htr = pd.concat(samp, ignore_index=True)
    Hv = H[H["rfid_id"].isin(lab_ids)].sort_values(["rfid_id", "ts"]).reset_index(drop=True)
    return Htr, Hv


def main(a):
    L = pd.read_csv(C.RESULT_DIR / "anom_labels.csv", parse_dates=["label_ts"])
    L = L[L["tier"].isin(["A", "B"]) & L["label_ts"].notna()].drop_duplicates("rfid_id")
    Htr, Hv = load_light(L["rfid_id"].unique())
    models = fit_if(Htr, list(C.CLASSES))
    del Htr
    D = components(Hv, models, Hv["fishing_code"].to_numpy())
    X = build(Hv, D)
    y = labels(Hv, L, a.tol)
    print(f"라벨 선박 {Hv['rfid_id'].nunique()}척 / 시간대 {len(Hv):,} / 양성(±{a.tol}h) {y.sum()}")

    ids = Hv["rfid_id"].unique()
    rows = []
    rng = np.random.default_rng(0)
    # 사고 시간대 사전분포: 항적과 매칭 안 된 사고(검증 라벨과 겹치지 않음)의 발생 시각, +1 평활
    Lall = pd.read_csv(C.RESULT_DIR / "anom_labels.csv", parse_dates=["acc_time"])
    hp = (Lall.loc[Lall["tier"] == "none", "acc_time"].dt.hour.value_counts().reindex(range(24), fill_value=0) + 1)
    hp = (hp / hp.sum()).to_numpy()
    X["hour_prior"] = hp[X["hour"].astype(int).to_numpy()]
    end = np.where(X["in_trip"] == 1, np.exp(-X["h_to_trip_end"].fillna(99) / 3), 0.05)
    ano = (2 * D["if_type"] + D["own_dev"]).to_numpy() / 3
    base = {"무작위": rng.random(len(Hv)),
            "기존 제출식 (IF2:z1)": ano,
            "기존 event 규칙": D["event"].to_numpy(),
            "규칙: 출어 끝 근접": end + 1e-3 * ano,
            "규칙: 출어 끝 × 사고 시간대": end * X["hour_prior"].to_numpy() + 1e-4 * ano,
            "규칙: 출어 끝 × 시간대 × 이상점수": end * X["hour_prior"].to_numpy() * (0.5 + ano)}
    for k, s in base.items():
        rows.append(dict(method=k, rep="-") | vessel_metrics(Hv, s, L))
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)
    feats = list(X.columns)
    small = ["in_trip", "h_to_trip_end", "is_trip_last", "trip_pos", "trip_len_h", "hsin", "hcos", "hour_prior",
             "vp_own_dev", "vp_if_type", "vp_event", "idle_after_h", "at_sea"]
    for kind in ["rank", "bin", "logit", "logit_small"]:
        fs = small if kind == "logit_small" else feats
        kind_ = "logit" if kind == "logit_small" else kind
        for rep in range(a.reps):
            oof = np.zeros(len(Hv))
            for tr_i, va_i in KFold(5, shuffle=True, random_state=rep).split(ids):
                tr = Hv["rfid_id"].isin(ids[tr_i]).to_numpy()
                oof[~tr] = fit_predict(kind_, X.loc[tr, fs], y[tr], Hv.loc[tr, "rfid_id"].to_numpy(),
                                       X.loc[~tr, fs])
            rows.append(dict(method=kind, rep=rep) | vessel_metrics(Hv, oof, L))
            print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False, header=False), flush=True)
    R = pd.DataFrame(rows)
    S = R.groupby("method", sort=False).mean(numeric_only=True).round(3)
    S.to_csv(C.RESULT_DIR / f"ny_anom_rank_tol{a.tol}.csv", encoding="utf-8-sig")
    print("\n(반복 평균)\n", S.to_string())

    # 전체 라벨로 학습한 rank 모델의 피처 중요도 (해석용)
    sizes = pd.Series(Hv["rfid_id"]).value_counts(sort=False).reindex(ids).to_numpy()
    m = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1, n_jobs=C.N_JOBS,
                       random_state=0).fit(X[feats], y.astype(int), group=sizes)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\n중요 피처:", ", ".join(f"{k}({v:.0f})" for k, v in imp.head(15).items()))
    imp.to_csv(C.RESULT_DIR / "ny_anom_rank_importance.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    main(ap.parse_args())
