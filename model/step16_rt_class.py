"""
[STEP16-3] 관측 기간별 실시간 분류 곡선 ─ 그 배의 첫 기록 후 N일까지만 보고 업종을 맞힐 수 있나
  지금까지 분류 성능은 배마다 6개월 전체 항적으로 만든 피처 기준. 실시간에는 미래 항적이 없음
  자르기: 배별 첫 기록 + N일 (N = 3, 7, 14, 30, 90, 150). 항구 격자·출어일 유형 모델은 전체 학습 것을 고정
          모항·조업지·피처는 잘린 항적으로 다시 계산. 사전확률 참조 선박은 학습 fold 의 전체 항적 (운영 때 이미 있는 자료)
  학습 방식 (모두 fold_rankings 의 같은 fold, 2026 만, LGB_BASE 클래스 가중 없음 2시드, 피처 = 선박 피처 + rv_·pd_ + 지역 사전확률)
    full    : 전체 항적으로 학습, 잘린 항적으로 예측
    matched : 같은 길이로 자른 항적으로 학습·예측
    aug     : 학습 선박의 전체 + 모든 길이를 한꺼번에 학습 (배당 7행), 잘린 항적으로 예측. fold 는 배 단위라 누수 없음
  한계: 첫 기록이 대부분 1월이므로 N 이 짧을수록 겨울 항적만 봄. 계절 차이는 이 실험에서 분리 안 됨
  출력: features_v/rt_<N>/ship_features.parquet, results/step16_oof_<방식>_<N>.npy, results/step16_rt_<시각>.csv
실행: python step16_rt_class.py [--eval-only]
"""
import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import cv_common as V
import fe_aggregate as A
import fe_config as C
import step4_behavior as B
import train_classifiers as TC
from step1_experiments import region_prior
from step4_final import attach

STAMP = time.strftime("%m%d_%H%M")
CUTS = [3, 7, 14, 30, 90, 150]
VAR_ROOT = C.MODEL_ROOT / "features_v"
SEEDS = [0, 1]


def build(n):
    out = VAR_ROOT / f"rt_{n}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ship_features.parquet"
    if path.exists():
        print(f"N={n}: 있음", flush=True)
        return
    t0 = time.time()
    H = A.load_hourly("hourly")
    H["ts"] = pd.to_datetime(H["part_dt"].astype(str)) + pd.to_timedelta(H["hour"].astype(int), unit="h")
    first = H.groupby("rfid_id")["ts"].transform("min")
    H = H[H["ts"] < first + pd.Timedelta(days=n)].reset_index(drop=True)
    print(f"N={n}: {len(H):,}행", flush=True)
    H = A.add_port_state(H, pd.read_csv(C.FEAT_DIR / "port_cells.csv"))
    D, cells1k = A.build_shipday(H)
    S, _ = A.build_ship(H, D, cells1k, joblib.load(C.FEAT_DIR / "daytype_model.joblib"))
    # 행동 피처 rv_·pd_ (step4_behavior 와 같은 함수, 잘린 항적 기준)
    Hb = H[["rfid_id", "ts", "lon_c", "lat_c", "port_stay", "port_km", "state"]].copy()
    Hb["at_sea"] = ((Hb["port_stay"] == 0) & (Hb["port_km"] > 1)).astype(np.float32)
    Hb["t"] = (Hb["ts"].astype("int64") // 3_600_000_000_000).astype(np.int64)
    F = Hb[(Hb["at_sea"] == 1) & Hb["state"].isin(B.FISH_STATES)].dropna(subset=["lon_c", "lat_c"])
    Be = pd.concat([B.revisit(F), B.period(Hb)], axis=1)
    Be.index.name = "rfid_id"
    Be = Be.reset_index()
    S = S.merge(Be, on="rfid_id", how="left")
    S.to_parquet(path, index=False)
    print(f"N={n}: 생성 {S.shape} {(time.time() - t0) / 60:.1f}분", flush=True)


def load_all():
    S, y = V.load_ship()
    pool = V.feature_pool(S, "B")
    attach(S, "")
    add = [c for c in S.columns if c.startswith(("rv_", "pd_"))]
    ids = S["rfid_id"].astype(str).to_numpy()
    Sc = {}
    for n in CUTS:
        X = pd.read_parquet(VAR_ROOT / f"rt_{n}" / "ship_features.parquet")
        X["_rid"] = X["rfid_id"].astype(str)
        X = X.set_index("_rid").reindex(ids).reset_index(drop=True)
        Sc[n] = X.reindex(columns=list(dict.fromkeys(pool + add + ["home_lat", "home_lon", "fishing_lat", "fishing_lon"])))
        print(f"N={n}: 결측 home {Sc[n]['home_lat'].isna().mean():.3f} fishing {Sc[n]['fishing_lat'].isna().mean():.3f}", flush=True)
    return S, y, pool + add, Sc


def fit_pred(Xa, ya, Xbs):
    params = dict(TC.LGB_BASE, class_weight=None)
    out = [np.zeros((len(x), len(C.CLASSES))) for x in Xbs]
    for sd in SEEDS:
        m = lgb.LGBMClassifier(**params, random_state=sd, n_jobs=C.N_JOBS).fit(Xa, ya)
        for i, x in enumerate(Xbs):
            out[i] += m.predict_proba(x) / len(SEEDS)
    return out


def main(a):
    t0 = time.time()
    if not a.eval_only:
        for n in CUTS:
            build(n)
    S, y, cols, Sc = load_all()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    keys = ["full"] + CUTS
    O = {(m, n): np.zeros((len(y), len(C.CLASSES))) for m in ("full", "matched", "aug") for n in keys}

    def design(X, ref, yr, self_in):
        return pd.concat([X[cols].reset_index(drop=True), region_prior(ref, yr, X.reset_index(drop=True), self_in_ref=self_in).reset_index(drop=True)], axis=1)

    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), va)
        Str = S.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        Xa_full = design(S.iloc[tr], Str, ytr, True)
        Xb = {"full": design(S.iloc[va], Str, ytr, False)}
        Xa = {"full": Xa_full}
        for n in CUTS:
            Xb[n] = design(Sc[n].iloc[va], Str, ytr, False)
            Xa[n] = design(Sc[n].iloc[tr], Str, ytr, True)
        P = fit_pred(Xa_full, ytr, [Xb[k] for k in keys])
        for k, p in zip(keys, P):
            O["full", k][va] = p
        for n in keys:
            if n == "full":
                O["matched", n][va] = O["full", n][va]
                continue
            O["matched", n][va] = fit_pred(Xa[n], ytr, [Xb[n]])[0]
        Xau = pd.concat([Xa[k] for k in keys], ignore_index=True)
        yau = np.tile(ytr, len(keys))
        P = fit_pred(Xau, yau, [Xb[k] for k in keys])
        for k, p in zip(keys, P):
            O["aug", k][va] = p
        print(f"fold {f} 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    base = (O["full", "full"].argmax(1) == y).astype(float)
    rng = np.random.default_rng(0)
    rows = []
    for (m, n), P in O.items():
        np.save(C.RESULT_DIR / f"step16_oof_{m}_{n}.npy", P)
        acc = (P.argmax(1) == y).astype(float)
        cur = (O["full", n].argmax(1) == y).astype(float)
        d = acc - cur
        boot = d[rng.integers(0, len(d), (3000, len(d)))].mean(1)
        rows.append(dict(방식=m, N일=n, cv=acc.mean(), macro=V.macro_f1(y, P.argmax(1)), logloss=log_loss(y, P), 대비full학습=d.mean(), SE=boot.std(),
                         fold차=" ".join(f"{d[fold_id == f].mean():+.3f}" for f in range(len(folds))), 대비전체항적=acc.mean() - base.mean()))
    E = pd.DataFrame(rows)
    E.round(4).to_csv(C.RESULT_DIR / f"step16_rt_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(E.round(4).to_string(index=False), flush=True)
    # 활동량이 적은 배(기록 일수 하위 25%)만 따로
    q = S["n_days"].to_numpy() <= np.quantile(S["n_days"], 0.25)
    print("기록 일수 하위 25% 배 정확도:", {f"{m}_{n}": round(float((P[q].argmax(1) == y[q]).mean()), 4) for (m, n), P in O.items() if n == "full" or m == "aug"}, flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    main(ap.parse_args())
