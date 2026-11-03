"""
[STEP4] 행동 관점 피처 4종 + 행동 유사 kNN 측정 (단일 LightGBM, 1단계와 같은 fold·피처 선택)
  base     : 1단계 최종과 같은 입력 (fold 별 상위 k + 주변 30척 업종 비율, 연도 차이 피처 제외)
  +revisit / +dbscan / +period / +congest : step4_behavior.py 피처 그룹 추가
  +bknn    : 행동 피처 공간(상위 k)에서 가까운 학습 선박 30·100척 업종 비율 (학습 선박 자신 제외)
  +all     : 전부   (설정 이름은 revisit+period 처럼 묶고, @시드 로 LightGBM 시드 지정)
지표: cv(micro-F1 = 정확도), cross_seen(테스트 기간 항적으로 예측한 학습 겹침 선박 정확도)
실행: python step4_eval.py [설정 ...]
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

import cv_common as V
import fe_config as C
from step1_experiments import region_prior, shift_feats

GROUPS = {"revisit": "rv_", "dbscan": "db_", "period": "pd_", "congest": "cg_"}
CONFIGS = ["base", "revisit", "dbscan", "period", "congest", "bknn", "all"]
LGB = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, random_state=0, n_jobs=-1)


def bknn(Ztr, ytr, Zq, self_in):
    out = {}
    for k in (30, 100):
        nn = NearestNeighbors(n_neighbors=k + int(self_in)).fit(Ztr)
        _, i = nn.kneighbors(Zq)
        i = i[:, 1:] if self_in else i
        for c, cl in enumerate(C.CLASSES):
            out[f"bknn{k}_{cl}"] = (ytr[i] == c).mean(1)
    return pd.DataFrame(out)


def main(cfgs):
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    for D, tag in [(S, ""), (T, "_test")]:
        B = pd.read_parquet(C.FEAT_DIR / f"behav_feats{tag}.parquet")
        B["rfid_id"] = B["rfid_id"].astype(str)
        m = D[["rfid_id"]].astype(str).merge(B, on="rfid_id", how="left").drop(columns="rfid_id")
        for c in m:
            D[c] = m[c].to_numpy()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    k = R["B"]["k_star"]
    lab = dict(zip(S["rfid_id"].astype(str), y))
    seen = T["rfid_id"].astype(str).isin(lab).to_numpy()
    y_ov = T["rfid_id"].astype(str)[seen].map(lab).to_numpy()
    drop = set(shift_feats(S, T, V.feature_pool(S, "B")))
    gcols = {g: [c for c in S.columns if c.startswith(p)] for g, p in GROUPS.items()}
    gshift = {g: set(shift_feats(S, T, cols)) for g, cols in gcols.items()}
    print(f"학습 {len(y)}척, 테스트 {len(T)}척 (겹침 {seen.sum()}), k={k}, 그룹 피처 수 "
          f"{ {g: len(c) for g, c in gcols.items()} }, 연도 차이로 뺀 것 { {g: sorted(s) for g, s in gshift.items() if s} }",
          flush=True)
    rows = []
    for cfg in cfgs:
        name, _, seed = cfg.partition("@")
        LGB["random_state"] = int(seed or 0)
        gs = list(GROUPS) if name == "all" else [g for g in name.split("+") if g in GROUPS]
        add = [c for g in gs for c in gcols[g] if c not in gshift[g]]
        oof = np.zeros((len(y), len(C.CLASSES)))
        cs = np.zeros((seen.sum(), len(C.CLASSES)))
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), va)
            feats = [c for c in R["B"]["folds"][f] if c not in drop][:k]
            Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
            Q = [Str, Sva, T]
            X = [pd.concat([D[feats + add].reset_index(drop=True),
                            region_prior(Str, y[tr], D, self_in_ref=(j == 0)).reset_index(drop=True)], axis=1)
                 for j, D in enumerate(Q)]
            if "bknn" in name or name == "all":
                med = Str[feats].median()
                lo, hi = Str[feats].quantile(0.01), Str[feats].quantile(0.99)
                mu, sd = Str[feats].clip(lo, hi, axis=1).fillna(med).mean(), Str[feats].clip(lo, hi, axis=1).fillna(med).std() + 1e-9
                Z = [((D[feats].clip(lo, hi, axis=1).fillna(med) - mu) / sd).to_numpy() for D in Q]
                X = [pd.concat([x, bknn(Z[0], y[tr], z, j == 0)], axis=1) for j, (x, z) in enumerate(zip(X, Z))]
            m = lgb.LGBMClassifier(**LGB).fit(X[0], y[tr])
            oof[va] = m.predict_proba(X[1])
            cs += m.predict_proba(X[2][seen]) / len(folds)
        r = dict(config=cfg, n_add=len(add), cv=accuracy_score(y, oof.argmax(1)), cv_macro=V.macro_f1(y, oof.argmax(1)),
                 cross_seen=accuracy_score(y_ov, cs.argmax(1)))
        rows.append(r)
        np.save(C.RESULT_DIR / f"step4_oof_{cfg}.npy", oof)
        print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, f"({(time.time() - t0) / 60:.1f}분)",
              flush=True)
    E = pd.DataFrame(rows)
    b = E[E.config.str.split("@").str[0] == "base"].iloc[0] if (E.config == "base").any() else None
    if b is not None:
        E["d_cv"], E["d_cross_seen"] = E.cv - b.cv, E.cross_seen - b.cross_seen
    print(E.round(4).to_string(index=False))
    E.round(4).to_csv(C.RESULT_DIR / ("step4_eval.csv" if cfgs == CONFIGS else "step4_eval_extra.csv"), index=False, encoding="utf-8-sig")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main(sys.argv[1:] or CONFIGS)
