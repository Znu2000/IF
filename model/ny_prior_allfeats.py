"""
[NY-5] 진우 step1(지역 사전확률) + 나연 실험(피처 전체가 top-k 보다 나음) 결합 확인
  step1 은 top-50 피처 + 3모델 앙상블. 여기서는 413개 전체 + 지역 사전확률을 LightGBM 하나로 확인.
  규칙: 학습은 2026 만. 평가 = cv26 (검증 선박 2026) / val25 (검증 선박 2025 항적 = step1 의 cross_unseen)
  지역 사전확률 계산은 step1_experiments.region_prior 와 같은 방식 (학습 fold 선박만 이웃, 학습 선박은 자기 제외)
실행: python ny_prior_allfeats.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import BallTree

import cv_common as V
import fe_config as C
from ny_common import LGB, load

K_NB = 30
PRIOR_SETS = {"home": ["home_lat", "home_lon"], "fish": ["fishing_lat", "fishing_lon"]}
SHIFT = ["mv_lowfreq_ratio_mean", "mv_lowfreq_ratio_median"]  # step1 이 찾은 연도 차이 피처


def region_prior(S_ref, y_ref, Q, self_in_ref=False):
    out = {}
    for name, cols in PRIOR_SETS.items():
        ref = S_ref[cols].to_numpy(float)
        ok = ~np.isnan(ref).any(1)
        tree = BallTree(np.radians(ref[ok]), metric="haversine")
        yr = y_ref[ok]
        q = Q[cols].to_numpy(float)
        qok = ~np.isnan(q).any(1)
        P = np.full((len(Q), 4), np.nan)
        dist = np.full(len(Q), np.nan)
        d, i = tree.query(np.radians(q[qok]), k=K_NB + int(self_in_ref))
        if self_in_ref:
            d, i = d[:, 1:], i[:, 1:]
        P[qok] = np.stack([(yr[i] == c).mean(1) for c in range(4)], 1)
        dist[qok] = d.mean(1) * 6371.0
        for c, cl in enumerate(C.CLASSES):
            out[f"prior_{name}_{cl}"] = P[:, c]
        out[f"prior_{name}_dist_km"] = dist
    return pd.DataFrame(out, index=Q.index)


def main():
    t0 = time.time()
    S, T25, _ = load()
    S = S.reset_index(drop=True)
    T25 = T25.reset_index(drop=True)
    yo = {c: i for i, c in enumerate(C.CLASSES)}
    y, y25 = S["fishing_code"].map(yo).to_numpy(), T25["fishing_code"].map(yo).to_numpy()
    F = V.folds(S, y)
    allf = [c for c in V.feature_pool(S, "B") if c in T25.columns and c != "year"]
    cfgs = {"전체 413": (allf, False, "balanced"),
            "전체 413, 가중 없음": (allf, False, None),
            "전체 413 + 지역사전": (allf, True, "balanced"),
            "전체 413 + 지역사전, 가중 없음": (allf, True, None),
            "전체 − 연도차이 2 + 지역사전, 가중 없음": ([c for c in allf if c not in SHIFT], True, None)}
    rows, oof = [], {}
    for name, (fs, use_prior, cw) in cfgs.items():
        P26, P25 = np.zeros((len(S), 4)), np.zeros((len(T25), 4))
        for tr, va in F:
            Str, Sva = S.iloc[tr], S.iloc[va]
            mv = T25["rfid_id"].isin(Sva["rfid_id"]).to_numpy()
            Q25 = T25[mv]
            Xtr, Xva, X25 = Str[fs], Sva[fs], Q25[fs]
            if use_prior:
                Xtr = Xtr.join(region_prior(Str, y[tr], Str, self_in_ref=True))
                Xva = Xva.join(region_prior(Str, y[tr], Sva))
                X25 = X25.join(region_prior(Str, y[tr], Q25))
            m = lgb.LGBMClassifier(**(LGB | dict(class_weight=cw))).fit(Xtr, y[tr])
            P26[va], P25[mv] = m.predict_proba(Xva), m.predict_proba(X25)
        p26, p25 = P26.argmax(1), P25.argmax(1)
        rows.append(dict(config=name, acc_cv26=accuracy_score(y, p26), acc_val25=accuracy_score(y25, p25),
                         macro_val25=f1_score(y25, p25, average="macro")))
        oof[name] = P25
        print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False, header=len(rows) == 1),
              f" ({(time.time() - t0) / 60:.1f}분)", flush=True)
    rng = np.random.default_rng(0)
    base = (oof["전체 413"].argmax(1) == y25).astype(float)
    idx = [rng.integers(0, len(y25), len(y25)) for _ in range(1000)]
    for r in rows:
        d = (oof[r["config"]].argmax(1) == y25) - base
        bs = [d[i].mean() for i in idx]
        r["d_val25"], r["ci_val25"] = d.mean(), f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
    R = pd.DataFrame(rows)
    R.round(4).to_csv(C.RESULT_DIR / "ny_prior_allfeats.csv", index=False, encoding="utf-8-sig")
    print("\n", R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
