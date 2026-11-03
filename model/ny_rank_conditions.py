"""
[NY-3] 피처 랭킹 '조건'을 바꿔 피처 구성을 만들고 비교 ─ 나연
  rank_features.py 와 같은 지표(SHAP·순열 중요도·MI·Boruta 그림자 검정·Spearman 중복 제거)를 fold 마다
  한 번만 계산해 두고(캐시), 조건(지표 비율·중복 기준·개수 k·노이즈 검정 통과 여부·채점 기준)을 바꿔
  fold 별 피처 목록을 만든다. 피처 선정은 전부 그 fold 의 학습 선박(2026 항적)에서만 → 선정 누수 없음.

  채점 (팀장 지침: 학습 2026 만, 2025 는 검증만):
    · fold 학습 선박의 2026 항적으로 LightGBM 학습
    · fold 검증 선박의 2025 항적으로 정확도(리더보드 micro-F1) / macro-F1
실행: python ny_rank_conditions.py            (1단계 캐시가 있으면 재사용)
"""
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score

import cv_common as V
import fe_config as C
from ny_common import LGB, load
from rank_features import RANK_PARAMS

CACHE = C.RESULT_DIR / "ny_rank_cache.pkl"
NC = len(C.CLASSES)


def prefilter(X):
    ok = (X.isna().mean() <= 0.6) & (X.nunique() > 1)
    return X.loc[:, ok]


def raw_importance(X, y, rng):
    """rank_features.rank_on_train 과 같은 계산 + 순열 중요도를 정확도 기준으로도 측정"""
    cols = list(X.columns)
    n = len(cols)
    Xn = X.to_numpy(np.float32)
    shap_sum, pm, pa, hits = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    mk = lambda: lgb.LGBMClassifier(**RANK_PARAMS, random_state=C.RANDOM_STATE)
    inner = V.StratifiedGroupKFold(3, shuffle=True, random_state=C.RANDOM_STATE).split(Xn, y, np.arange(len(y)))
    for tr, va in inner:
        Xt = Xn[tr]
        shadow = np.column_stack([rng.permutation(Xt[:, j]) for j in range(n)])
        m = mk().fit(np.hstack([Xt, shadow]), y[tr])
        sub = rng.choice(len(tr), min(2000, len(tr)), replace=False)
        contrib = m.booster_.predict(np.hstack([Xt, shadow])[sub], pred_contrib=True)
        contrib = np.abs(contrib.reshape(len(sub), NC, -1)[:, :, :-1]).sum(1).mean(0)
        real, sh = contrib[:n], contrib[n:]
        shap_sum += real
        hits += real > sh.max()
        m2 = mk().fit(Xt, y[tr])
        Xv = Xn[va].copy()
        p0 = m2.booster_.predict(Xv).argmax(1)
        bm, ba = V.macro_f1(y[va], p0), accuracy_score(y[va], p0)
        for j in np.argsort(-real)[:120]:
            keep = Xv[:, j].copy()
            Xv[:, j] = rng.permutation(keep)
            p = m2.booster_.predict(Xv).argmax(1)
            pm[j] += bm - V.macro_f1(y[va], p)
            pa[j] += ba - accuracy_score(y[va], p)
            Xv[:, j] = keep
    sub = rng.choice(len(y), min(3000, len(y)), replace=False)
    Xm = X.iloc[sub]
    mi = mutual_info_classif(Xm.fillna(Xm.median()).fillna(0), y[sub], random_state=C.RANDOM_STATE)
    R = pd.DataFrame({"feature": cols, "shap": shap_sum / 3, "perm_macro": pm / 3, "perm_acc": pa / 3,
                      "mi": mi, "hits": hits})
    rho = X.rank().corr().abs()
    return R, rho


def select(R, rho, w=(0.4, 0.4, 0.2), perm="perm_macro", corr=0.95, k=None, boruta=False):
    R = R.copy()
    R["score"] = (w[0] * R["shap"].rank(ascending=False) + w[1] * R[perm].rank(ascending=False)
                  + w[2] * R["mi"].rank(ascending=False))
    R = R.sort_values("score")
    if boruta:  # 내부 3-fold 중 2번 이상 그림자 최댓값보다 SHAP 이 큰 피처만
        R = R[R["hits"] >= 2]
    feats = R["feature"].tolist()
    keep = []
    for f in feats:
        if not any(rho.at[f, g] >= corr for g in keep):
            keep.append(f)
    return keep if k is None else keep[:k]


CONDITIONS = {
    # 이름: select() 인자 (None 이면 413개 전체)
    "C0 전체 피처 (기준)": None,
    "C1 팀 기본 (0.4/0.4/0.2, ρ0.95, k50)": dict(k=50),
    "C2 팀 기본 k30": dict(k=30),
    "C3 팀 기본 k100": dict(k=100),
    "C4 팀 기본 비중복 전체": dict(),
    "C5 SHAP 만 k50": dict(w=(1, 0, 0), k=50),
    "C6 정확도 기준 순열 k50": dict(w=(0.3, 0.5, 0.2), perm="perm_acc", k=50),
    "C7 정확도 기준 순열 k100": dict(w=(0.3, 0.5, 0.2), perm="perm_acc", k=100),
    "C8 중복 기준 ρ0.90 k50": dict(corr=0.90, k=50),
    "C9 노이즈 검정 통과만 (비중복)": dict(boruta=True),
    "C10 노이즈 검정 통과 + 정확도 순열 k100": dict(boruta=True, w=(0.3, 0.5, 0.2), perm="perm_acc", k=100),
}


def main():
    t0 = time.time()
    S, T25, _ = load()
    S = S.reset_index(drop=True)
    y = S["fishing_code"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()
    F = V.folds(S, y)
    poolB = [c for c in V.feature_pool(S, "B") if c in T25.columns and c != "year"]

    if CACHE.exists():
        cache = pickle.loads(CACHE.read_bytes())
    else:
        cache = []
        rng = np.random.default_rng(C.RANDOM_STATE)
        for fi, (tr, va) in enumerate(F):
            Xtr = prefilter(S.iloc[tr][poolB])
            cache.append(raw_importance(Xtr, y[tr], rng))
            print(f"[1단계] fold {fi}: 후보 {Xtr.shape[1]}개 중요도 계산 ({(time.time() - t0) / 60:.1f}분)", flush=True)
        CACHE.write_bytes(pickle.dumps(cache))

    y25 = T25["fishing_code"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()
    rows = []
    for name, cond in CONDITIONS.items():
        P = np.zeros((len(T25), NC))
        nfe = []
        for fi, (tr, va) in enumerate(F):
            cols = poolB if cond is None else select(*cache[fi], **cond)
            nfe.append(len(cols))
            va_ids = set(S["rfid_id"].iloc[va])
            mv = T25["rfid_id"].isin(va_ids).to_numpy()
            P[mv] = lgb.LGBMClassifier(**LGB).fit(S.iloc[tr][cols], y[tr]).predict_proba(T25.loc[mv, cols])
        p = P.argmax(1)
        rows.append(dict(condition=name, n_feats=round(np.mean(nfe)), acc_val25=accuracy_score(y25, p),
                         macro_val25=f1_score(y25, p, average="macro")))
        print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False, header=len(rows) == 1),
              f"  ({(time.time() - t0) / 60:.1f}분)", flush=True)
    R = pd.DataFrame(rows)
    R.round(4).to_csv(C.RESULT_DIR / "ny_rank_conditions.csv", index=False, encoding="utf-8-sig")
    print("\n", R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
