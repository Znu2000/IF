"""
[RANK] 피처 랭킹 + 상위 k 선정  (선박 단위, 전부 fold 안에서 계산 → 선정 누수 없음)

방법 (외부 5-fold 각각의 train 부분에서만)
  0) 라벨 미사용 사전 필터 : 결측 60% 초과 / 상수 피처 제거
  1) 내부 3-fold LightGBM 으로 세 가지 지표
       a. mean |SHAP|  (LightGBM pred_contrib)
       b. permutation importance (내부 valid macro-F1 감소량)
       c. Mutual Information (외부 train 전체)
     + Boruta 식 shadow 검정: 모든 피처를 행 순서만 섞은 그림자 피처를 같이 넣고,
       실제 피처 SHAP > 그림자 최댓값 SHAP 인 내부 fold 수(0~3) = hits
  2) 통합 점수 = 0.4·rank(SHAP) + 0.4·rank(perm) + 0.2·rank(MI)  (작을수록 상위)
  3) 중복 제거 : 순위 순서대로 훑으며 이미 뽑힌 피처와 Spearman |ρ| ≥ 0.95 면 건너뜀
  4) k 스윕 : 외부 train 랭킹 상위 k 로 학습 → 외부 valid macro-F1.
     k* = "최고점 − 1 표준오차" 안에 드는 가장 작은 k  (1-SE rule, Breiman et al. 1984)

출력 (results/)
  feature_ranking_A.csv / _B.csv : 전체 fold 평균 랭킹 (A=위치 제외, B=위치 포함)
  k_sweep.csv, k_sweep.png       : k 별 macro-F1 평균±SE
  fold_rankings.json             : fold 별 순위 목록 + k*  (train_classifiers.py 가 그대로 사용)

실행: python rank_features.py
"""
import json
import time
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import cv_common as V
import fe_config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
K_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150]
RANK_PARAMS = dict(n_estimators=200, learning_rate=0.06, num_leaves=31, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced",
                   reg_lambda=1.0, max_bin=63, verbose=-1, n_jobs=C.N_JOBS)
CORR_THR = 0.95


def lgbm(**kw):
    return lgb.LGBMClassifier(**(RANK_PARAMS | kw), random_state=C.RANDOM_STATE)


def rank_on_train(X, y, rng):
    """외부 train(X: DataFrame) → 순위 DataFrame"""
    cols = list(X.columns)
    n = len(cols)
    Xn = X.to_numpy(np.float32)
    shap_sum, perm_sum, hits = np.zeros(n), np.zeros(n), np.zeros(n)
    inner = V.StratifiedGroupKFold(3, shuffle=True, random_state=C.RANDOM_STATE).split(Xn, y, np.arange(len(y)))
    for tr, va in inner:
        Xt = Xn[tr]
        shadow = np.column_stack([rng.permutation(Xt[:, j]) for j in range(n)])  # 열마다 행 순서만 섞음
        m = lgbm().fit(np.hstack([Xt, shadow]), y[tr])
        sub = rng.choice(len(tr), min(2000, len(tr)), replace=False)
        contrib = m.booster_.predict(np.hstack([Xt, shadow])[sub], pred_contrib=True)
        contrib = np.abs(contrib.reshape(len(sub), len(C.CLASSES), -1)[:, :, :-1]).sum(1).mean(0)
        real, sh = contrib[:n], contrib[n:]
        shap_sum += real
        hits += real > sh.max()
        # 순열 중요도: 그림자 없는 모델, 이번 내부 fold SHAP 상위 120개만 (나머지는 사실상 0)
        m2 = lgbm().fit(Xt, y[tr])
        Xv = Xn[va].copy()
        base = V.macro_f1(y[va], m2.booster_.predict(Xv).argmax(1))
        for j in np.argsort(-real)[:120]:
            keep = Xv[:, j].copy()
            Xv[:, j] = rng.permutation(keep)
            perm_sum[j] += base - V.macro_f1(y[va], m2.booster_.predict(Xv).argmax(1))
            Xv[:, j] = keep
    sub = rng.choice(len(y), min(3000, len(y)), replace=False)
    Xm = X.iloc[sub]
    mi = mutual_info_classif(Xm.fillna(Xm.median()).fillna(0), y[sub], random_state=C.RANDOM_STATE)
    R = pd.DataFrame({"feature": cols, "shap": shap_sum / 3, "perm": perm_sum / 3, "mi": mi, "hits": hits})
    R["score"] = (0.4 * R["shap"].rank(ascending=False) + 0.4 * R["perm"].rank(ascending=False)
                  + 0.2 * R["mi"].rank(ascending=False))
    R = R.sort_values("score").reset_index(drop=True)
    # 중복 제거 (Spearman |ρ| ≥ 0.95, 상위 피처가 대표)
    rho = X[R["feature"]].rank().corr().abs().to_numpy()
    keep, rep = [], {}
    for i, f in enumerate(R["feature"]):
        dup = [j for j in keep if rho[i, j] >= CORR_THR]
        if dup:
            rep[f] = R["feature"][dup[0]]
        else:
            keep.append(i)
    R["redundant_of"] = R["feature"].map(rep)
    return R


def prefilter(X):
    ok = (X.isna().mean() <= 0.6) & (X.nunique() > 1)
    return X.loc[:, ok]


def run_pool(S, y, F, pool):
    cols = V.feature_pool(S, pool)
    rng = np.random.default_rng(C.RANDOM_STATE)
    fold_rank, sweep, tables = [], [], []
    for fi, (tr, va) in enumerate(F):
        t = time.time()
        Xtr = prefilter(S.iloc[tr][cols])
        R = rank_on_train(Xtr, y[tr], rng)
        R["fold"] = fi
        tables.append(R)
        order = R.loc[R["redundant_of"].isna(), "feature"].tolist()
        fold_rank.append(order)
        for k in K_GRID + [len(order)]:
            if k > len(order):
                continue
            fs = order[:k]
            m = lgbm().fit(S.iloc[tr][fs], y[tr])
            sweep.append(dict(pool=pool, fold=fi, k=k if k in K_GRID else -1, n=k,
                              f1=V.macro_f1(y[va], m.predict(S.iloc[va][fs]))))
        print(f"  [{pool}] fold {fi}: 후보 {Xtr.shape[1]} → 비중복 {len(order)}  ({time.time() - t:.0f}s)", flush=True)

    sw = pd.DataFrame(sweep)
    sw.loc[sw["k"] == -1, "k"] = 9999  # "전체(비중복)"
    agg = sw.groupby("k")["f1"].agg(["mean", "std", "count"])
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    best = agg["mean"].idxmax()
    k_star = int(agg.index[agg["mean"] >= agg.loc[best, "mean"] - agg.loc[best, "se"]].min())
    if k_star == 9999:
        k_star = int(sw.loc[sw["k"] == 9999, "n"].min())
    print(f"  [{pool}] k* = {k_star}  (best k={best}, F1 {agg.loc[best, 'mean']:.4f}±{agg.loc[best, 'se']:.4f})")

    # 전체 fold 평균 랭킹 (보고용)
    T = pd.concat(tables)
    G = T.groupby("feature").agg(shap=("shap", "mean"), perm=("perm", "mean"), mi=("mi", "mean"),
                                 boruta_hits=("hits", "sum"), score=("score", "mean"),
                                 redundant_folds=("redundant_of", lambda s: int(s.notna().sum())),
                                 redundant_of=("redundant_of", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""))
    G["topk_freq"] = pd.Series({f: sum(f in o[:k_star] for o in fold_rank) / len(fold_rank) for f in G.index})
    G = G.sort_values(["topk_freq", "score"], ascending=[False, True])
    G["final_rank"] = np.arange(1, len(G) + 1)
    meta = [V.feat_meta(f) for f in G.index]
    G.insert(0, "unit", [m[0] for m in meta])
    G.insert(1, "category", [m[1] for m in meta])
    G.insert(2, "is_loc", [f in V.LOC_COLS for f in G.index])
    G.insert(3, "is_hypothesis", [V.is_hyp(f) for f in G.index])
    G["signal"] = (G["boruta_hits"] >= 0.5 * 3 * len(F)) & (G["perm"] > 0)
    G.index.name = "feature"
    G.round(5).to_csv(C.RESULT_DIR / f"feature_ranking_{pool}.csv", encoding="utf-8-sig")
    return dict(k_star=k_star, folds=fold_rank), agg.assign(pool=pool).reset_index()


def main():
    S, y = V.load_ship()
    F = V.folds(S, y)
    print(f"선박 {len(S):,}척, 클래스 분포 {np.bincount(y)}")
    out, aggs = {}, []
    for pool in ["A", "B"]:
        out[pool], agg = run_pool(S, y, F, pool)
        aggs.append(agg)
    out["fold_val_idx"] = [va.tolist() for _, va in F]
    (C.RESULT_DIR / "fold_rankings.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    A = pd.concat(aggs)
    A.to_csv(C.RESULT_DIR / "k_sweep.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    for pool, lab in [("A", "Model A (위치 제외)"), ("B", "Model B (위치 포함)")]:
        a = A[(A["pool"] == pool) & (A["k"] != 9999)]
        ax.errorbar(a["k"], a["mean"], yerr=a["se"], marker="o", capsize=3, label=f"{lab}  k*={out[pool]['k_star']}")
        ax.axvline(out[pool]["k_star"], ls=":", color=ax.lines[-1].get_color())
    ax.set_xscale("log"); ax.set_xlabel("상위 k 피처 수"); ax.set_ylabel("선박 macro-F1 (5-fold 평균 ± SE)")
    ax.set_title("k 스윕 (fold 내부 랭킹, 1-SE rule)"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(C.RESULT_DIR / "k_sweep.png", bbox_inches="tight", dpi=120)
    print("✅ 랭킹 완료")


if __name__ == "__main__":
    V.set_font()
    main()
