"""
[RANK] 피처 랭킹 + 상위 k 선정  (선박 단위, 전부 fold 안에서 계산 → 선정 누수 없음)

방법 (외부 5-fold 각각의 train 부분에서만)
  0) 라벨 미사용 사전 필터 : 결측 60% 초과 / 상수 피처 제거
  1) 내부 3-fold LightGBM 으로 세 가지 지표
       a. mean |SHAP|  (LightGBM pred_contrib)
       b. permutation importance (내부 valid macro-F1 감소량)
       c. Mutual Information (외부 train 전체)
     + Boruta 식 shadow 검정: 모든 피처를 행 순서만 섞은 그림자 피처를 같이 넣고,
       실제 피처 SHAP > 그림자 SHAP 기준값 인 내부 fold 수(0~3) = hits
  2) 통합 점수 = w_shap·rank(SHAP) + w_perm·rank(perm) + w_mi·rank(MI)  (작을수록 상위)
  3) 중복 제거 : 순위 순서대로 훑으며 이미 뽑힌 피처와 Spearman |ρ| ≥ 0.95 면 건너뜀
  4) k 스윕 : 외부 train 랭킹 상위 k 로 학습 → 외부 valid macro-F1.
     k* = "최고점 − 1 표준오차" 안에 드는 가장 작은 k  (1-SE rule, Breiman et al. 1984)

랭킹 조건 플래그 (아래 FLAGS / 명령행 인자로 켜고 끔)
  [1] shap_on_valid : SHAP·그림자 비교를 내부 valid 행에서 계산 (False = 학습 행, 기존)
  [4] class_norm    : "none"  = 클래스별 |SHAP| 단순 합 (기존)
                      "mean"  = 클래스마다 합이 1이 되게 정규화한 뒤 평균 (macro-F1 목표와 일치)
                      "max"   = 정규화 후 클래스 최댓값 (한 클래스에서라도 중요하면 상위)
  [2] shadow_pct    : 그림자 기준 백분위 (100 = 그림자 최댓값, 기존 / 95 = 완화)
      hits_mode     : "none"   = hits 는 보고용만 (기존)
                      "sort"   = hits 많은 순 → score 순으로 정렬
                      "filter" = hits < min_hits 인 피처를 순위에서 제외
                                 (남는 비중복 피처가 min(K_GRID) 미만이면 그 fold 는 제외 취소)
      min_hits      : filter 기준 (내부 3-fold 중 그림자를 이긴 횟수)
  [5] w_shap / w_perm / w_mi : 통합 점수 가중치
      stability     : 0  = 내부 fold 평균값의 순위로 점수 (기존)
                      >0 = 내부 fold 마다 통합 점수를 따로 매겨
                           score = 평균 + stability × 표준편차  (fold 마다 순위가 흔들리는 피처에 벌점)

  python rank_features.py                       # 추천 설정 (FLAGS)
  python rank_features.py --legacy              # 수정 전과 동일한 결과
  python rank_features.py --class-norm max --hits-mode sort --tag v2
  python rank_features.py --weights 0.5 0.5 0 --stability 1.0 --tag w55

출력 (results/, --tag 를 주면 파일명 끝에 _<tag>)
  feature_ranking_A.csv / _B.csv : 전체 fold 평균 랭킹 (A=위치 제외, B=위치 포함)
  k_sweep.csv, k_sweep.png       : k 별 macro-F1 평균±SE
  fold_rankings.json             : fold 별 순위 목록 + k*  (train_classifiers.py 가 그대로 사용)
  rank_flags.json                : 이번 실행에 쓴 플래그
  ※ train_classifiers.py 는 tag 없는 fold_rankings.json 을 읽으므로,
    실험용 실행은 --tag 를 붙여 기존 결과를 덮어쓰지 않게 할 것.
"""
import argparse
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
N_INNER = 3

# ── 랭킹 조건 플래그 ─────────────────────────────────────────────
# 기본값 = 추천 설정. 수치(95, 0.45 등)는 출발점일 뿐이니 k_sweep 결과로 비교해서 고를 것.
FLAGS = dict(
    shap_on_valid=True,             # [1]
    class_norm="mean",              # [4] "none" | "mean" | "max"
    shadow_pct=95.0,                # [2]
    hits_mode="sort",               # [2] "none" | "sort" | "filter"
    min_hits=2,                     # [2]
    w_shap=0.45, w_perm=0.45, w_mi=0.10,  # [5]
    stability=0.5,                  # [5]
    tag="",
)
# 수정 전 코드와 똑같이 동작하는 설정 (--legacy)
LEGACY = dict(
    shap_on_valid=False, class_norm="none", shadow_pct=100.0, hits_mode="none", min_hits=2,
    w_shap=0.4, w_perm=0.4, w_mi=0.2, stability=0.0, tag="",
)


def lgbm(**kw):
    return lgb.LGBMClassifier(**(RANK_PARAMS | kw), random_state=C.RANDOM_STATE)


def class_importance(contrib, n_rows, mode):
    """pred_contrib 결과 → 피처별 중요도 (실제+그림자 열 모두)
    contrib: (n_rows, K*(p+1)) → (K, p) 로 바꿔 클래스별 mean|SHAP|"""
    K = len(C.CLASSES)
    c = np.abs(contrib.reshape(n_rows, K, -1)[:, :, :-1]).mean(0)  # (K, p), 마지막 열 = bias 제외
    if mode == "none":  # 기존: 클래스 단순 합 → 다수 클래스가 지배
        return c.sum(0)
    c = c / np.maximum(c.sum(1, keepdims=True), 1e-12)  # 클래스마다 합 = 1
    if mode == "max":
        return c.max(0)
    if mode == "mean":
        return c.mean(0)
    raise ValueError(f"지원하지 않는 class_norm 값입니다: {mode}")


def _rank_desc(v):
    return pd.Series(v).rank(ascending=False).to_numpy()


def rank_on_train(X, y, rng, cfg):
    """외부 train(X: DataFrame) → 순위 DataFrame"""
    cols = list(X.columns)
    n = len(cols)
    if n == 0:
        raise ValueError("랭킹할 피처가 없습니다. 입력 피처와 결측·상수 필터를 확인하세요.")
    Xn = X.to_numpy(np.float32)
    shap_f, perm_f, hits = [], [], np.zeros(n)
    inner = V.StratifiedGroupKFold(N_INNER, shuffle=True, random_state=C.RANDOM_STATE).split(Xn, y, np.arange(len(y)))
    for tr, va in inner:
        Xt = Xn[tr]
        shadow = np.column_stack([rng.permutation(Xt[:, j]) for j in range(n)])  # 열마다 행 순서만 섞음
        m = lgbm().fit(np.hstack([Xt, shadow]), y[tr])

        # [1] SHAP 을 계산할 행: 내부 valid (그림자도 valid 안에서 새로 섞음) 또는 학습 행(기존)
        if cfg["shap_on_valid"]:
            Xva = Xn[va]
            Xe = np.hstack([Xva, np.column_stack([rng.permutation(Xva[:, j]) for j in range(n)])])
        else:
            Xe = np.hstack([Xt, shadow])
        sub = rng.choice(len(Xe), min(2000, len(Xe)), replace=False)
        contrib = m.booster_.predict(Xe[sub], pred_contrib=True)
        imp = class_importance(contrib, len(sub), cfg["class_norm"])  # [4]
        real, sh = imp[:n], imp[n:]
        shap_f.append(real)
        hits += real > np.percentile(sh, cfg["shadow_pct"])  # [2] 100 = max

        # 순열 중요도: 그림자 없는 모델, 이번 내부 fold SHAP 상위 120개만 (나머지는 사실상 0)
        m2 = lgbm().fit(Xt, y[tr])
        Xv = Xn[va].copy()
        base = V.macro_f1(y[va], m2.booster_.predict(Xv).argmax(1))
        pf = np.zeros(n)
        for j in np.argsort(-real)[:120]:
            keep = Xv[:, j].copy()
            Xv[:, j] = rng.permutation(keep)
            pf[j] = base - V.macro_f1(y[va], m2.booster_.predict(Xv).argmax(1))
            Xv[:, j] = keep
        perm_f.append(pf)

    shap_f, perm_f = np.array(shap_f), np.array(perm_f)
    sub = rng.choice(len(y), min(3000, len(y)), replace=False)
    Xm = X.iloc[sub]
    mi = mutual_info_classif(Xm.fillna(Xm.median()).fillna(0), y[sub], random_state=C.RANDOM_STATE)
    R = pd.DataFrame({"feature": cols, "shap": shap_f.mean(0), "perm": perm_f.mean(0), "mi": mi, "hits": hits})

    # [5] 통합 점수
    ws, wp, wm = cfg["w_shap"], cfg["w_perm"], cfg["w_mi"]
    if cfg["stability"] > 0:
        rk_mi = _rank_desc(mi)
        per = np.array([ws * _rank_desc(s) + wp * _rank_desc(p) + wm * rk_mi for s, p in zip(shap_f, perm_f)])
        R["score_mean"], R["score_std"] = per.mean(0), per.std(0)
        R["score"] = R["score_mean"] + cfg["stability"] * R["score_std"]
    else:  # 기존: 내부 fold 평균값의 순위
        R["score"] = (ws * R["shap"].rank(ascending=False) + wp * R["perm"].rank(ascending=False)
                      + wm * R["mi"].rank(ascending=False))

    # [2] hits 반영 정렬
    R["hits_pass"] = R["hits"] >= cfg["min_hits"]
    if cfg["hits_mode"] == "sort":
        R = R.sort_values(["hits", "score"], ascending=[False, True])
    elif cfg["hits_mode"] == "filter":
        R = R.sort_values(["hits_pass", "score"], ascending=[False, True])  # 통과 피처가 먼저 → 대표도 통과 피처
    else:
        R = R.sort_values("score")
    R = R.reset_index(drop=True)

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

    R["excluded"] = False
    if cfg["hits_mode"] == "filter":
        R["excluded"] = ~R["hits_pass"]
        n_left = int((~R["excluded"] & R["redundant_of"].isna()).sum())
        if n_left < min(K_GRID):
            print(f"    ⚠ hits ≥ {cfg['min_hits']} 통과 비중복 피처 {n_left}개 < {min(K_GRID)} → 이 fold 는 제외 취소", flush=True)
            R["excluded"] = False
    return R


def prefilter(X):
    ok = (X.isna().mean() <= 0.6) & (X.nunique() > 1)
    return X.loc[:, ok]


def run_pool(S, y, F, pool, cfg, suffix):
    cols = V.feature_pool(S, pool)
    rng = np.random.default_rng(C.RANDOM_STATE)
    fold_rank, sweep, tables = [], [], []
    for fi, (tr, va) in enumerate(F):
        t = time.time()
        Xtr = prefilter(S.iloc[tr][cols])
        if Xtr.shape[1] == 0:
            raise ValueError(f"[{pool}] fold {fi}: 결측·상수 필터 후 사용할 피처가 없습니다.")
        R = rank_on_train(Xtr, y[tr], rng, cfg)
        R["fold"] = fi
        tables.append(R)
        order = R.loc[R["redundant_of"].isna() & ~R["excluded"], "feature"].tolist()
        fold_rank.append(order)
        candidate_ks = [k for k in K_GRID if k <= len(order)]
        if len(order) and len(order) not in K_GRID:
            candidate_ks.append(len(order))
        for k in candidate_ks:
            fs = order[:k]
            m = lgbm().fit(S.iloc[tr][fs], y[tr])
            sweep.append(dict(pool=pool, fold=fi, k=k if k in K_GRID else -1, n=k,
                              f1=V.macro_f1(y[va], m.predict(S.iloc[va][fs]))))
        print(f"  [{pool}] fold {fi}: 후보 {Xtr.shape[1]} → 비중복 {int(R['redundant_of'].isna().sum())}"
              f" → 순위 사용 {len(order)}  ({time.time() - t:.0f}s)", flush=True)

    sw = pd.DataFrame(sweep)
    if sw.empty:
        raise ValueError(f"[{pool}] 모든 fold에서 평가할 피처가 없습니다.")
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
                                 excluded_folds=("excluded", "sum"),
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
    G["signal"] = (G["boruta_hits"] >= 0.5 * N_INNER * len(F)) & (G["perm"] > 0)
    G.index.name = "feature"
    G.round(5).to_csv(C.RESULT_DIR / f"feature_ranking_{pool}{suffix}.csv", encoding="utf-8-sig")
    return dict(k_star=k_star, folds=fold_rank), agg.assign(pool=pool).reset_index()


def main(cfg):
    suffix = f"_{cfg['tag']}" if cfg["tag"] else ""
    print("랭킹 플래그:", json.dumps(cfg, ensure_ascii=False))
    S, y = V.load_ship()
    F = V.folds(S, y)
    print(f"선박 {len(S):,}척, 클래스 분포 {np.bincount(y)}")
    out, aggs = {}, []
    for pool in ["A", "B"]:
        out[pool], agg = run_pool(S, y, F, pool, cfg, suffix)
        aggs.append(agg)
    out["fold_val_idx"] = [va.tolist() for _, va in F]
    (C.RESULT_DIR / f"fold_rankings{suffix}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (C.RESULT_DIR / f"rank_flags{suffix}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    A = pd.concat(aggs)
    A.to_csv(C.RESULT_DIR / f"k_sweep{suffix}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    for pool, lab in [("A", "Model A (위치 제외)"), ("B", "Model B (위치 포함)")]:
        a = A[(A["pool"] == pool) & (A["k"] != 9999)]
        ax.errorbar(a["k"], a["mean"], yerr=a["se"], marker="o", capsize=3, label=f"{lab}  k*={out[pool]['k_star']}")
        ax.axvline(out[pool]["k_star"], ls=":", color=ax.lines[-1].get_color())
    ax.set_xscale("log"); ax.set_xlabel("상위 k 피처 수"); ax.set_ylabel("선박 macro-F1 (5-fold 평균 ± SE)")
    ax.set_title(f"k 스윕 (fold 내부 랭킹, 1-SE rule){'  [' + cfg['tag'] + ']' if cfg['tag'] else ''}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(C.RESULT_DIR / f"k_sweep{suffix}.png", bbox_inches="tight", dpi=120)
    print("✅ 랭킹 완료")


def parse_args():
    p = argparse.ArgumentParser(description="피처 랭킹 + 상위 k 선정 (랭킹 조건 플래그)")
    p.add_argument("--legacy", action="store_true", help="모든 플래그를 수정 전 동작으로 (다른 인자로 일부만 덮어쓰기 가능)")
    p.add_argument("--shap-on-valid", type=int, choices=[0, 1], help="[1] 1=내부 valid 에서 SHAP, 0=학습 행")
    p.add_argument("--class-norm", choices=["none", "mean", "max"], help="[4] 클래스별 SHAP 정규화")
    p.add_argument("--shadow-pct", type=float, help="[2] 그림자 기준 백분위 (100=max)")
    p.add_argument("--hits-mode", choices=["none", "sort", "filter"], help="[2] hits 반영 방식")
    p.add_argument("--min-hits", type=int, help="[2] filter 기준 (0~3)")
    p.add_argument("--weights", type=float, nargs=3, metavar=("SHAP", "PERM", "MI"), help="[5] 점수 가중치")
    p.add_argument("--stability", type=float, help="[5] 0=끔, >0 = 표준편차 벌점 계수")
    p.add_argument("--tag", help="출력 파일 접미사")
    a = p.parse_args()
    cfg = dict(LEGACY if a.legacy else FLAGS)
    for k in ["shap_on_valid", "class_norm", "shadow_pct", "hits_mode", "min_hits", "stability", "tag"]:
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    if a.weights:
        cfg["w_shap"], cfg["w_perm"], cfg["w_mi"] = a.weights
    cfg["shap_on_valid"] = bool(cfg["shap_on_valid"])
    if not 0 <= cfg["shadow_pct"] <= 100:
        p.error("--shadow-pct 값은 0부터 100 사이여야 합니다.")
    if not 0 <= cfg["min_hits"] <= N_INNER:
        p.error(f"--min-hits 값은 0부터 {N_INNER} 사이여야 합니다.")
    if not np.isfinite(cfg["stability"]) or cfg["stability"] < 0:
        p.error("--stability 값은 0 이상의 유한한 수여야 합니다.")
    weights = np.array([cfg["w_shap"], cfg["w_perm"], cfg["w_mi"]])
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        p.error("--weights 값은 음수가 아닌 유한한 수여야 하고, 합이 0보다 커야 합니다.")
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    V.set_font()
    main(cfg)