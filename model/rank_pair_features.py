"""
[RANK-PAIR] 쌍 안정 피처 랭킹 — pair_features.py 로 걸러진 피처 안에서 rank_features.py 방식으로 순위 + k* 선정

배경 (pair_features 결과)
  · 단변량 AUC 가 0.41~0.59 로 약함 → 피처 하나로는 통발·자망·연안복합을 거의 못 가름
  · 기존 상위 k 에 "더하기" 방식으로는 fold 당 0~3 개만 채택되고, 2개 이상 fold 에서 채택된 피처가 없음
  · 그래서 "더하기" 대신, 쌍 구분에 꾸준히 쓰이는 피처들로 후보군 자체를 다시 짜고 그 안에서 순위를 매긴다.

rank_features.py 와 같은 것 (그대로 import 해서 사용)
  · 내부 3-fold LightGBM: SHAP(클래스 정규화) · 순열 중요도 · MI · 그림자 hits → 통합 점수 (rank_on_train)
  · 중복 제거, k 스윕, 1-SE rule, fold_rankings / rank_flags 저장 형식 (train_classifiers_v2.py 와 호환)

달라진 것
  ① 후보군  : 레지스트리 L3 전체 → 계열 대표(수준 1·변동 1) → 쌍 안정성 π_max ≥ pi_thr 인 것만
              (π 는 c_scale 로 L1 규제를 세게 해서 계산 → "거의 다 0.6 이상" 문제 해소)
              + 그 fold 에서 pair_features 가 채택한 피처(pairsel_added.json)는 항상 포함
  ② 점수    : (1 − w_pair) · rank(rank_features 통합 점수) + w_pair · rank(쌍 점수)
              쌍 점수 = Σ_p  w_p · π_p · |AUC_p − 0.5|   (자주 헷갈리는 쌍 · 꾸준히 선택 · 실제로 갈라 줌)
  ③ k 스윕  : macro-F1 과 함께 쌍별 F1 · 자망→연안복합 · 통발→연안복합 혼동률을 같이 기록
              k* 기준은 --k-metric macro (기본) 또는 pair (쌍 구분 점수 J)

모든 계산은 외부 fold 의 train 안에서만 (fold = 기준 랭킹 파일과 동일) → 선정 누수 없음.
  ※ pairsel_scores_*.csv 는 5개 fold 결과를 합친 것이라 후보군으로 쓰면 누수가 생겨 사용하지 않음.

필요 파일 (같은 폴더): rank_features.py (플래그 버전), pair_features.py, cv_common.py, fe_config.py
출력 (results/, 접미사 _<tag>)
  feature_ranking_<pool>.csv · k_sweep.csv · k_sweep.png · fold_rankings.json · rank_flags.json
  rank_pair_info.json (fold 별 후보 수 · 쌍 가중치)

실행: python rank_pair_features.py --added results/pairsel_added_pairboost.json --tag pairrank
      python rank_pair_features.py --w-pair 0.5 --k-metric pair --tag pairrank_j
      python train_classifiers_v2.py --rank-tag pairrank
"""
import argparse
from pathlib import Path
import json
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

import cv_common as V
import fe_config as C
import pair_features as PF
import rank_features as RF

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
NC = len(C.CLASSES)
PAIRS = PF.PAIRS
CI = PF.CI


def fold_pool(a, Str, ytr, reg, S, pool, base, forced, rng):
    """fold train 안에서: 후보 → 계열 대표 → 쌍 안정성 → 후보군 + 쌍 점수"""
    cand = PF.candidates(reg, S, pool, Str)
    W = PF.pair_weights_from(ytr, PF.inner_oof(Str[base], ytr, seed=a._fold))
    reps, _ = PF.family_reps(Str[cand], ytr, reg, W, a.corr_thr)
    pi = pd.DataFrame(index=reps, dtype=float)
    au = pd.DataFrame(index=reps, dtype=float)
    for p in PAIRS:
        m = np.isin(ytr, [CI[p[0]], CI[p[1]]])
        yb = (ytr[m] == CI[p[0]]).astype(int)
        fr, _, _ = PF.stability(Str.loc[m, reps], yb, a.n_boot, rng, c_scale=a.c_scale)
        pi[PF.pname(p)] = fr
        au[PF.pname(p)] = PF.auc_matrix(Str.loc[m, reps], yb)
    pair_score = sum(W[p] * pi[PF.pname(p)] * (au[PF.pname(p)] - 0.5).abs() for p in PAIRS)
    keep = pi.max(1) >= a.pi_thr
    pool_feats = list(dict.fromkeys([f for f in forced if f in Str.columns] + list(pi.index[keep])))
    info = pd.DataFrame({"pair_score": pair_score, "pi_max": pi.max(1),
                         **{f"pi_{c}": pi[c] for c in pi}, **{f"auc_{c}": au[c] for c in au}})
    return pool_feats, info, W, len(cand), len(reps)


def rank_fold(a, X, y, info, cfg, rng):
    """rank_features 의 rank_on_train → 쌍 점수와 결합 → 재정렬 → 중복 제거"""
    R = RF.rank_on_train(X, y, rng, cfg).drop(columns=["redundant_of", "excluded"])
    R = R.merge(info, left_on="feature", right_index=True, how="left")
    R["pair_score"] = R["pair_score"].fillna(0)  # 강제 포함 피처 중 계열 대표가 아니었던 것
    R["rf_rank"] = R["score"].rank()
    R["pair_rank"] = R["pair_score"].rank(ascending=False)
    R["final_score"] = (1 - a.w_pair) * R["rf_rank"] + a.w_pair * R["pair_rank"]
    if cfg["hits_mode"] == "sort":
        R = R.sort_values(["hits", "final_score"], ascending=[False, True])
    else:
        R = R.sort_values("final_score")
    R = R.reset_index(drop=True)
    rho = X[R["feature"]].rank().corr().abs().fillna(0).to_numpy()
    keep, rep = [], {}
    for i, f in enumerate(R["feature"]):
        dup = [j for j in keep if rho[i, j] >= RF.CORR_THR]
        if dup:
            rep[f] = R["feature"][dup[0]]
        else:
            keep.append(i)
    R["redundant_of"] = R["feature"].map(rep)
    return R


def sweep_metrics(y, P):
    pred = P.argmax(1)
    cm = PF.confusion_rates(y, pred)
    return dict(f1=V.macro_f1(y, pred),
                **{f"pairF1_{PF.pname(p)}": PF.pair_f1(y, P, CI[p[0]], CI[p[1]]) for p in PAIRS},
                **{"자망→연안복합": cm[CI["자망"], CI["연안복합"]], "통발→연안복합": cm[CI["통발"], CI["연안복합"]],
                   "통발→자망": cm[CI["통발"], CI["자망"]]})


def run_pool(a, S, y, reg, R0, added, pool, cfg, suffix):
    val_idx = [np.array(v) for v in R0["fold_val_idx"]]
    rng = np.random.default_rng(C.RANDOM_STATE)
    base_k = min(a.base_k, R0[pool]["k_star"])
    fold_rank, sweep, tables, finfo = [], [], [], []
    for f, va in enumerate(val_idx):
        t = time.time()
        a._fold = f
        tr = np.setdiff1d(np.arange(len(y)), va)
        Str = S.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        base = [c for c in R0[pool]["folds"][f][:base_k] if c in Str.columns]
        forced = [x["feature"] for x in added.get(pool, {}).get("folds", [{}] * len(val_idx))[f].get("added", [])] \
            if added else []
        feats, info, W, n_cand, n_reps = fold_pool(a, Str, ytr, reg, S, pool, base, forced, rng)
        R = rank_fold(a, Str[feats], ytr, info, cfg, rng)
        R["fold"] = f
        tables.append(R)
        order = R.loc[R["redundant_of"].isna(), "feature"].tolist()
        fold_rank.append(order)
        Wp = {p: W[p] for p in PAIRS}
        ks = [k for k in RF.K_GRID if k <= len(order)] + ([len(order)] if len(order) not in RF.K_GRID else [])
        for k in ks:
            fs = order[:k]
            P = RF.lgbm().fit(S.iloc[tr][fs], y[tr]).predict_proba(S.iloc[va][fs])
            sm = sweep_metrics(y[va], P)
            sweep.append(dict(pool=pool, fold=f, k=k if k in RF.K_GRID else -1, n=k, J=PF.pair_objective(y[va], P, Wp),
                              **sm))
        finfo.append(dict(fold=f, n_candidates=n_cand, n_reps=n_reps, n_pool=len(feats), n_forced=len(forced),
                          n_ranked=len(order), pair_weights={PF.pname(p): round(w, 3) for p, w in W.items()}))
        print(f"  [{pool}] fold {f}: 후보 {n_cand} → 계열대표 {n_reps} → 쌍안정(π≥{a.pi_thr}) {len(feats)} "
              f"(강제 {len(forced)}) → 비중복 {len(order)}  ({time.time() - t:.0f}s)", flush=True)

    sw = pd.DataFrame(sweep)
    sw.loc[sw["k"] == -1, "k"] = 9999
    metric = "f1" if a.k_metric == "macro" else "J"
    agg = sw.groupby("k").agg(**{c: (c, "mean") for c in sw.columns if c not in ("pool", "fold", "k", "n")},
                              se=(metric, lambda s: s.std() / np.sqrt(len(s))))
    best = agg[metric].idxmax()
    k_star = int(agg.index[agg[metric] >= agg.loc[best, metric] - agg.loc[best, "se"]].min())
    if k_star == 9999:
        k_star = int(sw.loc[sw["k"] == 9999, "n"].min())
    print(f"  [{pool}] k* = {k_star}  (기준 {a.k_metric}, best k={best}: macro-F1 {agg.loc[best, 'f1']:.4f}, "
          f"J {agg.loc[best, 'J']:.4f}, 자망→연안 {agg.loc[best, '자망→연안복합']:.3f}, 통발→연안 {agg.loc[best, '통발→연안복합']:.3f})")

    T = pd.concat(tables)
    G = T.groupby("feature").agg(rf_score=("score", "mean"), pair_score=("pair_score", "mean"),
                                 final_score=("final_score", "mean"), shap=("shap", "mean"), perm=("perm", "mean"),
                                 mi=("mi", "mean"), boruta_hits=("hits", "sum"), pi_max=("pi_max", "mean"),
                                 **{c: (c, "mean") for c in T.columns if c.startswith(("pi_", "auc_")) and c != "pi_max"},
                                 in_pool_folds=("fold", "nunique"),
                                 redundant_folds=("redundant_of", lambda s: int(s.notna().sum())))
    G["topk_freq"] = pd.Series({f: sum(f in o[:k_star] for o in fold_rank) / len(fold_rank) for f in G.index})
    G = G.sort_values(["topk_freq", "final_score"], ascending=[False, True])
    G["final_rank"] = np.arange(1, len(G) + 1)
    info = reg.reindex(G.index)[["category", "subcategory", "hypothesis", "description"]]
    G = info.join(G)
    G.insert(0, "is_loc", [f in V.LOC_COLS for f in G.index])
    G.index.name = "feature"
    G.round(5).to_csv(C.RESULT_DIR / f"feature_ranking_{pool}{suffix}.csv", encoding="utf-8-sig")
    return dict(k_star=k_star, folds=fold_rank), agg.assign(pool=pool).reset_index(), finfo


def main(a):
    V.set_font()
    S, y = V.load_ship()
    reg = PF.load_registry(a.registry)
    pa = argparse.Namespace(rank_tag=a.rank_tag)
    R0, base_flags = PF.load_base_ranking(pa, S, y)
    added = json.loads(open(a.added, encoding="utf-8").read()) if a.added else None
    cfg = dict(RF.LEGACY if a.rank_legacy else RF.FLAGS)
    cfg["tag"] = a.tag
    suffix = f"_{a.tag}" if a.tag else ""
    print(f"선박 {len(S):,}척 · rank_features 설정: {'legacy' if a.rank_legacy else '추천(FLAGS)'} · "
          f"w_pair {a.w_pair} · c_scale {a.c_scale} · pi_thr {a.pi_thr} · 강제 포함 {'있음' if added else '없음'}")
    out, aggs, finfos = {}, [], {}
    for pool in a.pools:
        print(f"\n━━ Pool {pool} ━━")
        out[pool], agg, finfos[pool] = run_pool(a, S, y, reg, R0, added, pool, cfg, suffix)
        aggs.append(agg)
    out["fold_val_idx"] = R0["fold_val_idx"]
    A = pd.concat(aggs)
    A.round(5).to_csv(C.RESULT_DIR / f"k_sweep{suffix}.csv", index=False, encoding="utf-8-sig")
    flags = dict(cfg, mode="rank_pair_features", base_rank_tag=a.rank_tag, base_flags=base_flags, base_k=a.base_k,
                 w_pair=a.w_pair, c_scale=a.c_scale, pi_thr=a.pi_thr, n_boot=a.n_boot, corr_thr=a.corr_thr,
                 k_metric=a.k_metric, forced_from=a.added)
    (C.RESULT_DIR / f"fold_rankings{suffix}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (C.RESULT_DIR / f"rank_flags{suffix}.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2,
                                                                     default=str), encoding="utf-8")
    (C.RESULT_DIR / f"rank_pair_info{suffix}.json").write_text(json.dumps(finfos, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for pool in a.pools:
        s = A[(A["pool"] == pool) & (A["k"] != 9999)]
        axes[0].errorbar(s["k"], s["f1"], yerr=s["se"] if a.k_metric == "macro" else None, marker="o", capsize=3,
                         label=f"Pool {pool}  k*={out[pool]['k_star']}")
        axes[0].axvline(out[pool]["k_star"], ls=":", color=axes[0].lines[-1].get_color())
        axes[1].plot(s["k"], s["자망→연안복합"], marker="o", label=f"{pool} 자망→연안복합")
        axes[1].plot(s["k"], s["통발→연안복합"], marker="s", ls="--", label=f"{pool} 통발→연안복합")
    axes[0].set_xscale("log"); axes[0].set_xlabel("상위 k"); axes[0].set_ylabel("macro-F1 (5-fold 평균)")
    axes[0].set_title("k 스윕 (쌍 안정 피처 랭킹)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xscale("log"); axes[1].set_xlabel("상위 k"); axes[1].set_ylabel("혼동률 (낮을수록 좋음)")
    axes[1].set_title("연안복합으로 흡수되는 비율"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.savefig(C.RESULT_DIR / f"k_sweep{suffix}.png", bbox_inches="tight", dpi=120)
    print(f"\n✅ 완료 → python train_classifiers_v2.py --rank-tag {a.tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="쌍 안정 피처 랭킹")
    ap.add_argument("--registry", default=str(Path(__file__).parent / "results" / "SeD_feature_registry_code.xlsx"),
                    help="피처 레지스트리 (기본: results/SeD_feature_registry_code.xlsx)")
    ap.add_argument("--rank-tag", default="", help="기준 랭킹 tag (fold 분할 + 쌍 가중치용 기준 세트)")
    ap.add_argument("--added", default=None, help="pairsel_added_<tag>.json (fold 별 채택 피처를 강제 포함)")
    ap.add_argument("--tag", default="pairrank")
    ap.add_argument("--pools", nargs="*", default=["A", "B"])
    ap.add_argument("--base-k", type=int, default=60, help="쌍 가중치 계산용 기준 세트 크기")
    ap.add_argument("--corr-thr", type=float, default=0.90)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--c-scale", type=float, default=0.3, help="안정성 선택의 L1 규제 강도 (CV 로 고른 C × 이 값)")
    ap.add_argument("--pi-thr", type=float, default=0.7, help="후보군에 넣을 쌍 안정성 기준 (어느 한 쌍에서라도)")
    ap.add_argument("--w-pair", type=float, default=0.4, help="최종 점수에서 쌍 점수 비중 (0 = rank_features 와 동일)")
    ap.add_argument("--k-metric", choices=["macro", "pair"], default="macro", help="k* 를 고르는 기준")
    ap.add_argument("--rank-legacy", action="store_true", help="rank_features 를 LEGACY 설정으로")
    a = ap.parse_args()
    if not 0 <= a.w_pair <= 1:
        ap.error("--w-pair 는 0~1")
    main(a)
