"""
[PAIR] 혼동쌍 표적 피처 보강 — 통발·자망이 연안복합으로 흡수되는 문제를 겨냥한 피처 선택

왜: Pool A ensemble_w+adj 혼동행렬 (정답 → 예측)
      자망 → 연안복합 0.51 · 통발 → 연안복합 0.49 · 통발 → 자망 0.28 · 연안복합 → 자망 0.19
    → 전체 4클래스 중요도(rank_features.py)로 뽑은 피처는 다수 클래스·쉬운 구분(채낚기)에 끌려간다.
      "헷갈리는 두 클래스만 놓고 봤을 때 갈라 주는 피처"를 따로 찾아 기존 피처 세트에 더한다.

rank_features.py 와 다른 점
  rank_features.py : 4클래스 전체 · 모델 중요도(SHAP/순열/MI) · 그림자 검정 · 상위 k 개수(1-SE) 선택
  pair_features.py : 혼동쌍(이진) 단위 · 계열 대표 → 안정성 선택(L1 로지스틱 반복) · 기존 세트에 하나씩 더해
                     "쌍 구분력이 실제로 오르는 것만" 채택(전진 추가) · 피처 레지스트리의 카테고리·가설 정보 사용

절차 (외부 5-fold 각 train 안에서만 → 선정 누수 없음, fold 는 fold_rankings.json 과 동일)
  0) 후보   : 레지스트리 L3_ship 피처 ∩ 데이터 열. Pool A 는 is_location=False, 결측 60% 초과·상수 제거
  1) 기준 세트: 기존 랭킹 상위 base_k 개 (Pool A k*=251 은 너무 커서 기본 60 으로 제한)
  2) 쌍 가중치: 기준 세트 LightGBM 의 내부 3-fold OOF 혼동행렬에서 계산
                w(a,b) ∝ P(a→b) + P(b→a)   (train 안에서 계산, 자주 헷갈리는 쌍일수록 크게)
  3) 계열 대표: 같은 base_feature 의 평균/중앙값/q50/q90 중 하나(수준), 표준편차 하나(변동)만 남김
                (쌍별 AUC 를 가중합한 점수 기준) → 이어서 Spearman |ρ| ≥ 0.90 제거 → 공선성 구조적으로 차단
  4) 안정성 선택: 쌍마다 train 선박의 절반을 50번 뽑아 L1 로지스틱 → 선택 빈도 π, 계수 부호(어느 클래스가 큰가)
                  (Meinshausen & Bühlmann 2010). π ≥ 0.6 인 피처만 후보
  5) 전진 추가: 후보를 (쌍 가중치 × π × 도메인 prior) 순으로 하나씩 기준 세트에 더해 보고,
                내부 3-fold 의 "쌍 구분 점수 J = Σ w·pairF1" 가 δ 이상 오르고 4클래스 macro-F1 이
                guard 이상 떨어지지 않을 때만 채택 (최대 max_add 개)
  6) 외부 평가: 기준 세트 vs 기준+보강 세트를 같은 LightGBM 으로 외부 valid 에서 비교
                (macro-F1, 클래스 F1, 쌍별 F1, 자망→연안복합 등 혼동률)

도메인 prior (후보 순서만 바꿈, 채택 여부는 데이터로 결정, --prior 1 이면 끔)
  통발 vs 자망     : 재방문·어장반복(A) · 체류점 · 경로형태·직선투망(B) · 속도 구조(C)
                     (통발 = 같은 어장 반복 방문·부표마다 정지, 자망 = 저속 직선 투망 후 대기)
  자망 vs 연안복합 : 경로형태·직선투망(B) · 시간 사용·항차 · 하루 리듬·시각(E) · 일 유형·이질성(G)
  통발 vs 연안복합 : 재방문·어장반복(A) · 체류점 · 일 유형·이질성(G) · 계절(H)

출력 (results/, 접미사 _<tag>)
  pairsel_scores_<pool>.csv : 피처별 쌍마다 안정성 π · AUC · 방향 · 채택 fold 수 + 레지스트리 정보
  pairsel_eval.csv          : 기준 vs 보강 외부 평가
  pairsel_added.json        : fold 별 채택 피처와 내부 개선량
  fold_rankings_<tag>.json + rank_flags_<tag>.json : 보강 세트 (train_classifiers_v2.py --rank-tag <tag> 로 바로 사용)

실행: python pair_features.py --tag pairboost        (레지스트리 기본 경로: results/SeD_feature_registry_code.xlsx)
      python pair_features.py --pools A --base-k 40 --max-add 20
      python train_classifiers_v2.py --rank-tag pairboost          (보강 세트로 분류 실험)
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

import cv_common as V
import fe_config as C

warnings.filterwarnings("ignore")
NC = len(C.CLASSES)
CI = {c: i for i, c in enumerate(C.CLASSES)}
PAIRS = [("통발", "자망"), ("자망", "연안복합"), ("통발", "연안복합")]
PRIOR_CATS = {
    ("통발", "자망"): ["재방문·어장반복(A)", "체류점", "경로형태·직선투망(B)", "속도 구조(C)"],
    ("자망", "연안복합"): ["경로형태·직선투망(B)", "시간 사용·항차", "하루 리듬·시각(E)", "일 유형·이질성(G)"],
    ("통발", "연안복합"): ["재방문·어장반복(A)", "체류점", "일 유형·이질성(G)", "계절(H)"],
}
EVAL_LGB = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=30, colsample_bytree=0.6,
                subsample=0.8, subsample_freq=1, reg_lambda=5.0, max_bin=63, verbose=-1)
_SK18 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
L1_KW = dict(l1_ratio=1.0) if _SK18 else dict(penalty="l1")        # LogisticRegression 용
L1_CV_KW = dict(l1_ratios=[1.0]) if _SK18 else dict(penalty="l1")  # LogisticRegressionCV 용


def pname(p):
    return f"{p[0]}-{p[1]}"


def sqrt_w(y):
    cw = np.sqrt(len(y) / (NC * np.maximum(np.bincount(y, minlength=NC), 1)))
    w = cw[y]
    return w / w.mean()


# ─────────────────────────────────────────────
# 지표
# ─────────────────────────────────────────────
def pair_f1(y, P, a, b):
    """정답이 a 또는 b 인 선박만, 두 클래스 확률 중 큰 쪽으로 판정한 이진 macro-F1"""
    m = np.isin(y, [a, b])
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return np.nan
    return f1_score(y[m] == b, P[m][:, [a, b]].argmax(1) == 1, average="macro")


def confusion_rates(y, pred):
    cm = np.zeros((NC, NC))
    np.add.at(cm, (y, pred), 1)
    return cm / np.maximum(cm.sum(1, keepdims=True), 1)


def pair_objective(y, P, weights):
    return sum(w * pair_f1(y, P, CI[a], CI[b]) for (a, b), w in weights.items())


# ─────────────────────────────────────────────
# 레지스트리, 후보
# ─────────────────────────────────────────────
def load_registry(path):
    R = pd.read_excel(path, sheet_name="all_features")
    R = R[R["level"] == "L3_ship"].copy()
    R["is_location"] = R["is_location"].astype(bool)
    spread = R["build_method"].astype(str).str.contains("표준편차") | R["feature_name"].str.endswith("_std")
    R["family"] = R["base_feature"].fillna(R["feature_name"]).astype(str) + np.where(spread, "|변동", "|수준")
    return R.set_index("feature_name")


def candidates(reg, S, pool, Xtr):
    pool_cols = set(V.feature_pool(S, pool))
    c = [f for f in reg.index if f in S.columns and f in pool_cols]
    if pool == "A":
        c = [f for f in c if not reg.at[f, "is_location"]]
    X = Xtr[c]
    ok = (X.isna().mean() <= 0.6) & (X.nunique() > 1)
    return [f for f in c if ok[f]]


# ─────────────────────────────────────────────
# 단변량 AUC, 계열 대표, 안정성 선택
# ─────────────────────────────────────────────
def auc_matrix(X, yb):
    """열마다 AUC (결측 = 중앙값). >0.5 면 yb=1 쪽이 큼"""
    Z = X.fillna(X.median()).to_numpy(float)
    r = np.apply_along_axis(rankdata, 0, Z)
    n1, n0 = yb.sum(), (1 - yb).sum()
    return (r[yb == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def family_reps(X, y, reg, weights, corr_thr):
    """계열(같은 base_feature 의 수준/변동)마다 대표 1개 → Spearman |ρ| ≥ corr_thr 제거"""
    sc = pd.Series(0.0, index=X.columns)
    for (a, b), w in weights.items():
        m = np.isin(y, [CI[a], CI[b]])
        sc += w * np.abs(auc_matrix(X[m], (y[m] == CI[b]).astype(int)) - 0.5)
    fam = reg.loc[X.columns, "family"]
    reps = sc.groupby(fam).idxmax().tolist()
    reps = sorted(reps, key=lambda f: -sc[f])
    rho = X[reps].rank().corr().abs().fillna(0).to_numpy()
    keep = []
    for j in range(len(reps)):
        if all(rho[j, i] < corr_thr for i in keep):
            keep.append(j)
    return [reps[j] for j in keep], sc


def stability(X, yb, n_boot, rng, c_scale=1.0):
    """절반 부분표본 × n_boot 번 L1 로지스틱 → 선택 빈도, 평균 부호
    c_scale < 1 이면 CV 로 고른 C 보다 규제를 세게 → 더 적은 피처만 살아남아 π 가 변별력을 가짐"""
    imp = SimpleImputer(strategy="median").fit(X)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(200, len(X)), random_state=0)
    Z = qt.fit_transform(imp.transform(X))
    cv = LogisticRegressionCV(Cs=np.logspace(-2.5, 0.5, 8), solver="liblinear", scoring="roc_auc",
                              cv=StratifiedKFold(3, shuffle=True, random_state=0), class_weight="balanced",
                              max_iter=3000, **L1_CV_KW).fit(Z, yb)
    Cbest = float(np.atleast_1d(cv.C_)[0]) * c_scale
    freq, sign = np.zeros(Z.shape[1]), np.zeros(Z.shape[1])
    i1, i0 = np.flatnonzero(yb == 1), np.flatnonzero(yb == 0)
    for _ in range(n_boot):
        idx = np.r_[rng.choice(i1, len(i1) // 2, replace=False), rng.choice(i0, len(i0) // 2, replace=False)]
        m = LogisticRegression(C=Cbest, solver="liblinear", class_weight="balanced", max_iter=3000,
                               **L1_KW).fit(Z[idx], yb[idx])
        nz = m.coef_[0] != 0
        freq += nz
        sign += np.sign(m.coef_[0])
    return freq / n_boot, sign / n_boot, Cbest


# ─────────────────────────────────────────────
# 내부 평가 (LightGBM, 내부 3-fold OOF)
# ─────────────────────────────────────────────
def inner_oof(X, y, seed):
    P = np.zeros((len(y), NC))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(**EVAL_LGB, random_state=seed, n_jobs=C.N_JOBS)
        m.fit(X.iloc[tr], y[tr], sample_weight=sqrt_w(y[tr]))
        P[va] = m.predict_proba(X.iloc[va])
    return P


def fit_predict(Xtr, ytr, Xva, seed):
    m = lgb.LGBMClassifier(**EVAL_LGB, random_state=seed, n_jobs=C.N_JOBS)
    return m.fit(Xtr, ytr, sample_weight=sqrt_w(ytr)).predict_proba(Xva)


def pair_weights_from(y, P):
    cm = confusion_rates(y, P.argmax(1))
    raw = {p: cm[CI[p[0]], CI[p[1]]] + cm[CI[p[1]], CI[p[0]]] for p in PAIRS}
    s = sum(raw.values()) or 1.0
    return {p: v / s for p, v in raw.items()}


# ─────────────────────────────────────────────
# 랭킹 파일 (fold 분할 + 기준 세트)
# ─────────────────────────────────────────────
def load_base_ranking(a, S, y):
    sfx = f"_{a.rank_tag}" if a.rank_tag else ""
    rk = C.RESULT_DIR / f"fold_rankings{sfx}.json"
    fl = C.RESULT_DIR / f"rank_flags{sfx}.json"
    if not rk.exists():
        raise SystemExit(f"❌ 기준 랭킹 파일이 없습니다: {rk}  → rank_features.py 를 먼저 실행하세요.")
    R = json.loads(rk.read_text(encoding="utf-8"))
    F = V.folds(S, y)
    if len(F) != len(R["fold_val_idx"]) or any(set(va.tolist()) != set(v) for (_, va), v in zip(F, R["fold_val_idx"])):
        raise SystemExit("❌ 랭킹 파일의 fold 분할이 현재 데이터와 다릅니다. 랭킹을 다시 만드세요.")
    flags = json.loads(fl.read_text(encoding="utf-8")) if fl.exists() else None
    print(f"기준 랭킹: {rk}  (rank_flags {'있음' if flags else '없음'})")
    return R, flags


# ─────────────────────────────────────────────
def run_pool(a, S, y, reg, R, pool, rng):
    val_idx = [np.array(v) for v in R["fold_val_idx"]]
    k = min(a.base_k, R[pool]["k_star"])
    rows, added_all, fold_lists = [], [], []
    P_base, P_boost = np.zeros((len(y), NC)), np.zeros((len(y), NC))
    for f, va in enumerate(val_idx):
        t = time.time()
        tr = np.setdiff1d(np.arange(len(y)), va)
        Str = S.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        cand = candidates(reg, S, pool, Str)
        base = [c for c in R[pool]["folds"][f][:k] if c in Str.columns]

        # 2) 쌍 가중치 (기준 세트 내부 OOF 혼동행렬)
        Pin = inner_oof(Str[base], ytr, seed=f)
        W = pair_weights_from(ytr, Pin)

        # 3) 계열 대표 + 상관 제거 (기준 세트에 이미 있는 피처는 후보에서 제외)
        pool_c = [c for c in cand if c not in base]
        reps, uni = family_reps(Str[pool_c], ytr, reg, W, a.corr_thr)
        # 기준 세트와도 |ρ| ≥ corr_thr 이면 제외 (같은 정보 반복 방지)
        if base and reps:
            rb = pd.concat([Str[base], Str[reps]], axis=1).rank().corr().abs().fillna(0)
            reps = [c for c in reps if rb.loc[c, base].max() < a.corr_thr]

        # 4) 쌍별 안정성 선택
        stab = {}
        for p in PAIRS:
            m = np.isin(ytr, [CI[p[0]], CI[p[1]]])
            yb = (ytr[m] == CI[p[0]]).astype(int)  # 1 = 쌍의 앞 클래스 (통발 / 자망)
            fr, sg, Cb = stability(Str.loc[m, reps], yb, a.n_boot, rng)
            au = auc_matrix(Str.loc[m, reps], yb)
            stab[p] = pd.DataFrame({"pi": fr, "sign": sg, "auc": au}, index=reps)
            for feat in reps:
                rows.append(dict(pool=pool, fold=f, feature=feat, pair=pname(p), pair_w=W[p],
                                 pi=stab[p].at[feat, "pi"], sign=stab[p].at[feat, "sign"],
                                 auc=stab[p].at[feat, "auc"]))

        # 5) 전진 추가: 후보 순서 = max_p (w_p × π_p × prior)
        prio = {}
        for feat in reps:
            best = 0.0
            for p in PAIRS:
                pi = stab[p].at[feat, "pi"]
                if pi < a.pi_thr:
                    continue
                pr = a.prior if reg.at[feat, "category"] in PRIOR_CATS[p] else 1.0
                best = max(best, W[p] * pi * pr)
            if best > 0:
                prio[feat] = best
        order = sorted(prio, key=lambda c: -prio[c])[:a.max_try]

        cur = list(base)
        J0, M0 = pair_objective(ytr, Pin, W), V.macro_f1(ytr, Pin.argmax(1))
        added = []
        for feat in order:
            if len(added) >= a.max_add:
                break
            Pc = inner_oof(Str[cur + [feat]], ytr, seed=f)
            J, M = pair_objective(ytr, Pc, W), V.macro_f1(ytr, Pc.argmax(1))
            if J - J0 >= a.delta and M - M0 >= -a.guard:
                added.append(dict(feature=feat, dJ=round(J - J0, 4), dMacro=round(M - M0, 4),
                                  category=str(reg.at[feat, "category"])))
                cur.append(feat)
                J0, M0 = J, M
        added_all.append(dict(fold=f, pair_weights={pname(p): round(w, 3) for p, w in W.items()},
                              n_candidates=len(cand), n_reps=len(reps), n_tried=len(order), added=added))
        fold_lists.append(cur)

        # 6) 외부 평가
        Sva = S.iloc[va]
        P_base[va] = fit_predict(Str[base], ytr, Sva[base], seed=f)
        P_boost[va] = fit_predict(Str[cur], ytr, Sva[cur], seed=f)
        print(f"  [{pool}] fold {f}: 후보 {len(cand)} → 계열대표 {len(reps)} → 시도 {len(order)} → 채택 {len(added)}  "
              f"쌍가중 {', '.join(f'{pname(p)} {w:.2f}' for p, w in W.items())}  "
              f"valid F1 {V.macro_f1(y[va], P_base[va].argmax(1)):.3f} → {V.macro_f1(y[va], P_boost[va].argmax(1)):.3f}"
              f"  ({time.time() - t:.0f}s)", flush=True)
    return pd.DataFrame(rows), added_all, fold_lists, P_base, P_boost, k


def eval_rows(pool, y, P, name, fold_id):
    pred = P.argmax(1)
    cm = confusion_rates(y, pred)
    per_fold = [V.macro_f1(y[fold_id == f], pred[fold_id == f]) for f in np.unique(fold_id)]
    cls = f1_score(y, pred, average=None, labels=list(range(NC)))
    r = dict(pool=pool, set=name, macro_f1=V.macro_f1(y, pred), f1_fold_std=np.std(per_fold, ddof=1),
             **{f"f1_{c}": v for c, v in zip(C.CLASSES, cls)},
             **{f"pairF1_{pname(p)}": pair_f1(y, P, CI[p[0]], CI[p[1]]) for p in PAIRS})
    for s, d in [("자망", "연안복합"), ("통발", "연안복합"), ("통발", "자망"), ("연안복합", "자망")]:
        r[f"{s}→{d}"] = cm[CI[s], CI[d]]
    return r


def summarize(reg, D, added_all, n_folds):
    if D.empty:
        return pd.DataFrame()
    g = D.groupby(["feature", "pair"]).agg(pi=("pi", "mean"), auc=("auc", "mean"), sign=("sign", "mean"),
                                           folds_seen=("fold", "nunique")).reset_index()
    out = g.pivot(index="feature", columns="pair", values=["pi", "auc", "sign"])
    out.columns = [f"{m}_{p}" for m, p in out.columns]
    for p in PAIRS:
        s = out.get(f"sign_{pname(p)}")
        if s is not None:
            out[f"dir_{pname(p)}"] = np.where(s > 0.2, f"{p[0]}↑", np.where(s < -0.2, f"{p[1]}↑", "-"))
    cnt = pd.Series([x["feature"] for fa in added_all for x in fa["added"]]).value_counts()
    out["added_folds"] = cnt.reindex(out.index).fillna(0).astype(int)
    out["added_ratio"] = out["added_folds"] / n_folds
    pi_cols = [c for c in out.columns if c.startswith("pi_")]
    out["pi_max"] = out[pi_cols].max(1)
    info = reg.reindex(out.index)[["category", "subcategory", "hypothesis", "description", "build_method"]]
    out = info.join(out).sort_values(["added_folds", "pi_max"], ascending=False)
    out.index.name = "feature"
    return out


def main(a):
    S, y = V.load_ship()
    reg = load_registry(a.registry)
    R, base_flags = load_base_ranking(a, S, y)
    print(f"선박 {len(S):,}척 · 레지스트리 L3 {len(reg)}개 (데이터에 있는 것 {sum(f in S.columns for f in reg.index)}개)")
    rng = np.random.default_rng(C.RANDOM_STATE)
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(R["fold_val_idx"]):
        fold_id[np.array(v)] = f
    sfx = f"_{a.tag}" if a.tag else ""
    evals, added_json, out_rank = [], {}, {"fold_val_idx": R["fold_val_idx"]}
    for pool in a.pools:
        print(f"\n━━ Pool {pool} ━━")
        D, added_all, lists, Pb, Pp, k = run_pool(a, S, y, reg, R, pool, rng)
        evals += [eval_rows(pool, y, Pb, f"기준 (top-{k})", fold_id), eval_rows(pool, y, Pp, "기준+쌍보강", fold_id)]
        summarize(reg, D, added_all, len(lists)).round(4).to_csv(C.RESULT_DIR / f"pairsel_scores_{pool}{sfx}.csv",
                                                                 encoding="utf-8-sig")
        added_json[pool] = dict(base_k=k, folds=added_all)
        out_rank[pool] = dict(k_star=max(len(x) for x in lists), folds=lists)
    E = pd.DataFrame(evals)
    E.round(4).to_csv(C.RESULT_DIR / f"pairsel_eval{sfx}.csv", index=False, encoding="utf-8-sig")
    (C.RESULT_DIR / f"pairsel_added{sfx}.json").write_text(json.dumps(added_json, ensure_ascii=False, indent=2,
                                                                      default=float), encoding="utf-8")
    # 분류 실험용 랭킹 파일 (fold 마다 기준+보강 목록, k_star = 최장 길이 → [:k] 로 자르면 fold 별 목록 전체)
    if a.tag:
        flags = dict(mode="pair_features", base_rank_tag=a.rank_tag, base_flags=base_flags, base_k=a.base_k,
                     pi_thr=a.pi_thr, n_boot=a.n_boot, delta=a.delta, guard=a.guard, max_add=a.max_add,
                     corr_thr=a.corr_thr, prior=a.prior, tag=a.tag)
        (C.RESULT_DIR / f"fold_rankings{sfx}.json").write_text(json.dumps(out_rank, ensure_ascii=False), encoding="utf-8")
        (C.RESULT_DIR / f"rank_flags{sfx}.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2,
                                                                       default=str), encoding="utf-8")
        print(f"\n분류 실험용 랭킹 저장: fold_rankings{sfx}.json  → python train_classifiers_v2.py --rank-tag {a.tag}")
    show = ["pool", "set", "macro_f1", "f1_fold_std", "f1_자망", "f1_통발"] + \
           [f"pairF1_{pname(p)}" for p in PAIRS] + ["자망→연안복합", "통발→연안복합", "통발→자망"]
    print("\n" + E[[c for c in show if c in E]].round(4).to_string(index=False))
    for pool in a.pools:
        cnt = pd.Series([x["feature"] for fa in added_json[pool]["folds"] for x in fa["added"]]).value_counts()
        print(f"\n[{pool}] 여러 fold 에서 채택된 피처 (fold 수):")
        print(cnt.head(15).to_string() if len(cnt) else "  (없음)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="혼동쌍 표적 피처 보강")
    ap.add_argument("--registry", default=str(Path(__file__).parent / "results" / "SeD_feature_registry_code.xlsx"),
                    help="피처 레지스트리 (기본: results/SeD_feature_registry_code.xlsx)")
    ap.add_argument("--rank-tag", default="", help="기준 랭킹(fold_rankings_<tag>.json) tag. 비우면 fold_rankings.json")
    ap.add_argument("--tag", default="pairboost", help="결과 파일 접미사 + 분류 실험용 랭킹 tag")
    ap.add_argument("--pools", nargs="*", default=["A", "B"])
    ap.add_argument("--base-k", type=int, default=60, help="기준 세트 크기 상한 (랭킹 상위 k)")
    ap.add_argument("--corr-thr", type=float, default=0.90, help="계열 대표 간·기준 세트와의 Spearman |ρ| 상한")
    ap.add_argument("--n-boot", type=int, default=50, help="안정성 선택 반복 수")
    ap.add_argument("--pi-thr", type=float, default=0.6, help="안정성 선택 빈도 기준")
    ap.add_argument("--max-try", type=int, default=30, help="전진 추가에서 시험할 최대 후보 수")
    ap.add_argument("--max-add", type=int, default=15, help="fold 당 최대 채택 수")
    ap.add_argument("--delta", type=float, default=0.003, help="채택에 필요한 쌍 구분 점수 J 최소 개선")
    ap.add_argument("--guard", type=float, default=0.005, help="허용하는 4클래스 macro-F1 최대 하락")
    ap.add_argument("--prior", type=float, default=1.15, help="도메인 prior 카테고리 가중 (1 = 끔, 순서에만 영향)")
    main(ap.parse_args())
