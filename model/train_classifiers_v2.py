"""
[CLS v2] 과제 1 조업형태 분류 실험 — 과대적합·다중공선성 대응 강화판
(선박 단위, rfid_id StratifiedGroupKFold 5-fold, 랭킹 = 플래그 버전 rank_features.py)

기존 train_classifiers.py 와 무엇이 다른가
┌──────────────┬───────────────────────────────────┬──────────────────────────────────────────────────────┐
│ 영역          │ 기존                               │ v2                                                   │
├──────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────┤
│ 다중공선성     │ 랭킹 단계 |ρ|≥0.95 제거만          │ + fold train 에서 |ρ|≥0.90 추가 제거 (모든 모델 공통)    │
│              │                                   │ + 로지스틱은 VIF ≤ 10 될 때까지 제거 + elastic-net      │
│ base (LR)    │ L2, C=0.5 고정                     │ elastic-net, C·l1_ratio 를 내부 3-fold 로 선택          │
│ lgbm_tuned   │ 트리 수까지 튜닝, macro-F1 목적,    │ early stopping 으로 트리 수 결정, 가중 log-loss 목적    │
│              │ class_weight=balanced, 단일 재학습 │ (매끄러워 튜닝 과적합↓), max_depth·min_split_gain·      │
│              │                                   │ extra_trees 추가, 규제 강한 탐색범위, Median 가지치기,  │
│              │                                   │ √balanced 가중, 최종 = 내부 3-fold 모델 평균(CV-bagging) │
│ catboost     │ 15% 홀드아웃 → 1.15배 반복 재학습  │ 내부 3-fold CV-bagging (각자 early stopping, 재학습 없음)│
│              │ depth 6, l2 3                     │ depth 5, l2 6, rsm 0.7, Bayesian bootstrap, random_str │
│ hier_hyp     │ 1·2단계 고정 트리 수, 가설 피처     │ 1·2단계 모두 CV-bagging + early stopping, √balanced,    │
│              │ 그대로 추가                        │ 2단계 피처(top-k + 가설)도 |ρ|≥0.90 제거, 규제 강화       │
│ fold 처리     │ 외부 5-fold (랭킹과 동일)          │ 외부 5-fold 는 그대로(누수 방지에 필수). 내부 분할 seed │
│              │ 내부 분할 seed 고정                │ 를 외부 fold 마다 바꿈, 전처리·제거 기준은 train 에서만 │
│ 앙상블        │ 단순 평균                          │ 단순 평균 + LOFO 가중 평균(가중 log-loss 최소화,         │
│              │                                   │ 균등 가중으로 30% 수축 → 멤버 간 상관에 덜 민감)         │
│ 클래스 보정    │ 4개 배율, 0.3~3.3배, 3회           │ 1번 클래스 고정(배율은 스케일 불변 → 중복 자유도 제거),  │
│              │                                   │ 0.5~2배, 2회, log 배율 50% 수축                         │
│ 평가지표      │ macro-F1, fold 평균·표준편차 등     │ + 95% 부트스트랩 CI, log-loss, MCC, fold 표준오차(ddof=1)│
│              │                                   │ + train macro-F1 과 과대적합 gap (train − valid)        │
└──────────────┴───────────────────────────────────┴──────────────────────────────────────────────────────┘
결과 파일은 모두 _clsv2 가 붙어 기존 결과를 덮어쓰지 않는다 (metrics_clsv2_<tag>.csv, oof_<pool>_<model>_clsv2_<tag>.npy …).

실행: python train_classifiers_v2.py --rank-tag v1                (랭킹 v1 로 CPU 기본 4개 + 앙상블)
      python train_classifiers_v2.py --rerank --rank-tag v1       (랭킹부터 새로)
      python train_classifiers_v2.py --rank-tag v1 --trials 15    (더 빠르게)
      python train_classifiers_v2.py --rank-tag v1 --corr-thr 0.85 --vif-max 5   (공선성 기준 더 엄격하게)
"""
import argparse
import inspect
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

import cv_common as V
import fe_config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
NC = len(C.CLASSES)
I_COAST, I_GILL = C.CLASSES.index("연안복합"), C.CLASSES.index("자망")
VERSION = "clsv2"  # 팀 기존 결과(metrics_v2.csv = train_classifiers.py --tag v2)와 이름이 겹치지 않게

# LightGBM 공통: 트리 수는 early stopping 이 정함 (n_estimators 는 상한)
LGB_COMMON = dict(n_estimators=3000, learning_rate=0.03, subsample=0.8, subsample_freq=1, max_bin=63, verbose=-1)
LGB_STAGE1 = LGB_COMMON | dict(num_leaves=24, max_depth=6, min_child_samples=30, colsample_bytree=0.6,
                               reg_lambda=5.0, reg_alpha=0.1, min_split_gain=0.01)
LGB_STAGE2 = LGB_COMMON | dict(num_leaves=16, max_depth=5, min_child_samples=30, colsample_bytree=0.5,
                               reg_lambda=10.0, reg_alpha=0.1, min_split_gain=0.01)
ES_ROUNDS = 100


# ─────────────────────────────────────────────
# 공통 도구: 클래스 가중, 다중공선성 제거, CV-bagging
# ─────────────────────────────────────────────
def sqrt_w(y, n):
    """√balanced 샘플 가중 (평균 1). balanced 는 소수 클래스를 과보정함 (CatBoost 실험 결과)"""
    cw = np.sqrt(len(y) / (n * np.maximum(np.bincount(y, minlength=n), 1)))
    w = cw[y]
    return w / w.mean()


def decorrelate(X, thr):
    """열 순서(=랭킹 순서)대로 훑으며, 이미 남긴 열과 Spearman |ρ| ≥ thr 이면 제거 (train 에서만 계산)"""
    cols = list(X.columns)
    rho = X.rank().corr().abs().fillna(0).to_numpy()
    keep = []
    for j in range(len(cols)):
        if all(rho[j, i] < thr for i in keep):
            keep.append(j)
    return [cols[j] for j in keep]


def vif_prune(Z, vmax):
    """VIF_j = (상관행렬 역행렬)_jj. 최댓값이 vmax 이하가 될 때까지 가장 큰 열을 제거"""
    keep = [j for j in range(Z.shape[1]) if Z[:, j].std() > 1e-8]
    v = np.array([1.0])
    while len(keep) > 1:
        v = np.diag(np.linalg.pinv(np.corrcoef(Z[:, keep], rowvar=False)))
        j = int(np.argmax(v))
        if v[j] <= vmax:
            break
        keep.pop(j)
    return keep, float(v.max())


def lgb_cv_bag(params, Xtr, ytr, Xev, n_classes, n_splits=3, seed=0):
    """내부 n_splits-fold 마다 early stopping 으로 학습 → 평가 행 확률을 평균.
    재학습 없이 모든 train 행을 쓰고, 트리 수가 정직한 검증으로 정해짐"""
    P, iters = 0.0, []
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xtr, ytr):
        m = lgb.LGBMClassifier(**params, random_state=seed, n_jobs=C.N_JOBS)
        m.fit(Xtr.iloc[tr], ytr[tr], sample_weight=sqrt_w(ytr[tr], n_classes),
              eval_set=[(Xtr.iloc[va], ytr[va])], eval_sample_weight=[sqrt_w(ytr[va], n_classes)],
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        P = P + m.predict_proba(Xev)
        iters.append(int(m.best_iteration_ or params["n_estimators"]))
    return P / n_splits, iters


# ─────────────────────────────────────────────
# 모델 정의: fit_predict(Xtr, ytr, Xev) → proba (len(Xev), 4)
#   Xev = [valid 행; train 행] (train 행은 과대적합 gap 계산용)
# ─────────────────────────────────────────────
def m_base(Xtr, ytr, Xev, vif_max=10.0, log=None, **_):
    """elastic-net 로지스틱 + VIF 제거: 선형 모델은 공선성에 가장 민감하므로 이중으로 대응"""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(200, len(Xtr)), random_state=0)
    A = qt.fit_transform(imp.transform(Xtr))
    B = qt.transform(imp.transform(Xev))
    keep, vmax = vif_prune(A, vif_max)
    cw = np.sqrt(len(ytr) / (NC * np.bincount(ytr, minlength=NC)))
    kw = dict(Cs=np.logspace(-3, 1, 6), solver="saga", l1_ratios=[0.1, 0.5, 0.9], scoring="f1_macro",
              cv=StratifiedKFold(3, shuffle=True, random_state=0), class_weight=dict(enumerate(cw)),
              max_iter=4000, n_jobs=C.N_JOBS, random_state=0)
    if "penalty" in inspect.signature(LogisticRegressionCV).parameters:
        kw["penalty"] = "elasticnet"
    m = LogisticRegressionCV(**kw).fit(A[:, keep], ytr)
    if log is not None:
        log.append(dict(n_in=A.shape[1], n_after_vif=len(keep), max_vif=round(vmax, 2),
                        C=float(np.atleast_1d(m.C_)[0]), l1_ratio=float(np.atleast_1d(m.l1_ratio_)[0])))
    return m.predict_proba(B[:, keep])


def m_rf(Xtr, ytr, Xev, **_):
    m = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(600, min_samples_leaf=3, max_features="sqrt", max_samples=0.8,
                                             class_weight="balanced_subsample", n_jobs=C.N_JOBS, random_state=0))
    return m.fit(Xtr, ytr).predict_proba(Xev)


def m_lgbm_tuned(Xtr, ytr, Xev, trials=30, log=None, seed=0, n_bag=3, **_):
    def obj(t):
        depth = t.suggest_int("max_depth", 3, 8)
        p = dict(LGB_COMMON,
                 learning_rate=t.suggest_float("learning_rate", 0.02, 0.08, log=True),
                 max_depth=depth,
                 num_leaves=min(t.suggest_int("num_leaves", 8, 64, log=True), 2 ** depth),
                 min_child_samples=t.suggest_int("min_child_samples", 15, 120, log=True),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 0.8),
                 subsample=t.suggest_float("subsample", 0.6, 0.95),
                 reg_lambda=t.suggest_float("reg_lambda", 0.1, 50, log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5, log=True),
                 min_split_gain=t.suggest_float("min_split_gain", 0.0, 0.3),
                 extra_trees=t.suggest_categorical("extra_trees", [False, True]))
        losses, f1s, iters = [], [], []
        for step, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed + 1).split(Xtr, ytr)):
            m = lgb.LGBMClassifier(**p, random_state=seed, n_jobs=C.N_JOBS)
            m.fit(Xtr.iloc[tr], ytr[tr], sample_weight=sqrt_w(ytr[tr], NC),
                  eval_set=[(Xtr.iloc[va], ytr[va])], eval_sample_weight=[sqrt_w(ytr[va], NC)],
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
            pr = m.predict_proba(Xtr.iloc[va])
            losses.append(log_loss(ytr[va], np.clip(pr, 1e-7, 1), sample_weight=sqrt_w(ytr[va], NC),
                                   labels=list(range(NC))))
            f1s.append(V.macro_f1(ytr[va], pr.argmax(1)))
            iters.append(int(m.best_iteration_ or p["n_estimators"]))
            t.report(float(np.mean(losses)), step)
            if t.should_prune():
                raise optuna.TrialPruned()
        t.set_user_attr("inner_f1", float(np.mean(f1s)))
        t.set_user_attr("best_iter", float(np.mean(iters)))
        return float(np.mean(losses))

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    st.enqueue_trial({k: LGB_STAGE1[k] for k in ["learning_rate", "max_depth", "num_leaves", "min_child_samples",
                                                 "colsample_bytree", "subsample", "reg_lambda", "reg_alpha",
                                                 "min_split_gain"]} | {"extra_trees": False})
    st.optimize(obj, n_trials=trials)
    bp = dict(st.best_params)
    bp["num_leaves"] = min(bp["num_leaves"], 2 ** bp["max_depth"])
    P, iters = lgb_cv_bag(LGB_COMMON | bp, Xtr, ytr, Xev, NC, n_splits=n_bag, seed=seed)
    if log is not None:
        log.append(dict(best_inner_wlogloss=st.best_value, **st.best_trial.user_attrs, final_iters=iters,
                        n_pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in st.trials), **bp))
    return P


def m_catboost(Xtr, ytr, Xev, device="cpu", cb_weights="SqrtBalanced", seed=0, n_bag=3, log=None, **_):
    kw = dict(loss_function="MultiClass", auto_class_weights=None if cb_weights == "None" else cb_weights,
              depth=5, learning_rate=0.04, iterations=4000, l2_leaf_reg=6, random_strength=1.5,
              bootstrap_type="Bayesian", bagging_temperature=0.5, border_count=128,
              od_type="Iter", od_wait=150, verbose=0, thread_count=C.N_JOBS,
              task_type="GPU" if device == "cuda" else "CPU")
    if device != "cuda":
        kw["rsm"] = 0.7  # 열 무작위 추출 (CPU 에서만 지원)
    P, iters = 0.0, []
    for i, (tr, va) in enumerate(StratifiedKFold(n_bag, shuffle=True, random_state=seed).split(Xtr, ytr)):
        m = CatBoostClassifier(**(kw | dict(random_seed=seed * 10 + i)))
        m.fit(Xtr.iloc[tr], ytr[tr], eval_set=(Xtr.iloc[va], ytr[va]), use_best_model=True)
        P = P + m.predict_proba(Xev)
        iters.append(int(m.get_best_iteration() or 0))
    if log is not None:
        log.append(dict(best_iters=iters))
    return P / n_bag


def m_tabm(Xtr, ytr, Xev, device="cpu", seed=0, **_):
    import torch
    import torch.nn.functional as Fn
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from tabm import TabM

    torch.manual_seed(seed)
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    pre = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                        QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed))
    A = pre.fit_transform(Xtr).astype(np.float32)
    Bv = pre.transform(Xev).astype(np.float32)
    tr, va = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=seed)
    xt = torch.tensor(A, device=dev)
    yt = torch.tensor(ytr, device=dev)
    bins = compute_bins(xt[tr], n_bins=32)
    model = TabM.make(n_num_features=A.shape[1], d_out=NC, k=32, n_blocks=2, d_block=256, dropout=0.2,
                      num_embeddings=PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=False, version="B")
                      ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    cw = torch.tensor(np.sqrt(len(ytr) / (NC * np.bincount(ytr, minlength=NC))), dtype=torch.float32, device=dev)

    def predict(x):
        model.eval()
        with torch.no_grad():
            out = [torch.softmax(model(x[i:i + 2048]), -1).mean(1) for i in range(0, len(x), 2048)]
        return torch.cat(out).cpu().numpy()

    best, best_state, bad, bs = -1, None, 0, 256
    for ep in range(300):
        model.train()
        perm = torch.tensor(np.random.default_rng(seed + ep).permutation(tr), device=dev)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            logits = model(xt[idx])
            loss = Fn.cross_entropy(logits.flatten(0, 1), yt[idx].repeat_interleave(logits.shape[1]), weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = V.macro_f1(ytr[va], predict(xt[va]).argmax(1))
        if f1 > best + 1e-4:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    model.load_state_dict(best_state)
    return predict(torch.tensor(Bv, device=dev))


def m_tabpfn(Xtr, ytr, Xev, device="cpu", **_):
    from tabpfn import TabPFNClassifier
    m = TabPFNClassifier(device=device, n_estimators=8, ignore_pretraining_limits=True, balance_probabilities=True)
    return m.fit(Xtr.to_numpy(np.float32), ytr).predict_proba(Xev.to_numpy(np.float32))


def m_lgbm_base(Xtr, ytr, Xev, seed=0, n_bag=3, **_):
    """ablation: hier_hyp 의 1단계와 동일 (2단계 전문가 효과 분리용)"""
    return lgb_cv_bag(LGB_STAGE1, Xtr, ytr, Xev, NC, n_splits=n_bag, seed=seed)[0]


def m_hier_hyp(Xtr, ytr, Xev, hyp_cols=(), seed=0, n_bag=3, corr_thr=0.90, log=None, **_):
    """2단계 계층 분류: 1단계 4클래스 → 2단계 자망 vs 연안복합 전문가 (가설 피처 포함, 공선성 제거)"""
    p, it1 = lgb_cv_bag(LGB_STAGE1, Xtr, ytr, Xev, NC, n_splits=n_bag, seed=seed)
    pair = np.isin(ytr, [I_COAST, I_GILL])
    full_tr, full_ev = Xtr.attrs["full"], Xev.attrs["full"]
    cand = list(dict.fromkeys(list(Xtr.columns) + list(hyp_cols)))
    cols = decorrelate(full_tr.loc[pair, cand], corr_thr)
    yb = (ytr[pair] == I_GILL).astype(int)
    q, it2 = lgb_cv_bag(LGB_STAGE2, full_tr.loc[pair, cols].reset_index(drop=True), yb, full_ev[cols], 2,
                        n_splits=n_bag, seed=seed)
    q = q[:, 1]
    if log is not None:
        log.append(dict(stage2_cand=len(cand), stage2_after_corr=len(cols), stage1_iters=it1, stage2_iters=it2))
    tot = p[:, I_COAST] + p[:, I_GILL]
    p = p.copy()
    p[:, I_GILL], p[:, I_COAST] = tot * q, tot * (1 - q)
    return p


MODELS = {"base": m_base, "rf": m_rf, "lgbm_tuned": m_lgbm_tuned, "catboost": m_catboost,
          "tabm": m_tabm, "hier_hyp": m_hier_hyp, "tabpfn": m_tabpfn, "lgbm_base": m_lgbm_base}
ENS_MEMBERS = ["lgbm_tuned", "catboost", "tabm", "hier_hyp", "tabpfn"]


# ─────────────────────────────────────────────
# 앙상블 가중 + 클래스 보정 (모두 LOFO: 다른 fold OOF 로 맞춰 해당 fold 에 적용)
# ─────────────────────────────────────────────
def fit_ens_weights(Ps, y, shrink):
    """가중 log-loss 최소 볼록결합 가중치 → 균등 가중으로 shrink 만큼 수축 (멤버 간 상관 대응)"""
    M = len(Ps)
    sw = sqrt_w(y, NC)

    def loss(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        return log_loss(y, np.clip(np.tensordot(w, Ps, 1), 1e-7, 1), sample_weight=sw, labels=list(range(NC)))

    z = minimize(loss, np.zeros(M), method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-3, fatol=1e-7)).x
    w = np.exp(z - z.max())
    w /= w.sum()
    return (1 - shrink) * w + shrink / M


def ensemble_lofo(Ps, y, fold_id, shrink):
    Ps = np.asarray(Ps)
    out, W = np.zeros(Ps.shape[1:]), {}
    for f in np.unique(fold_id):
        o = fold_id != f
        w = fit_ens_weights(Ps[:, o], y[o], shrink)
        out[fold_id == f] = np.tensordot(w, Ps[:, fold_id == f], 1)
        W[int(f)] = np.round(w, 3).tolist()
    return out, W


def fit_class_weights(P, y, shrink, iters=2):
    """클래스 0 배율 고정 (argmax 는 배율 전체 스케일에 불변 → 남는 1개 자유도는 중복),
    탐색 0.5~2배, 결과 log 배율을 shrink 만큼 0 쪽으로 수축"""
    w = np.ones(NC)
    grid = np.exp(np.linspace(-0.7, 0.7, 15))
    for _ in range(iters):
        for c in range(1, NC):
            best = (V.macro_f1(y, (P * w).argmax(1)), w[c])
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = V.macro_f1(y, (P * w2).argmax(1))
                if s > best[0] + 1e-9:
                    best = (s, g)
            w[c] = best[1]
    return w ** (1 - shrink)


def adjust_lofo(P, y, fold_id, shrink):
    out = P.copy()
    for f in np.unique(fold_id):
        o = fold_id != f
        out[fold_id == f] = P[fold_id == f] * fit_class_weights(P[o], y[o], shrink)
    return out


# ─────────────────────────────────────────────
# 평가지표
# ─────────────────────────────────────────────
def fast_macro_f1(y, p):
    cm = np.bincount(y * NC + p, minlength=NC * NC).reshape(NC, NC)
    tp = np.diag(cm)
    d = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)))


def boot_ci(y, pred, n_boot, seed=0):
    """선박 단위 부트스트랩 95% 신뢰구간"""
    if n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    s = [fast_macro_f1(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def metrics_row(name, pool, y, P, fold_id, n_boot=1000, train_f1=None):
    pred = P.argmax(1)
    Pn = P / P.sum(1, keepdims=True)
    per_fold = [V.macro_f1(y[fold_id == f], pred[fold_id == f]) for f in np.unique(fold_id)]
    cls = f1_score(y, pred, average=None, labels=list(range(NC)))
    pair = np.isin(y, [I_COAST, I_GILL])
    lo, hi = boot_ci(y, pred, n_boot)
    row = dict(pool=pool, model=name, macro_f1=V.macro_f1(y, pred), f1_ci_lo=lo, f1_ci_hi=hi,
               f1_fold_mean=np.mean(per_fold), f1_fold_std=np.std(per_fold, ddof=1),
               f1_fold_se=np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)),
               logloss=log_loss(y, np.clip(Pn, 1e-7, 1), labels=list(range(NC))), mcc=matthews_corrcoef(y, pred),
               acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
               **{f"f1_{c}": v for c, v in zip(C.CLASSES, cls)},
               gill_vs_coast_f1=f1_score(y[pair] == I_GILL, P[pair][:, [I_COAST, I_GILL]].argmax(1) == 1,
                                         average="macro"))
    if train_f1 is not None:
        row["train_f1"] = float(np.mean(train_f1))
        row["overfit_gap"] = row["train_f1"] - row["f1_fold_mean"]
    return row


def plot_cm(y, P, title, path):
    cm = confusion_matrix(y, P.argmax(1), labels=list(range(NC)), normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NC):
        for j in range(NC):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(NC), C.CLASSES, rotation=20); ax.set_yticks(range(NC), C.CLASSES)
    ax.set_xlabel("예측"); ax.set_ylabel("정답"); ax.set_title(title)
    fig.savefig(path, bbox_inches="tight", dpi=120); plt.close(fig)


# ─────────────────────────────────────────────
# 랭킹 연결: 플래그 버전 rank_features.py 결과만 사용 (train_classifiers.py 와 동일)
# ─────────────────────────────────────────────
def ranking_paths(a):
    if a.rankings:
        rk = Path(a.rankings)
        fl = rk.with_name(rk.name.replace("fold_rankings", "rank_flags", 1)) if "fold_rankings" in rk.name else None
        return rk, fl
    sfx = f"_{a.rank_tag}" if a.rank_tag else ""
    return C.RESULT_DIR / f"fold_rankings{sfx}.json", C.RESULT_DIR / f"rank_flags{sfx}.json"


def run_my_ranking(a):
    import rank_features as RF
    if not hasattr(RF, "FLAGS") or not hasattr(RF, "LEGACY"):
        raise SystemExit("❌ 같은 폴더의 rank_features.py 가 랭킹 조건 플래그 버전이 아닙니다 (FLAGS/LEGACY 없음).")
    cfg = dict(RF.LEGACY if a.rank_legacy else RF.FLAGS)
    cfg["tag"] = a.rank_tag
    print(f"━━ 피처 랭킹 실행: rank_features.py ({'legacy' if a.rank_legacy else '추천 설정'}, tag='{a.rank_tag}') ━━")
    RF.main(cfg)


def load_ranking(a, S, y):
    rk, fl = ranking_paths(a)
    print(f"랭킹 파일: {rk}")
    if not rk.exists():
        raise SystemExit(f"❌ 랭킹 파일이 없습니다: {rk}\n"
                         f"   → python rank_features.py --tag {a.rank_tag or '<tag>'}  를 먼저 실행하거나 --rerank 를 붙이세요.")
    R = json.loads(rk.read_text(encoding="utf-8"))
    flags = json.loads(fl.read_text(encoding="utf-8")) if fl is not None and fl.exists() else None
    problems = []
    if flags is None:
        problems.append(f"짝이 되는 {fl.name if fl else 'rank_flags*.json'} 가 없습니다 → 수정 전(다른) 랭킹 코드로 만든 파일입니다.")
    else:
        dt = fl.stat().st_mtime - rk.stat().st_mtime
        if not -5 <= dt <= 300:
            problems.append(f"{rk.name} 와 {fl.name} 의 저장 시각이 {dt:+.0f}s 어긋납니다 "
                            f"→ 랭킹 파일이 다른 실행(다른 랭킹 코드)으로 덮어써졌을 수 있습니다.")
    for pool in a.pools:
        if pool not in R or "folds" not in R[pool] or "k_star" not in R[pool]:
            problems.append(f"랭킹 파일에 Pool {pool} 결과가 없습니다.")
            continue
        missing = sorted({c for o in R[pool]["folds"] for c in o} - set(S.columns))
        if missing:
            problems.append(f"Pool {pool} 랭킹 피처 {len(missing)}개가 현재 데이터에 없습니다 (예: {missing[:3]}).")
    F = V.folds(S, y)
    if len(F) != len(R.get("fold_val_idx", [])) or any(
            set(va.tolist()) != set(v) for (_, va), v in zip(F, R["fold_val_idx"])):
        problems.append("랭킹 파일의 fold 분할이 현재 데이터의 fold 분할과 다릅니다 → 랭킹 이후 데이터가 바뀌었습니다.")
    if problems:
        msg = "\n".join(f"   - {p}" for p in problems)
        if a.skip_rank_check:
            print(f"⚠ 랭킹 검증 경고 (--skip-rank-check 로 계속 진행):\n{msg}")
        else:
            raise SystemExit(f"❌ 내 랭킹 코드(rank_features.py)의 결과로 확인되지 않아 중단합니다:\n{msg}\n"
                             f"   → python train_classifiers_v2.py --rerank --rank-tag {a.rank_tag or '<tag>'}")
    if flags is not None:
        print("랭킹 플래그:", json.dumps(flags, ensure_ascii=False))
    return R, flags, rk


# ─────────────────────────────────────────────
def main(a):
    V.set_font()
    if a.rerank:
        run_my_ranking(a)
    S, y = V.load_ship()
    R, rank_flags, rk_path = load_ranking(a, S, y)
    out_tag = a.tag or a.rank_tag
    sfx = f"_{VERSION}" + (f"_{out_tag}" if out_tag else "")
    val_idx = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(val_idx):
        fold_id[v] = f
    models = [m for m in a.models if m != "tabpfn" or a.tabpfn]
    rows, logs, k_used, fs_info, ens_info = [], {}, {}, {}, {}
    for pool in a.pools:
        k = a.k or R[pool]["k_star"]
        k_used[pool] = k
        pool_cols = V.feature_pool(S, pool)
        hyp_all = [c for c in pool_cols if V.is_hyp(c)]

        # fold 별 피처: 랭킹 상위 k → train 에서만 |ρ| ≥ corr_thr 추가 제거 (모든 모델 공통)
        fold_feats = []
        for f, va in enumerate(val_idx):
            tr = np.setdiff1d(np.arange(len(y)), va)
            top = R[pool]["folds"][f][:k]
            fold_feats.append(decorrelate(S.iloc[tr][top].reset_index(drop=True), a.corr_thr))
        fs_info[pool] = [dict(fold=f, top_k=min(k, len(R[pool]["folds"][f])), after_corr=len(fs))
                         for f, fs in enumerate(fold_feats)]
        print(f"\n━━ Pool {pool}  (k={k} → |ρ|<{a.corr_thr} 후 {[len(x) for x in fold_feats]}개, "
              f"가설 피처 후보 {len(hyp_all)}개) ━━")

        OOF = {}
        for name in models:
            t = time.time()
            P = np.zeros((len(y), NC))
            log, tr_f1 = [], []
            for f, va in enumerate(val_idx):
                tr = np.setdiff1d(np.arange(len(y)), va)
                feats = fold_feats[f]
                Xtr = S.iloc[tr][feats].reset_index(drop=True)
                Xva = S.iloc[va][feats].reset_index(drop=True)
                Xev = pd.concat([Xva, Xtr], ignore_index=True)  # attrs 설정 전에 합쳐야 함
                Xtr.attrs["full"] = S.iloc[tr][pool_cols].reset_index(drop=True)
                Xev.attrs["full"] = pd.concat([S.iloc[va][pool_cols], S.iloc[tr][pool_cols]], ignore_index=True)
                hyp = [c for c in R[pool]["folds"][f] if c in hyp_all][:30]
                Pall = MODELS[name](Xtr, y[tr], Xev, trials=a.trials, device=a.device, log=log, hyp_cols=hyp,
                                    cb_weights=a.cb_weights, seed=f, n_bag=a.bag, corr_thr=a.corr_thr,
                                    vif_max=a.vif_max)
                P[va] = Pall[:len(va)]
                tr_f1.append(V.macro_f1(y[tr], Pall[len(va):].argmax(1)))
            OOF[name] = P
            logs[f"{pool}_{name}"] = log
            rows.append(metrics_row(name, pool, y, P, fold_id, a.boot, train_f1=tr_f1))
            r = rows[-1]
            print(f"  {name:<14s} macro-F1 {r['macro_f1']:.4f} [{r['f1_ci_lo']:.3f}, {r['f1_ci_hi']:.3f}]  "
                  f"train {r['train_f1']:.3f} (gap {r['overfit_gap']:+.3f})  {time.time() - t:5.0f}s", flush=True)
            np.save(C.RESULT_DIR / f"oof_{pool}_{name}{sfx}.npy", P)
        if a.reuse_oof:
            for m in ENS_MEMBERS:
                p = C.RESULT_DIR / f"oof_{pool}_{m}{sfx}.npy"
                if m not in OOF and p.exists():
                    OOF[m] = np.load(p)
                    rows.append(metrics_row(m, pool, y, OOF[m], fold_id, a.boot) | {"members": "(reused oof)"})
                    print(f"  {m:<14s} macro-F1 {rows[-1]['macro_f1']:.4f}  (저장된 OOF 재사용)")
        mem = [m for m in ENS_MEMBERS if m in OOF]
        if len(mem) >= 2:
            Ps = [OOF[m] for m in mem]
            flat = np.array([p.ravel() for p in Ps])
            ens_info[pool] = dict(members=mem, member_corr=np.round(np.corrcoef(flat), 3).tolist())
            Em = np.mean(Ps, axis=0)
            Ew, W = ensemble_lofo(Ps, y, fold_id, a.ens_shrink)
            Ewa = adjust_lofo(Ew, y, fold_id, a.adj_shrink)
            ens_info[pool]["fold_weights"] = W
            for nm, Q in [("ensemble_mean", Em), ("ensemble_w", Ew), ("ensemble_w+adj", Ewa)]:
                rows.append(metrics_row(nm, pool, y, Q, fold_id, a.boot) | {"members": "+".join(mem)})
                OOF[nm] = Q
                r = rows[-1]
                print(f"  {nm:<14s} macro-F1 {r['macro_f1']:.4f} [{r['f1_ci_lo']:.3f}, {r['f1_ci_hi']:.3f}]  "
                      f"logloss {r['logloss']:.3f}")
            print(f"  앙상블 가중치(fold별, {'/'.join(mem)}): {W}")
        best = max([r for r in rows if r["pool"] == pool], key=lambda r: r["macro_f1"])
        plot_cm(y, OOF[best["model"]], f"Pool {pool} · {best['model']}  F1={best['macro_f1']:.3f}",
                C.RESULT_DIR / f"cm_{pool}_best{sfx}.png")
        if models:
            base_f1 = V.macro_f1(y, OOF[models[0]].argmax(1))
            plot_cm(y, OOF[models[0]], f"Pool {pool} · {models[0]}  F1={base_f1:.3f}",
                    C.RESULT_DIR / f"cm_{pool}_{models[0]}{sfx}.png")

    M = pd.DataFrame(rows)
    M["k"] = M["pool"].map(k_used)
    M["rank_file"] = str(rk_path)
    M["rank_flags"] = json.dumps(rank_flags, ensure_ascii=False) if rank_flags else ""
    M.round(4).to_csv(C.RESULT_DIR / f"metrics{sfx}.csv", index=False, encoding="utf-8-sig")
    (C.RESULT_DIR / f"tuning_log{sfx}.json").write_text(json.dumps(logs, ensure_ascii=False, default=float),
                                                        encoding="utf-8")
    (C.RESULT_DIR / f"run_info{sfx}.json").write_text(json.dumps(
        dict(version=VERSION, rank_file=str(rk_path), rank_flags=rank_flags, k=k_used, feature_selection=fs_info,
             ensemble=ens_info, models=models, pools=a.pools, trials=a.trials, cb_weights=a.cb_weights,
             corr_thr=a.corr_thr, vif_max=a.vif_max, bag=a.bag, ens_shrink=a.ens_shrink, adj_shrink=a.adj_shrink),
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    show = ["pool", "model", "macro_f1", "f1_ci_lo", "f1_ci_hi", "f1_fold_std", "logloss", "train_f1",
            "overfit_gap"] + [f"f1_{c}" for c in C.CLASSES] + ["gill_vs_coast_f1"]
    print("\n" + M[[c for c in show if c in M]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=["A", "B"])
    ap.add_argument("--models", nargs="*", default=["base", "lgbm_tuned", "catboost", "hier_hyp"],
                    choices=list(MODELS))
    ap.add_argument("--rank-tag", default="", help="rank_features.py 의 --tag 값")
    ap.add_argument("--rerank", action="store_true", help="분류 전에 rank_features.py 를 직접 실행")
    ap.add_argument("--rank-legacy", action="store_true", help="--rerank 시 수정 전 랭킹 동작(LEGACY)")
    ap.add_argument("--rankings", default=None, help="랭킹 파일 경로 직접 지정 (--rank-tag 대신)")
    ap.add_argument("--skip-rank-check", action="store_true", help="랭킹 출처 검증 실패를 경고로만 처리")
    ap.add_argument("--tabpfn", action="store_true")
    ap.add_argument("--trials", type=int, default=30, help="Optuna 시도 횟수 (가지치기로 실제 비용은 더 적음)")
    ap.add_argument("--k", type=int, default=None, help="k* 대신 고정 k")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reuse-oof", action="store_true")
    ap.add_argument("--cb-weights", default="SqrtBalanced", choices=["Balanced", "SqrtBalanced", "None"])
    # v2 전용 설정
    ap.add_argument("--corr-thr", type=float, default=0.90, help="fold train 에서 추가 제거할 Spearman |ρ| 기준")
    ap.add_argument("--vif-max", type=float, default=10.0, help="로지스틱 회귀의 VIF 상한")
    ap.add_argument("--bag", type=int, default=3, help="CV-bagging 내부 fold 수")
    ap.add_argument("--ens-shrink", type=float, default=0.3, help="앙상블 가중치를 균등 가중으로 수축하는 비율")
    ap.add_argument("--adj-shrink", type=float, default=0.5, help="클래스 보정 배율(log)을 0 쪽으로 수축하는 비율")
    ap.add_argument("--boot", type=int, default=1000, help="macro-F1 부트스트랩 반복 수 (0 = 끔)")
    a = ap.parse_args()
    if a.rerank and a.rankings:
        ap.error("--rerank 는 --rankings 와 함께 쓸 수 없습니다.")
    if not 0 < a.corr_thr <= 1:
        ap.error("--corr-thr 는 0~1 사이여야 합니다.")
    if a.bag < 2:
        ap.error("--bag 은 2 이상이어야 합니다.")
    if a.tabpfn and "tabpfn" not in a.models:
        a.models.append("tabpfn")
    main(a)
