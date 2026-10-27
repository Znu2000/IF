"""
[CLS] 과제 1 조업형태 분류 실험 (선박 단위, rfid_id StratifiedGroupKFold 5-fold)

피처: rank_features.py 가 fold 별 train 에서 만든 순위의 상위 k* (선정 누수 없음)
풀  : A = 위치 제외(순수 행동) / B = 위치 포함

모델
  M0 base      : Logistic Regression (분위수 정규화 + class_weight)   ← 비교 기준
  M1 rf        : Random Forest (balanced_subsample)
  M2 lgbm_tuned: LightGBM + Optuna (fold 내부 3-fold 에서만 튜닝 = nested)
  M3 catboost  : CatBoost (auto_class_weights=SqrtBalanced, 내부 검증 early stopping)
  M4 tabm      : TabM (ICLR 2025, BatchEnsemble MLP + piecewise-linear 임베딩)  ← SOTA
  M5 hier_hyp  : [가설 모델] 2단계 계층 분류
                 1단계 4클래스 LightGBM → 2단계 자망 vs 연안복합 전문 분류기
                 (top-k + 가설 피처 A 재방문·B 직선투망·C 속도구조·D 선회·E 하루리듬·G 이질성·H 계절)
                 P(자망) = P1(자망∪연안복합) · P2(자망|둘 중 하나)
  M6 ensemble  : M2+M3+M4+M5 확률 평균  (+ macro-F1 클래스 가중 보정: 다른 fold OOF 로만 학습)
  (옵션) tabpfn: --tabpfn 플래그. TabPFN 가중치는 HF 라이선스 동의·로그인이 필요 → 사용자가 직접 설정 후 사용

실행: python train_classifiers.py                 (A, B 모두)
      python train_classifiers.py --pools B --models lgbm_tuned tabm
      python train_classifiers.py --trials 60 --device cuda   (서버)
"""
import argparse
import json
import time
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
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
LGB_BASE = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced", max_bin=63,
                verbose=-1)


def balanced_w(y):
    w = len(y) / (NC * np.bincount(y, minlength=NC))
    return w[y]


# ─────────────────────────────────────────────
# 모델 정의: fit_predict(Xtr, ytr, Xva) → proba (n_va, 4)
# ─────────────────────────────────────────────
def m_base(Xtr, ytr, Xva, **_):
    m = make_pipeline(SimpleImputer(strategy="median"),
                      QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0),
                      LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
    return m.fit(Xtr, ytr).predict_proba(Xva)


def m_rf(Xtr, ytr, Xva, **_):
    m = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(600, min_samples_leaf=2, max_features="sqrt",
                                             class_weight="balanced_subsample", n_jobs=C.N_JOBS, random_state=0))
    return m.fit(Xtr, ytr).predict_proba(Xva)


def _lgb_cv_score(params, X, y):
    sc = []
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=1).split(X, y):
        m = lgb.LGBMClassifier(**params, random_state=0, n_jobs=C.N_JOBS).fit(X.iloc[tr], y[tr])
        sc.append(V.macro_f1(y[va], m.predict(X.iloc[va])))
    return float(np.mean(sc))


def m_lgbm_tuned(Xtr, ytr, Xva, trials=30, log=None, **_):
    def obj(t):
        p = dict(LGB_BASE,
                 n_estimators=t.suggest_int("n_estimators", 200, 1000, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.015, 0.1, log=True),
                 num_leaves=t.suggest_int("num_leaves", 8, 96, log=True),
                 min_child_samples=t.suggest_int("min_child_samples", 5, 80, log=True),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))
        return _lgb_cv_score(p, Xtr, ytr)

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial({k: LGB_BASE[k] for k in ["n_estimators", "learning_rate", "num_leaves", "min_child_samples",
                                               "colsample_bytree", "subsample", "reg_lambda"]} | {"reg_alpha": 1e-3})
    st.optimize(obj, n_trials=trials)
    if log is not None:
        log.append(dict(best_inner_f1=st.best_value, **st.best_params))
    m = lgb.LGBMClassifier(**(LGB_BASE | st.best_params), random_state=0, n_jobs=C.N_JOBS).fit(Xtr, ytr)
    return m.predict_proba(Xva)


def m_catboost(Xtr, ytr, Xva, device="cpu", cb_weights="SqrtBalanced", **_):
    a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=0)
    kw = dict(loss_function="MultiClass", auto_class_weights=cb_weights, depth=6, learning_rate=0.05,
              iterations=3000, l2_leaf_reg=3, random_seed=0, verbose=0, od_type="Iter", od_wait=200,
              thread_count=C.N_JOBS, task_type="GPU" if device == "cuda" else "CPU")
    m = CatBoostClassifier(**kw).fit(a, ya, eval_set=(b, yb), use_best_model=True)
    best = max(m.get_best_iteration(), 100)
    m = CatBoostClassifier(**(kw | dict(iterations=int(best * 1.15), od_type=None, od_wait=None))).fit(Xtr, ytr)
    return m.predict_proba(Xva)


def m_tabm(Xtr, ytr, Xva, device="cpu", seed=0, **_):
    import torch
    import torch.nn.functional as Fn
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from tabm import TabM

    torch.manual_seed(seed)
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    pre = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                        QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed))
    A = pre.fit_transform(Xtr).astype(np.float32)
    Bv = pre.transform(Xva).astype(np.float32)
    tr, va = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=seed)
    xt = torch.tensor(A, device=dev)
    yt = torch.tensor(ytr, device=dev)
    bins = compute_bins(xt[tr], n_bins=32)
    model = TabM.make(n_num_features=A.shape[1], d_out=NC, k=32, n_blocks=2, d_block=256, dropout=0.15,
                      num_embeddings=PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=False, version="B")
                      ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    cw = torch.tensor(len(ytr) / (NC * np.bincount(ytr, minlength=NC)), dtype=torch.float32, device=dev)

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
            logits = model(xt[idx])  # (b, k, C): k 개 앙상블 멤버 각각 학습
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


def m_tabpfn(Xtr, ytr, Xva, device="cpu", **_):
    from tabpfn import TabPFNClassifier  # 가중치: 사용자가 HF 라이선스 동의·로그인 후 사용 가능
    m = TabPFNClassifier(device=device, n_estimators=8, ignore_pretraining_limits=True,
                         balance_probabilities=True)
    return m.fit(Xtr.to_numpy(np.float32), ytr).predict_proba(Xva.to_numpy(np.float32))


def m_lgbm_base(Xtr, ytr, Xva, **_):
    """ablation: hier_hyp 의 1단계와 동일한 LightGBM 단독 (2단계 전문가 효과 분리용)"""
    return lgb.LGBMClassifier(**LGB_BASE, random_state=0, n_jobs=C.N_JOBS).fit(Xtr, ytr).predict_proba(Xva)


def m_hier_hyp(Xtr, ytr, Xva, hyp_cols=(), **_):
    """가설 모델: 자망↔연안복합만 따로 푸는 전문가를 둔 2단계 계층 분류"""
    s1 = lgb.LGBMClassifier(**LGB_BASE, random_state=0, n_jobs=C.N_JOBS).fit(Xtr, ytr)
    p = s1.predict_proba(Xva)
    pair = np.isin(ytr, [I_COAST, I_GILL])
    cols = list(dict.fromkeys(list(Xtr.columns) + list(hyp_cols)))
    XtrF = Xtr.attrs["full"][cols]
    XvaF = Xva.attrs["full"][cols]
    yb = (ytr[pair] == I_GILL).astype(int)
    s2 = lgb.LGBMClassifier(**(LGB_BASE | dict(colsample_bytree=0.5, n_estimators=700)),
                            random_state=0, n_jobs=C.N_JOBS).fit(XtrF[pair], yb)
    q = s2.predict_proba(XvaF)[:, 1]
    tot = p[:, I_COAST] + p[:, I_GILL]
    p = p.copy()
    p[:, I_GILL], p[:, I_COAST] = tot * q, tot * (1 - q)
    return p


MODELS = {"base": m_base, "rf": m_rf, "lgbm_tuned": m_lgbm_tuned, "catboost": m_catboost,
          "tabm": m_tabm, "hier_hyp": m_hier_hyp, "tabpfn": m_tabpfn, "lgbm_base": m_lgbm_base}
ENS_MEMBERS = ["lgbm_tuned", "catboost", "tabm", "hier_hyp", "tabpfn"]


# ─────────────────────────────────────────────
# macro-F1 클래스 가중 보정 (다른 fold 의 OOF 로만 학습 → 해당 fold 에 적용)
# ─────────────────────────────────────────────
def fit_class_weights(P, y, iters=3):
    w = np.ones(NC)
    grid = np.exp(np.linspace(-1.2, 1.2, 25))
    for _ in range(iters):
        for c in range(NC):
            best = (V.macro_f1(y, (P * w).argmax(1)), w[c])
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = V.macro_f1(y, (P * w2).argmax(1))
                if s > best[0] + 1e-9:
                    best = (s, g)
            w[c] = best[1]
    return w


def adjust_lofo(P, y, fold_id):
    out = P.copy()
    for f in np.unique(fold_id):
        o = fold_id != f
        out[fold_id == f] = P[fold_id == f] * fit_class_weights(P[o], y[o])
    return out


# ─────────────────────────────────────────────
def metrics_row(name, pool, y, P, fold_id):
    pred = P.argmax(1)
    per_fold = [V.macro_f1(y[fold_id == f], pred[fold_id == f]) for f in np.unique(fold_id)]
    cls = f1_score(y, pred, average=None, labels=list(range(NC)))
    pair = np.isin(y, [I_COAST, I_GILL])
    return dict(pool=pool, model=name, macro_f1=V.macro_f1(y, pred), f1_fold_mean=np.mean(per_fold),
                f1_fold_std=np.std(per_fold), acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
                **{f"f1_{c}": v for c, v in zip(C.CLASSES, cls)},
                gill_vs_coast_f1=f1_score(y[pair] == I_GILL, P[pair][:, [I_COAST, I_GILL]].argmax(1) == 1,
                                          average="macro"))


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


def main(a):
    V.set_font()
    S, y = V.load_ship()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    val_idx = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(val_idx):
        fold_id[v] = f
    models = [m for m in a.models if m != "tabpfn" or a.tabpfn]
    rows, logs = [], {}
    for pool in a.pools:
        k = a.k or R[pool]["k_star"]
        pool_cols = V.feature_pool(S, pool)
        hyp_all = [c for c in pool_cols if V.is_hyp(c)]
        print(f"\n━━ Pool {pool}  (k={k}, 가설 피처 후보 {len(hyp_all)}개) ━━")
        OOF = {}
        for name in models:
            t = time.time()
            P = np.zeros((len(y), NC))
            log = []
            for f, va in enumerate(val_idx):
                tr = np.setdiff1d(np.arange(len(y)), va)
                feats = R[pool]["folds"][f][:k]
                Xtr, Xva = S.iloc[tr][feats].reset_index(drop=True), S.iloc[va][feats].reset_index(drop=True)
                Xtr.attrs["full"] = S.iloc[tr][pool_cols].reset_index(drop=True)
                Xva.attrs["full"] = S.iloc[va][pool_cols].reset_index(drop=True)
                # 가설 피처: fold 랭킹 순서대로 상위 30개 (fold 내부 선정)
                hyp = [c for c in R[pool]["folds"][f] if c in hyp_all][:30]
                P[va] = MODELS[name](Xtr, y[tr], Xva, trials=a.trials, device=a.device, log=log, hyp_cols=hyp,
                                     cb_weights=a.cb_weights)
            OOF[name] = P
            logs[f"{pool}_{name}"] = log
            rows.append(metrics_row(name, pool, y, P, fold_id))
            print(f"  {name:<11s} macro-F1 {rows[-1]['macro_f1']:.4f} (fold {rows[-1]['f1_fold_mean']:.4f}"
                  f"±{rows[-1]['f1_fold_std']:.4f})  {time.time() - t:5.0f}s", flush=True)
            np.save(C.RESULT_DIR / f"oof_{pool}_{name}.npy", P)
        if a.reuse_oof:  # 다른 실행(예: 서버의 TabM)에서 저장한 OOF 를 앙상블에 합류
            for m in ENS_MEMBERS:
                p = C.RESULT_DIR / f"oof_{pool}_{m}.npy"
                if m not in OOF and p.exists():
                    OOF[m] = np.load(p)
                    rows.append(metrics_row(m, pool, y, OOF[m], fold_id) | {"members": "(reused oof)"})
                    print(f"  {m:<11s} macro-F1 {rows[-1]['macro_f1']:.4f}  (저장된 OOF 재사용)")
        mem = [m for m in ENS_MEMBERS if m in OOF]
        if len(mem) >= 2:
            E = np.mean([OOF[m] for m in mem], axis=0)
            rows.append(metrics_row("ensemble", pool, y, E, fold_id) | {"members": "+".join(mem)})
            Ea = adjust_lofo(E, y, fold_id)
            rows.append(metrics_row("ensemble+adj", pool, y, Ea, fold_id) | {"members": "+".join(mem)})
            OOF["ensemble"], OOF["ensemble+adj"] = E, Ea
            for r in rows[-2:]:
                print(f"  {r['model']:<11s} macro-F1 {r['macro_f1']:.4f} (fold {r['f1_fold_mean']:.4f}±{r['f1_fold_std']:.4f})")
        best = max([r for r in rows if r["pool"] == pool], key=lambda r: r["macro_f1"])
        plot_cm(y, OOF[best["model"]], f"Pool {pool} · {best['model']}  F1={best['macro_f1']:.3f}",
                C.RESULT_DIR / f"cm_{pool}_best.png")
        if models:
            base_f1 = V.macro_f1(y, OOF[models[0]].argmax(1))
            plot_cm(y, OOF[models[0]], f"Pool {pool} · {models[0]}  F1={base_f1:.3f}",
                    C.RESULT_DIR / f"cm_{pool}_{models[0]}.png")
    M = pd.DataFrame(rows)
    out = C.RESULT_DIR / ("metrics.csv" if a.tag == "" else f"metrics_{a.tag}.csv")
    M.round(4).to_csv(out, index=False, encoding="utf-8-sig")
    (C.RESULT_DIR / f"tuning_log{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(logs, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n" + M[["pool", "model", "macro_f1", "f1_fold_std"] + [f"f1_{c}" for c in C.CLASSES]
                   + ["gill_vs_coast_f1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=["A", "B"])
    ap.add_argument("--models", nargs="*", default=["base", "rf", "lgbm_tuned", "catboost", "tabm", "hier_hyp"])
    ap.add_argument("--tabpfn", action="store_true", help="TabPFN 추가 (HF 라이선스 동의·로그인 선행 필요)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--k", type=int, default=None, help="k* 대신 고정 k")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reuse-oof", action="store_true", help="results/oof_{pool}_{model}.npy 가 있으면 앙상블에 포함")
    # Pool B 검증: Balanced 0.456 → SqrtBalanced 0.531 (소수 클래스 과보정 완화)
    ap.add_argument("--cb-weights", default="SqrtBalanced", choices=["Balanced", "SqrtBalanced", "None"])
    a = ap.parse_args()
    if a.tabpfn and "tabpfn" not in a.models:
        a.models.append("tabpfn")
    main(a)
