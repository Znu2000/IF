"""
[STEP2] TabPFN v2 (사전학습 표 데이터 파운데이션 모델) ─ 단독 성능과 1단계 앙상블에 섞었을 때 비교
  모델    : Prior Labs TabPFN v2 분류 원본 가중치 (tabpfn-v2-classifier.ckpt, Prior Labs License v1.1 = Apache 2.0 + 출처 표기)
            "Built with PriorLabs-TabPFN" ─ 보고서·README 에 표기. 가중치 파일은 깃헙에 올리지 않음
            model_path 를 직접 지정 → 패키지 기본값(다른 라이선스의 최신 버전) 자동 다운로드 없음
  입력    : 1단계 최종 설정 prior_now_noshift 와 같은 피처 (fold 별 상위 k + 지역 사전확률, 연도 차이 피처 제외)
            TabPFN v2 권장 범위 안 (학습 ≤ 1만 행, 피처 ≤ 500, 클래스 ≤ 10)
  평가    : step1_experiments 와 같은 fold·지표 (micro-F1). 앙상블 = 1단계 OOF/테스트 확률과 가중 평균
  출력    : results/step1_oof_{config}.npy, step1_test_{config}.npy, step1_results.csv 에 행 추가
            → python apply_step1.py --config <최고 설정> --base submission_combo_blank.xlsx 로 제출 파일 생성
실행: python tabpfn_exp.py --ckpt ~/IF/tabpfn-v2-classifier.ckpt [--device cuda]
      python tabpfn_exp.py --dry   (TabPFN 대신 로지스틱 회귀로 흐름만 확인)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import cv_common as V
import fe_config as C
import train_classifiers as TC
from step1_experiments import region_prior, shift_feats

BASE = "prior_now_noshift"
BLEND = [0.2, 0.35, 0.5]  # TabPFN 비중


def make_model(a):
    if a.dry:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=300))
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(model_path=str(Path(a.ckpt).expanduser()), device=a.device, n_estimators=a.n_est,
                            random_state=C.RANDOM_STATE, ignore_pretraining_limits=True)


def fit_predict(a, Xtr, ytr, Xq_list):
    clean = lambda X: X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    m = make_model(a).fit(clean(Xtr), ytr)
    return [m.predict_proba(clean(X)) for X in Xq_list]


def with_prior(X, P):
    return pd.concat([X.reset_index(drop=True), P.reset_index(drop=True)], axis=1)


def main(a):
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    k = R["B"]["k_star"]
    lab = dict(zip(S["rfid_id"].astype(str), y))
    T["_rid"] = T["rfid_id"].astype(str)
    ov = T[T["_rid"].isin(lab)].reset_index(drop=True)
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(S["rfid_id"].astype(str), fold_id))).to_numpy()
    drop = set(shift_feats(S, T, V.feature_pool(S, "B")))
    g = pd.read_csv(C.RESULT_DIR / "feature_ranking_B.csv")
    g = g[g["redundant_folds"] <= 2].sort_values("final_rank")["feature"].tolist()
    print(f"{'dry(로지스틱)' if a.dry else 'TabPFN v2 ' + a.ckpt}, 학습 {len(y)}척, 피처 k={k} + 지역 사전확률, "
          f"연도 차이 제외 {sorted(drop)}", flush=True)

    oof = np.zeros((len(y), len(C.CLASSES)))
    cross_u = np.zeros((len(ov), len(C.CLASSES)))
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), va)
        feats = [c for c in R["B"]["folds"][f] if c not in drop][:k]
        Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
        ovf = ov[ov_fold == f].reset_index(drop=True)
        Xs = [with_prior(Str[feats], region_prior(Str, y[tr], Str, self_in_ref=True)),
              with_prior(Sva[feats], region_prior(Str, y[tr], Sva)),
              with_prior(ovf[feats], region_prior(Str, y[tr], ovf))]
        oof[va], cross_u[ov_fold == f] = fit_predict(a, Xs[0], y[tr], Xs[1:])
        print(f"  fold {f}: acc {accuracy_score(y[va], oof[va].argmax(1)):.4f} ({(time.time() - t0) / 60:.1f}분)",
              flush=True)
    feats = [c for c in g if c not in drop][:k]
    Xa = with_prior(S[feats], region_prior(S, y, S, self_in_ref=True))
    Xt = with_prior(T[feats], region_prior(S, y, T))
    (Ptest,) = fit_predict(a, Xa, y, [Xt])
    print(f"  전체 학습 → 테스트 예측 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    seen = T["_rid"].isin(lab).to_numpy()
    base_oof = np.load(C.RESULT_DIR / f"step1_oof_{BASE}.npy")
    base_test = np.load(C.RESULT_DIR / f"step1_test_{BASE}.npy")
    tag = "dry" if a.dry else "tabpfn"
    cands = {BASE: (base_oof, base_test, None), tag: (oof, Ptest, cross_u)}
    for b in BLEND:
        cands[f"{tag}_blend{int(b * 100)}"] = ((1 - b) * base_oof + b * oof, (1 - b) * base_test + b * Ptest, None)
    rows = []
    for name, (O, P, CU) in cands.items():
        w = TC.fit_class_weights(O, y, score=accuracy_score)
        adj = TC.adjust_lofo(O, y, fold_id, score=accuracy_score)
        r = dict(config=name, cv=accuracy_score(y, O.argmax(1)), cv_adj=accuracy_score(y, adj.argmax(1)),
                 cv_macro=V.macro_f1(y, O.argmax(1)),
                 cross_unseen=accuracy_score(y_ov, CU.argmax(1)) if CU is not None else np.nan,
                 cross_seen=accuracy_score(y_ov, P[seen].argmax(1)),
                 cross_seen_adj=accuracy_score(y_ov, (P[seen] * w).argmax(1)),
                 weights=json.dumps([round(float(v), 3) for v in w]))
        rows.append(r)
        if name != BASE:
            np.save(C.RESULT_DIR / f"step1_oof_{name}.npy", O)
            np.save(C.RESULT_DIR / f"step1_test_{name}.npy", P)
    E = pd.DataFrame(rows)
    print(E.drop(columns="weights").round(4).to_string(index=False))
    p = C.RESULT_DIR / "step1_results.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    new = E[E["config"] != BASE]
    pd.concat([old[~old.get("config", pd.Series(dtype=str)).isin(new["config"])], new]).round(4).to_csv(
        p, index=False, encoding="utf-8-sig")
    best = E.sort_values(["cross_seen_adj", "cv"], ascending=False).iloc[0]["config"]
    print(f"cross_seen_adj 최고: {best}  (기존 {BASE} 대비 확인 후 apply_step1.py --config {best})")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/IF/tabpfn-v2-classifier.ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_est", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    main(ap.parse_args())
