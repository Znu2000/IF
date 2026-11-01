"""
[STEP1] 리더보드(micro-F1) 개선 실험 ─ 모델은 그대로(lgbm_base + catboost + hier_hyp), 피처·학습 목표만 바꿈
  1) 지역 사전확률 (region prior) : 모항 주변 / 주 조업지 주변 학습 선박 k=30 척의 업종 비율
       근거: 어촌·항구마다 주력 어업이 다름 (지역 어업 구조). 배 1척이 아니라 30척 분포라 특정 선박 식별이 아님
       누수 방지: CV 에서는 학습 fold 선박만 이웃으로 쓰고, 학습 선박 자신의 값은 자기 자신을 빼고 계산
  2) 학습 목표 : class_weight 균형(기존) vs 없음(리더보드 micro-F1 = 정확도 기준)
  3) 연도 차이 피처 제거 : 학습(2026)과 테스트(2025)를 혼자서도 잘 구분하는 피처(단변량 AUC 편차 > 0.1) 제외
  평가 (모두 micro-F1)
    cv          : 2026 선박 5-fold OOF (처음 보는 배)
    cross_unseen: fold 모델로 그 fold 검증 선박의 2025 테스트 항적을 예측 (처음 보는 배 + 연도 차이)
    cross_seen  : 학습 선박 전체로 학습 → 학습에도 있는 테스트 선박 2025 항적 예측 (리더보드와 같은 상황)
    라벨은 2026 업종 (같은 배의 업종이 해마다 같다고 가정)
  출력: results/step1_results.csv, step1_oof_{config}.npy, step1_test_{config}.npy
실행: python step1_experiments.py [--configs base prior prior_now prior_now_noshift]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import BallTree

import cv_common as V
import fe_config as C
import train_classifiers as TC

K_NB = 30
CORE = ["lgbm_base", "catboost", "hier_hyp"]
PRIOR_SETS = {"home": ["home_lat", "home_lon"], "fish": ["fishing_lat", "fishing_lon"]}
CONFIGS = {  # 이름: (지역 사전확률, 클래스 가중, 연도 차이 피처 제거)
    "base": (False, "balanced", False),
    "prior": (True, "balanced", False),
    "prior_now": (True, "none", False),
    "prior_now_noshift": (True, "none", True),
}


def region_prior(S_ref, y_ref, Q, self_in_ref=False):
    """Q 각 선박 주변 학습 선박 K_NB 척의 업종 비율 (모항 / 주 조업지 기준)"""
    out = {}
    for name, cols in PRIOR_SETS.items():
        ref = S_ref[cols].to_numpy(float)
        ok = ~np.isnan(ref).any(1)
        tree = BallTree(np.radians(ref[ok]), metric="haversine")
        yr = y_ref[ok]
        q = Q[cols].to_numpy(float)
        qok = ~np.isnan(q).any(1)
        P = np.full((len(Q), len(C.CLASSES)), np.nan)
        dist = np.full(len(Q), np.nan)
        if qok.any():
            k = K_NB + int(self_in_ref)
            d, i = tree.query(np.radians(q[qok]), k=k)
            if self_in_ref:  # 학습 선박 자신은 이웃에서 제외 (자기 라벨 누수 방지)
                d, i = d[:, 1:], i[:, 1:]
            P[qok] = np.stack([(yr[i] == c).mean(1) for c in range(len(C.CLASSES))], 1)
            dist[qok] = d.mean(1) * 6371.0
        for c, cl in enumerate(C.CLASSES):
            out[f"prior_{name}_{cl}"] = P[:, c]
        out[f"prior_{name}_dist_km"] = dist
    return pd.DataFrame(out, index=Q.index)


def shift_feats(S, T, cols, thr=0.1):
    bad = []
    for c in cols:
        a, b = S[c].dropna(), T[c].dropna()
        if len(a) > 50 and len(b) > 50 and a.nunique() > 1:
            auc = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
            if abs(auc - 0.5) > thr:
                bad.append(c)
    return bad


def fit_core(Xtr, ytr, Xq_list, full_tr, full_q_list, hyp, cw):
    """핵심 3모델 평균 확률. Xq_list 의 각 질의 집합에 대해 반환"""
    TC.LGB_BASE["class_weight"] = None if cw == "none" else "balanced"
    outs = [0] * len(Xq_list)
    for m in CORE:
        for j, (Xq, fq) in enumerate(zip(Xq_list, full_q_list)):
            Xtr.attrs["full"], Xq.attrs["full"] = full_tr, fq
            p = TC.MODELS[m](Xtr, ytr, Xq, hyp_cols=hyp, cb_weights=None if cw == "none" else "SqrtBalanced")
            outs[j] = outs[j] + p / len(CORE)
    return outs


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
    pool_cols = V.feature_pool(S, "B")
    lab = dict(zip(S["rfid_id"].astype(str), y))
    T["_rid"] = T["rfid_id"].astype(str)
    ov = T[T["_rid"].isin(lab)].reset_index(drop=True)          # 학습에도 있는 테스트 선박
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(S["rfid_id"].astype(str), fold_id))).to_numpy()
    shift = shift_feats(S, T, pool_cols)
    print(f"테스트 중 학습에도 있는 선박 {len(ov):,}척 / 연도 차이 피처 {len(shift)}개: {shift}")
    g = pd.read_csv(C.RESULT_DIR / "feature_ranking_B.csv")
    g = g[g["redundant_folds"] <= 2].sort_values("final_rank")["feature"].tolist()

    rows = []
    for name in a.configs:
        use_prior, cw, noshift = CONFIGS[name]
        drop = set(shift) if noshift else set()
        oof = np.zeros((len(y), len(C.CLASSES)))
        cross_u = np.zeros((len(ov), len(C.CLASSES)))
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), va)
            feats = [c for c in R["B"]["folds"][f] if c not in drop][:k]
            hyp = [c for c in R["B"]["folds"][f] if V.is_hyp(c) and c not in drop][:30]
            Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
            ovf = ov[ov_fold == f].reset_index(drop=True)
            Xs = [Str[feats], Sva[feats], ovf[feats]]
            Fs = [Str[pool_cols], Sva[pool_cols], ovf[pool_cols]]
            if use_prior:
                pr = [region_prior(Str, y[tr], Str, self_in_ref=True), region_prior(Str, y[tr], Sva),
                      region_prior(Str, y[tr], ovf)]
                Xs = [pd.concat([x.reset_index(drop=True), p.reset_index(drop=True)], axis=1) for x, p in zip(Xs, pr)]
                Fs = [pd.concat([x.reset_index(drop=True), p.reset_index(drop=True)], axis=1) for x, p in zip(Fs, pr)]
            Pva, Pov = fit_core(Xs[0], y[tr], Xs[1:], Fs[0], Fs[1:], hyp, cw)
            oof[va] = Pva
            cross_u[ov_fold == f] = Pov
            print(f"  [{name}] fold {f} done ({(time.time() - t0) / 60:.1f}분)", flush=True)
        # 학습 선박 전체 → 테스트 전체 (cross_seen + 제출용)
        feats = [c for c in g if c not in drop][:k]
        hyp = [c for c in g if V.is_hyp(c) and c not in drop][:30]
        Xa, Xt = S[feats].reset_index(drop=True), T[feats].reset_index(drop=True)
        Fa, Ft = S[pool_cols].reset_index(drop=True), T[pool_cols].reset_index(drop=True)
        if use_prior:
            pa, pt = region_prior(S, y, S, self_in_ref=True), region_prior(S, y, T)
            Xa, Xt = pd.concat([Xa, pa.reset_index(drop=True)], axis=1), pd.concat([Xt, pt.reset_index(drop=True)], axis=1)
            Fa, Ft = pd.concat([Fa, pa.reset_index(drop=True)], axis=1), pd.concat([Ft, pt.reset_index(drop=True)], axis=1)
        (Ptest,) = fit_core(Xa, y, [Xt], Fa, [Ft], hyp, cw)
        np.save(C.RESULT_DIR / f"step1_oof_{name}.npy", oof)
        np.save(C.RESULT_DIR / f"step1_test_{name}.npy", Ptest)
        P_seen = Ptest[T["_rid"].isin(lab).to_numpy()]
        w = TC.fit_class_weights(oof, y, score=accuracy_score)  # micro 기준 클래스 가중 (OOF 로 맞춤)
        adj = TC.adjust_lofo(oof, y, fold_id, score=accuracy_score)
        r = dict(config=name, cv=accuracy_score(y, oof.argmax(1)), cv_adj=accuracy_score(y, adj.argmax(1)),
                 cv_macro=V.macro_f1(y, oof.argmax(1)),
                 cross_unseen=accuracy_score(y_ov, cross_u.argmax(1)),
                 cross_unseen_adj=accuracy_score(y_ov, (cross_u * w).argmax(1)),
                 cross_seen=accuracy_score(y_ov, P_seen.argmax(1)),
                 cross_seen_adj=accuracy_score(y_ov, (P_seen * w).argmax(1)),
                 weights=json.dumps([round(float(v), 3) for v in w]))
        rows.append(r)
        print({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r.items()}, flush=True)
        pd.DataFrame(rows).round(4).to_csv(C.RESULT_DIR / "step1_results.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    main(ap.parse_args())
