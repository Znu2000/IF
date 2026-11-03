"""
[STEP7] 피처 전체 확인 ─ 상위 50개 대신 선박 피처 전체(연도 차이 2개 제외) + 지역 사전확률 + 재방문(rv_)·출어 주기(pd_)
  근거: 나연 ny_rank_conditions / ny_prior_allfeats 에서 전체 피처가 상위 50개보다 높았음 (2026 cv 도 0.604)
  모델·fold·후처리는 r4_rp 와 같음 (fit_core = lgbm_base + catboost + hier_hyp, 학습 가중 없음)
  판단: 2026 cv 로만 함. 2025 검증(cross_*)은 참고용이며 테스트A·B 시트의 선박은 빼고 채점
  기존 파일은 건드리지 않음: step1_results.csv 에 행을 추가하지 않고, 결과는 step7_* 새 파일에만 저장
  출력: results/step7_eval.csv, step7_{oof,test}_all_rp.npy   (같은 이름이 있으면 시각을 붙여 새 이름으로 저장)
실행: python step7_allfeats.py
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import cv_common as V
import fe_config as C
from step1_experiments import fit_core, region_prior, shift_feats
from step3_region import NCL, score
from step4_final import attach

SAMPLE = C.PROJECT_ROOT / "문제 데이터_수정_1027" / "테스트셋 및 결과제출 샘플" / "제출샘플.xlsx"
BASE = "r4_rp"
STAMP = time.strftime("%m%d_%H%M")


def out_path(name):
    p = C.RESULT_DIR / name
    return p.with_name(f"{p.stem}_{STAMP}{p.suffix}") if p.exists() else p


def main():
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    pool_cols = V.feature_pool(S, "B")
    attach(S, "")
    attach(T, "_test")
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f

    # 2025 검증 표본: 학습에도 있는 테스트 선박 중 테스트A·B 시트 선박 제외
    sheets = pd.read_excel(SAMPLE, sheet_name=None)
    scored = set().union(*[set(d["rfid_id"].astype(str)) for d in sheets.values()])
    lab = dict(zip(S["rfid_id"].astype(str), y))
    T["_rid"] = T["rfid_id"].astype(str)
    ov_mask = (T["_rid"].isin(lab) & ~T["_rid"].isin(scored)).to_numpy()
    ov = T[ov_mask].reset_index(drop=True)
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(S["rfid_id"].astype(str), fold_id))).to_numpy()

    drop = set(shift_feats(S, T, pool_cols))
    add = [c for c in S.columns if c.startswith(("rv_", "pd_"))]
    add = [c for c in add if c not in set(shift_feats(S, T, add))]
    feats = [c for c in pool_cols if c not in drop]
    g = pd.read_csv(C.RESULT_DIR / "feature_ranking_B.csv")
    g = g[g["redundant_folds"] <= 2].sort_values("final_rank")["feature"].tolist()
    print(f"피처 {len(feats)} + 행동 {len(add)} + 지역 사전확률 10 / 연도 차이 제외 {sorted(drop)}", flush=True)
    print(f"2025 검증 {len(ov):,}척 (테스트A·B 시트 {len(scored)}척 제외) ({(time.time() - t0) / 60:.1f}분)", flush=True)

    cat = lambda X, P: pd.concat([X[feats + add].reset_index(drop=True), P.reset_index(drop=True)], axis=1)
    oof = np.zeros((len(y), NCL))
    cross_u = np.zeros((len(ov), NCL))
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), va)
        hyp = [c for c in R["B"]["folds"][f] if V.is_hyp(c) and c not in drop][:30]
        Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
        ovf = ov[ov_fold == f].reset_index(drop=True)
        Ps = [region_prior(Str, y[tr], Str, self_in_ref=True), region_prior(Str, y[tr], Sva),
              region_prior(Str, y[tr], ovf)]
        Xs = [cat(X, P) for X, P in zip([Str, Sva, ovf], Ps)]
        Fs = [x.copy() for x in Xs]  # hier_hyp 2단계용 전체 피처 (자기 자신을 attrs 에 넣으면 재귀 오류)
        Pva, Pov = fit_core(Xs[0], y[tr], Xs[1:], Fs[0], Fs[1:], hyp, "none")
        oof[va], cross_u[ov_fold == f] = Pva, Pov
        print(f"  [all_rp] fold {f}: acc {accuracy_score(y[va], Pva.argmax(1)):.4f} "
              f"({(time.time() - t0) / 60:.1f}분)", flush=True)
    hyp = [c for c in g if V.is_hyp(c) and c not in drop][:30]
    Xa, Xt = cat(S, region_prior(S, y, S, self_in_ref=True)), cat(T, region_prior(S, y, T))
    (Ptest,) = fit_core(Xa, y, [Xt], Xa.copy(), [Xt.copy()], hyp, "none")
    np.save(out_path("step7_oof_all_rp.npy"), oof)
    np.save(out_path("step7_test_all_rp.npy"), Ptest)

    # 기준 r4_rp: 저장된 예측을 같은 표본으로 다시 채점 (cross_unseen 은 저장돼 있지 않아 빈칸)
    base = score(BASE, np.load(C.RESULT_DIR / f"step1_oof_{BASE}.npy"),
                 np.load(C.RESULT_DIR / f"step1_test_{BASE}.npy")[ov_mask], None, y, y_ov, fold_id)
    new = score("all_rp", oof, Ptest[ov_mask], cross_u, y, y_ov, fold_id)
    E = pd.DataFrame([base, new])
    E["d_cv"] = E["cv"] - base["cv"]
    E["d_cross_seen_adj"] = E["cross_seen_adj"] - base["cross_seen_adj"]
    E["n_val25"] = len(ov)
    E.round(4).to_csv(out_path("step7_eval.csv"), index=False, encoding="utf-8-sig")
    print(E.drop(columns="weights").round(4).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
