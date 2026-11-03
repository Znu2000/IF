"""
[STEP12-2] 선별된 조업행태 피처로 3모델 재학습 ─ all_rp(2025 정보 없음) + soak·active·season_prior
  선별(step12_theory --screen)과 같은 cv 로 고른 묶음이라 cv 상승은 약간 낙관적일 수 있음 → fold 부호와 클래스별 F1 을 같이 봄
  cross_* 는 공개 테스트 항적으로 보는 참고값 (시트 선박 제외). 학습·피처 구성에는 2025 를 쓰지 않음
  기존 파일은 건드리지 않음. 출력: results/step12_eval_<시각>.csv, step12_{oof,test}_select.npy
실행: python step12_final.py
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import cv_common as V
import fe_config as C
import step1_experiments as E1
from step1_experiments import fit_core, region_prior
from step3_region import NCL, score
from step4_final import attach
from step11_train26 import SAMPLE, out_path
from step12_theory import BASE_PRIORS, GROUPS, SEASON_PRIORS, attach_theory

SELECT = ["soak", "active"]
STAMP = time.strftime("%m%d_%H%M")


def main():
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    pool_cols = V.feature_pool(S, "B")
    attach(S, "")
    attach(T, "_test")
    attach_theory(S, "")
    attach_theory(T, "_test")
    E1.PRIOR_SETS.clear()
    E1.PRIOR_SETS.update(BASE_PRIORS | SEASON_PRIORS)
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f

    sheets = pd.read_excel(SAMPLE, sheet_name=None, dtype={"rfid_id": str})
    scored = set().union(*[set(d["rfid_id"].astype(str)) for d in sheets.values()])
    lab = dict(zip(S["rfid_id"].astype(str), y))
    T["_rid"] = T["rfid_id"].astype(str)
    ov_mask = (T["_rid"].isin(lab) & ~T["_rid"].isin(scored)).to_numpy()
    ov = T[ov_mask].reset_index(drop=True)
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(S["rfid_id"].astype(str), fold_id))).to_numpy()

    prefixes = ("rv_", "pd_") + tuple(GROUPS[g] for g in SELECT)
    add = [c for c in S.columns if c.startswith(prefixes)]
    g = pd.read_csv(C.RESULT_DIR / "feature_ranking_B.csv")
    g = g[g["redundant_folds"] <= 2].sort_values("final_rank")["feature"].tolist()
    print(f"풀 {len(pool_cols)} + 추가 {len(add)} + 지역 사전확률 {5 * len(E1.PRIOR_SETS)} ({(time.time() - t0) / 60:.1f}분)", flush=True)

    cat = lambda X, P: pd.concat([X[pool_cols + add].reset_index(drop=True), P.reset_index(drop=True)], axis=1)
    base_oof = np.load(C.RESULT_DIR / "step11_oof_all_rp.npy")
    oof = np.zeros((len(y), NCL))
    cross_u = np.zeros((len(ov), NCL))
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), va)
        hyp = [c for c in R["B"]["folds"][f] if V.is_hyp(c)][:30]
        Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
        ovf = ov[ov_fold == f].reset_index(drop=True)
        Ps = [region_prior(Str, y[tr], Str, self_in_ref=True), region_prior(Str, y[tr], Sva), region_prior(Str, y[tr], ovf)]
        Xs = [cat(X, P) for X, P in zip([Str, Sva, ovf], Ps)]
        Fs = [x.copy() for x in Xs]
        Pva, Pov = fit_core(Xs[0], y[tr], Xs[1:], Fs[0], Fs[1:], hyp, "none")
        oof[va], cross_u[ov_fold == f] = Pva, Pov
        print(f"  fold {f}: acc {accuracy_score(y[va], Pva.argmax(1)):.4f} / 기준 {accuracy_score(y[va], base_oof[va].argmax(1)):.4f} "
              f"({(time.time() - t0) / 60:.1f}분)", flush=True)
    hyp = [c for c in g if V.is_hyp(c)][:30]
    Xa, Xt = cat(S, region_prior(S, y, S, self_in_ref=True)), cat(T, region_prior(S, y, T))
    (Ptest,) = fit_core(Xa, y, [Xt], Xa.copy(), [Xt.copy()], hyp, "none")
    np.save(out_path("step12_oof_select.npy"), oof)
    np.save(out_path("step12_test_select.npy"), Ptest)
    E = pd.DataFrame([score("select", oof, Ptest[ov_mask], cross_u, y, y_ov, fold_id),
                      score("all_rp_no25", base_oof, np.load(C.RESULT_DIR / "step11_test_all_rp.npy")[ov_mask], None, y, y_ov, fold_id)])
    for name, O in [("select", oof), ("all_rp_no25", base_oof)]:
        for cl, v in zip(C.CLASSES, f1_score(y, O.argmax(1), average=None, labels=range(NCL))):
            E.loc[E["config"] == name, f"f1_{cl}"] = v
    E.round(4).to_csv(C.RESULT_DIR / f"step12_eval_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(E.drop(columns="weights").round(4).T.to_string())
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
