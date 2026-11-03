"""
[STEP8] 기록이 거의 없는 배(희소 선박) 따로 처리 ─ 출어일이 0일이라 행동 피처가 전부 빈값인 배
  새 테스트A 의 25%, 학습의 5%(389척). 기존 모델 정확도: 희소 0.52 / 나머지 0.61
  후보 (희소 선박에만 적용, 나머지는 기존 예측 그대로)
    base       : 기존 모델 (r4_rp, all_rp OOF)
    prior_k*   : 모항 주변 학습 선박 k 척의 업종 비율 그대로 (k = 10, 30, 100)
    loc_all    : 위치·지역 사전확률만 쓰는 작은 LightGBM, 학습 선박 전체로 학습 (행동 피처 없음)
    loc_sparse : 같은 피처, 학습의 희소 선박만으로 학습
    blend      : base 와 loc_all 확률 평균
  강건성: 2026 교차검증(fold 마다) + 2025 항적(학습에 있으나 fold 모델은 못 본 배, 테스트 시트 선박 제외) 둘 다 확인
          2025 의 base 는 학습 전체 모델 예측이라 base 쪽에 유리 (보수적 비교)
  기존 파일은 건드리지 않음. 출력: results/step8_sparse_eval.csv, step8_sparse_folds.csv
실행: python step8_sparse.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import cv_common as V
import fe_config as C
from step1_experiments import K_NB, region_prior

TPL = [C.PROJECT_ROOT / f"문제 데이터_수정_{v}" / "테스트셋 및 결과제출 샘플" / "제출샘플.xlsx" for v in ("1027", "1103")]
LOC = ["lon_c", "lat_c", "home_lon", "home_lat", "home_port_km"]
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)
BASES = {"r4_rp": ("step1_oof_r4_rp.npy", "step1_test_r4_rp.npy"),
         "all_rp": ("step7_oof_all_rp.npy", "step7_test_all_rp.npy")}


def sparse_mask(D):
    return D["longest_streak_mean"].isna().to_numpy()


def prior_k(Sref, yref, Q, k, self_in_ref=False):
    import step1_experiments as E1
    old, E1.K_NB = E1.K_NB, k
    try:
        P = region_prior(Sref, yref, Q, self_in_ref=self_in_ref)
    finally:
        E1.K_NB = old
    return P[[f"prior_home_{c}" for c in C.CLASSES]].to_numpy()


def loc_model(Str, ytr, Qs, only_sparse):
    pr = [region_prior(Str, ytr, Str, self_in_ref=True)] + [region_prior(Str, ytr, Q) for Q in Qs]
    X = [pd.concat([D[LOC].reset_index(drop=True), p.reset_index(drop=True)], axis=1) for D, p in zip([Str] + Qs, pr)]
    m = sparse_mask(Str) if only_sparse else np.ones(len(Str), bool)
    clf = lgb.LGBMClassifier(**LGB).fit(X[0][m], ytr[m])
    return [clf.predict_proba(x) for x in X[1:]]


def fill(P):
    """사전확률이 빈 배(모항 없음)는 균등 → argmax 가 연안복합(0)"""
    P = np.nan_to_num(P, nan=0.0)
    P[P.sum(1) == 0] = 1.0 / P.shape[1]
    return P


def main():
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    sp = sparse_mask(S)

    lab = dict(zip(S["rfid_id"].astype(str), y))
    fold_of = {}
    for f, v in enumerate(folds):
        fold_of.update({r: f for r in S["rfid_id"].astype(str).iloc[v]})
    sheet_ids = set()
    for p in TPL:
        for d in pd.read_excel(p, sheet_name=None).values():
            sheet_ids |= set(d["rfid_id"].astype(str))
    T["_rid"] = T["rfid_id"].astype(str)
    m25 = (T["_rid"].isin(lab) & ~T["_rid"].isin(sheet_ids)).to_numpy() & sparse_mask(T)
    Q25 = T[m25].reset_index(drop=True)
    y25 = Q25["_rid"].map(lab).to_numpy()
    f25 = Q25["_rid"].map(fold_of).to_numpy()
    print(f"희소 선박: 학습 {sp.sum()}척 / 2025 검증 {len(Q25)}척 (테스트 시트 {len(sheet_ids)}척 제외)", flush=True)

    cands = ["prior_k10", "prior_k30", "prior_k100", "loc_all", "loc_sparse"]
    oof = {c: np.zeros((len(y), 4)) for c in cands}
    p25 = {c: np.zeros((len(Q25), 4)) for c in cands}
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), va)
        Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
        q = Q25[f25 == f].reset_index(drop=True)
        for k in (10, 30, 100):
            oof[f"prior_k{k}"][va] = prior_k(Str, y[tr], Sva, k)
            p25[f"prior_k{k}"][f25 == f] = prior_k(Str, y[tr], q, k)
        for c, only in (("loc_all", False), ("loc_sparse", True)):
            a, b = loc_model(Str, y[tr], [Sva, q], only)
            oof[c][va], p25[c][f25 == f] = a, b
        print(f"  fold {f} ({(time.time() - t0) / 60:.1f}분)", flush=True)

    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    rows, frows = [], []
    for bname, (fo, ft) in BASES.items():
        Bo = np.load(C.RESULT_DIR / fo)
        Bt = np.load(C.RESULT_DIR / ft)[m25]
        sets = {"base": (Bo, Bt)} | {c: (fill(oof[c]), fill(p25[c])) for c in cands}
        sets["blend"] = ((Bo + oof["loc_all"]) / 2, (Bt + p25["loc_all"]) / 2)
        for name, (O, P) in sets.items():
            full = Bo.argmax(1).copy()
            full[sp] = O[sp].argmax(1)
            rows.append(dict(base=bname, method=name,
                             acc26_sparse=accuracy_score(y[sp], O[sp].argmax(1)),
                             acc26_all=accuracy_score(y, full),
                             acc25_sparse=accuracy_score(y25, P.argmax(1)),
                             pred_dist26=np.bincount(O[sp].argmax(1), minlength=4).tolist()))
            for f in range(len(folds)):
                m = sp & (fold_id == f)
                frows.append(dict(base=bname, method=name, fold=f, acc=accuracy_score(y[m], O[m].argmax(1))))
    E = pd.DataFrame(rows)
    F = pd.DataFrame(frows).pivot_table(index=["base", "method"], columns="fold", values="acc")
    for b in BASES:
        ref = F.loc[(b, "base")]
        E.loc[E.base == b, "folds_better"] = [int(((F.loc[(b, m)] - ref) > 0).sum()) for m in E.loc[E.base == b, "method"]]
    E.round(4).to_csv(C.RESULT_DIR / "step8_sparse_eval.csv", index=False, encoding="utf-8-sig")
    F.round(4).to_csv(C.RESULT_DIR / "step8_sparse_folds.csv", encoding="utf-8-sig")
    print(f"학습 희소 선박 업종 분포 {np.bincount(y[sp], minlength=4).tolist()} / 2025 {np.bincount(y25, minlength=4).tolist()}")
    print(E.round(4).to_string(index=False))
    print(F.round(3).to_string())
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
