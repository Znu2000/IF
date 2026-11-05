"""
[STEP11] 2025 정보 없는 재학습 ─ 피처 구성·모델 학습에 테스트(2025)를 쓰지 않음
  바뀐 점: shift_feats(학습·테스트 분포 비교로 피처 제외) 삭제 → mv_lowfreq_ratio_mean/median 복귀, rv_·pd_ 도 그 필터 없이 전부 사용
  그대로 : fold, fold 별 랭킹(2026 만으로 계산), 모델(fit_core), 지역 사전확률(학습 선박 라벨)
  평가   : cv·cv_adj·cv_macro 는 2026 만. cross_unseen·cross_seen 은 공개 테스트 항적으로 보는 참고값 (시트 선박 제외)
  설정   : r4_rp  = fold 별 상위 k + rv_·pd_ + 지역 사전확률 10
           all_rp = 선박 피처 전체 + rv_·pd_ + 지역 사전확률 10
  기존 파일은 건드리지 않음. 출력: results/step11_eval_<시각>.csv, step11_{oof,test}_<설정>.npy (같은 이름이 있으면 시각을 붙여 저장)
실행: python step11_train26.py [--configs r4_rp all_rp]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import cv_common as V
import fe_config as C
from step1_experiments import fit_core, region_prior
from step3_region import NCL, score
from step4_final import attach

SAMPLE = C.PROJECT_ROOT / "문제 데이터_수정_1103_v2" / "테스트셋 및 결과제출 샘플" / "제출샘플.xlsx"
OLD = {"r4_rp": "step1_{}_r4_rp.npy", "all_rp": "step7_{}_all_rp.npy"}
STAMP = time.strftime("%m%d_%H%M")


def out_path(name):
    p = C.RESULT_DIR / name
    return p.with_name(f"{p.stem}_{STAMP}{p.suffix}") if p.exists() else p


def main(a):
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
    k = R["B"]["k_star"]

    # 참고용 2025 표본: 학습에도 있는 테스트 선박 중 테스트A·B 시트 선박 제외 (학습에는 쓰지 않음)
    sheets = pd.read_excel(SAMPLE, sheet_name=None, dtype={"rfid_id": str})
    scored = set().union(*[set(d["rfid_id"].astype(str)) for d in sheets.values()])
    lab = dict(zip(S["rfid_id"].astype(str), y))
    T["_rid"] = T["rfid_id"].astype(str)
    ov_mask = (T["_rid"].isin(lab) & ~T["_rid"].isin(scored)).to_numpy()
    ov = T[ov_mask].reset_index(drop=True)
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(S["rfid_id"].astype(str), fold_id))).to_numpy()

    add = [c for c in S.columns if c.startswith(("rv_", "pd_"))]
    g = pd.read_csv(C.RESULT_DIR / "feature_ranking_B.csv")
    g = g[g["redundant_folds"] <= 2].sort_values("final_rank")["feature"].tolist()
    print(f"학습 {len(y)}척 / 풀 {len(pool_cols)} + 행동 {len(add)} / 참고 2025 {len(ov):,}척 (시트 {len(scored)}척 제외) "
          f"({(time.time() - t0) / 60:.1f}분)", flush=True)

    cat = lambda X, P, cols: pd.concat([X[cols + add].reset_index(drop=True), P.reset_index(drop=True)], axis=1)
    rows = []
    for name in a.configs:
        oof = np.zeros((len(y), NCL))
        cross_u = np.zeros((len(ov), NCL))
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), va)
            feats = R["B"]["folds"][f][:k] if name == "r4_rp" else pool_cols
            hyp = [c for c in R["B"]["folds"][f] if V.is_hyp(c)][:30]
            Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
            ovf = ov[ov_fold == f].reset_index(drop=True)
            Ps = [region_prior(Str, y[tr], Str, self_in_ref=True), region_prior(Str, y[tr], Sva),
                  region_prior(Str, y[tr], ovf)]
            Xs = [cat(X, P, feats) for X, P in zip([Str, Sva, ovf], Ps)]
            Fs = [cat(X, P, pool_cols) for X, P in zip([Str, Sva, ovf], Ps)]
            Pva, Pov = fit_core(Xs[0], y[tr], Xs[1:], Fs[0], Fs[1:], hyp, "none")
            oof[va], cross_u[ov_fold == f] = Pva, Pov
            print(f"  [{name}] fold {f}: acc {accuracy_score(y[va], Pva.argmax(1)):.4f} "
                  f"({(time.time() - t0) / 60:.1f}분)", flush=True)
        feats = g[:k] if name == "r4_rp" else pool_cols
        hyp = [c for c in g if V.is_hyp(c)][:30]
        Pa, Pt = region_prior(S, y, S, self_in_ref=True), region_prior(S, y, T)
        (Ptest,) = fit_core(cat(S, Pa, feats), y, [cat(T, Pt, feats)], cat(S, Pa, pool_cols), [cat(T, Pt, pool_cols)],
                            hyp, "none")
        np.save(out_path(f"step11_oof_{name}.npy"), oof)
        np.save(out_path(f"step11_test_{name}.npy"), Ptest)
        rows.append(score(f"{name}_no25", oof, Ptest[ov_mask], cross_u, y, y_ov, fold_id))
        rows.append(score(f"{name}_기존", np.load(C.RESULT_DIR / OLD[name].format("oof")),
                          np.load(C.RESULT_DIR / OLD[name].format("test"))[ov_mask], None, y, y_ov, fold_id))
        E = pd.DataFrame(rows)
        E["n_val25"] = len(ov)
        E.round(4).to_csv(C.RESULT_DIR / f"step11_eval_{STAMP}.csv", index=False, encoding="utf-8-sig")
        print(E.drop(columns="weights").round(4).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["r4_rp", "all_rp"])
    main(ap.parse_args())
