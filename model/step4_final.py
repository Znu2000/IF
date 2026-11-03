"""
[STEP4 최종] 행동 피처(step4_behavior.py)를 1단계 최종 앙상블(핵심 3모델)에 넣어 제출 후보 생성
  기준   : prior_now_noshift (fold 별 상위 k + 주변 30척 업종 비율, 연도 차이 피처 제외)
  설정   : revisit (재방문 간격), rp (재방문 + 출어 주기), rpd (재방문 + 주기 + 밀도 군집 어장)
  평가   : step3_region 과 같은 score (cv, cv_adj, cross_seen_adj ...), step1_results.csv 에 r4_<설정> 행 추가
  출력   : results/step1_{oof,test}_r4_<설정>.npy → python apply_step1.py --config r4_<설정> --base submission_combo_blank.xlsx --out submission_final.xlsx
실행: python step4_final.py [--configs revisit rp rpd]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import cv_common as V
import fe_config as C
from step1_experiments import fit_core, region_prior, shift_feats
from step3_region import BASE, NCL, save, score

GROUPS = {"revisit": "rv_", "period": "pd_", "dbscan": "db_", "congest": "cg_"}
CONFIGS = {"revisit": ["revisit"], "rp": ["revisit", "period"], "rpd": ["revisit", "period", "dbscan"]}


def attach(D, tag):
    B = pd.read_parquet(C.FEAT_DIR / f"behav_feats{tag}.parquet")
    B["rfid_id"] = B["rfid_id"].astype(str)
    m = D[["rfid_id"]].astype(str).merge(B, on="rfid_id", how="left").drop(columns="rfid_id")
    for c in m:
        D[c] = m[c].to_numpy()


def main(a):
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    pool_cols = V.feature_pool(S, "B")  # 행동 피처 붙이기 전 기본 피처 (설정별 추가분만 add 로 넣음)
    attach(S, "")
    attach(T, "_test")
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    k = R["B"]["k_star"]
    lab = dict(zip(S["rfid_id"].astype(str), y))
    T["_rid"] = T["rfid_id"].astype(str)
    ov_mask = T["_rid"].isin(lab).to_numpy()
    ov = T[ov_mask].reset_index(drop=True)
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(S["rfid_id"].astype(str), fold_id))).to_numpy()
    drop = set(shift_feats(S, T, pool_cols))
    g = pd.read_csv(C.RESULT_DIR / "feature_ranking_B.csv")
    g = g[g["redundant_folds"] <= 2].sort_values("final_rank")["feature"].tolist()
    print(f"학습 {len(y)}척, 테스트 {len(T)}척(학습에도 있는 {len(ov)}척) ({(time.time() - t0) / 60:.1f}분)", flush=True)

    rows = []
    for name in a.configs:
        add = [c for grp in CONFIGS[name] for c in S.columns if c.startswith(GROUPS[grp])]
        add = [c for c in add if c not in set(shift_feats(S, T, add))]
        cfg = f"r4_{name}"
        cat = lambda X, P, cols: pd.concat([X[cols + add].reset_index(drop=True), P.reset_index(drop=True)], axis=1)
        oof = np.zeros((len(y), NCL))
        cross_u = np.zeros((len(ov), NCL))
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), va)
            feats = [c for c in R["B"]["folds"][f] if c not in drop][:k]
            hyp = [c for c in R["B"]["folds"][f] if V.is_hyp(c) and c not in drop][:30]
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
        feats = [c for c in g if c not in drop][:k]
        hyp = [c for c in g if V.is_hyp(c) and c not in drop][:30]
        Pa, Pt = region_prior(S, y, S, self_in_ref=True), region_prior(S, y, T)
        (Ptest,) = fit_core(cat(S, Pa, feats), y, [cat(T, Pt, feats)], cat(S, Pa, pool_cols), [cat(T, Pt, pool_cols)],
                            hyp, "none")
        np.save(C.RESULT_DIR / f"step1_oof_{cfg}.npy", oof)
        np.save(C.RESULT_DIR / f"step1_test_{cfg}.npy", Ptest)
        rows.append(score(cfg, oof, Ptest[ov_mask], cross_u, y, y_ov, fold_id))
        print({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in rows[-1].items()}, flush=True)
        save(rows)
    base = score(BASE, np.load(C.RESULT_DIR / f"step1_oof_{BASE}.npy"),
                 np.load(C.RESULT_DIR / f"step1_test_{BASE}.npy")[ov_mask], None, y, y_ov, fold_id)
    E = pd.DataFrame([base] + rows)
    E["d_cross_seen_adj"] = E["cross_seen_adj"] - base["cross_seen_adj"]
    E["d_cv"] = E["cv"] - base["cv"]
    print(E.drop(columns="weights").round(4).to_string(index=False))
    ok = E[(E["d_cross_seen_adj"] >= 0.005) & (E["d_cv"] >= 0)]
    print("채택 기준(cross_seen_adj +0.005 이상, cv 하락 없음) 통과:",
          ok.sort_values("cross_seen_adj", ascending=False)["config"].tolist() or "없음 → 기존 유지")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    main(ap.parse_args())
