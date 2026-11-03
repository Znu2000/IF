"""
[STEP3] 지역 정보 확장 ─ 1단계 최종(prior_now_noshift: 모항·조업지 중심 주변 30척 업종 비율)에 지역 피처를 더해 비교
  A multik  : 모항·조업지 주변 10척 / 100척 업종 비율 추가 (항구 단위 + 넓은 지역 단위)
  B port    : 모항 격자(0.02°, 약 2km)가 같은 학습 선박의 업종 비율 (어촌계 단위, 전체 비율로 평활) + 그 격자 선박 수
  C grid    : 그 배가 조업(바다 위 정지·표류·조업 상태)한 0.05°(약 5km) 격자들에서, 같은 격자를 쓴 학습 선박들의
              업종 비율을 그 배의 조업 시간 비중으로 가중 평균 (중심점 1개로 줄이지 않고 어장 전체 사용)
  D distw   : 주변 100척을 거리 가중(가까울수록 큼, 반경 5km 커널)으로 낸 업종 비율
  all       : A+B+C+D
  누수 방지 : 모든 비율은 학습 fold 선박만 참조, 학습 선박 자신의 기여는 빼고 계산 (1단계와 같음)
  평가      : step1 과 같은 fold·지표 (micro-F1). 기준값은 1단계 저장 결과(step1_*_prior_now_noshift)
  입력      : features/vessel_cells.parquet (rfid_id, cell, hours ─ 로컬에서 anom_hourly 로 미리 만든 것, 선박 ID 포함)
  출력      : results/step1_oof_r3_{config}.npy, step1_test_r3_{config}.npy, step1_results.csv 에 행 추가
실행: python step3_region.py [--configs multik port grid distw all] [--build_cells]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.neighbors import BallTree

import cv_common as V
import fe_config as C
import train_classifiers as TC
from step1_experiments import PRIOR_SETS, fit_core, region_prior, shift_feats

BASE = "prior_now_noshift"
PORT_DEG, GRID_DEG, ALPHA, BW_KM = 0.02, 0.05, 5.0, 5.0
SEA_STATES = (1, 2, 3)  # SEA_STOP, DRIFT, FISH (fe_aggregate)
CONFIGS = {"multik": "A", "port": "B", "grid": "C", "distw": "D", "all": "ABCD"}
NCL = len(C.CLASSES)


def build_cells():
    """학습·테스트 선박별 조업 격자 시간 (로컬 1회)"""
    out = []
    for f in ["anom_hourly.parquet", "anom_hourly_test.parquet"]:
        H = pd.read_parquet(C.FEAT_DIR / f, columns=["rfid_id", "state", "lon_c", "lat_c", "at_sea"])
        H = H[(H["at_sea"] == 1) & H["state"].isin(SEA_STATES)].dropna(subset=["lon_c", "lat_c"])
        cell = (np.floor(H["lat_c"] / GRID_DEG).astype(int) * 10000 + np.floor(H["lon_c"] / GRID_DEG).astype(int))
        g = H.assign(cell=cell).groupby(["rfid_id", "cell"]).size().rename("hours").reset_index()
        out.append(g.assign(split="test" if "test" in f else "train"))
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(C.FEAT_DIR / "vessel_cells.parquet", index=False)
    print(f"vessel_cells.parquet: {len(D):,}행, 격자 {D['cell'].nunique():,}개")


def onehot(y):
    return np.eye(NCL)[y]


def knn_share(S_ref, y_ref, Q, k, self_in_ref, weighted=False):
    out = {}
    for name, cols in PRIOR_SETS.items():
        ref = S_ref[cols].to_numpy(float)
        ok = ~np.isnan(ref).any(1)
        tree, yr = BallTree(np.radians(ref[ok]), metric="haversine"), y_ref[ok]
        q = Q[cols].to_numpy(float)
        qok = ~np.isnan(q).any(1)
        P = np.full((len(Q), NCL), np.nan)
        if qok.any():
            d, i = tree.query(np.radians(q[qok]), k=k + int(self_in_ref))
            if self_in_ref:
                d, i = d[:, 1:], i[:, 1:]
            w = np.exp(-(d * 6371.0) / BW_KM) + 1e-9 if weighted else np.ones_like(d)
            P[qok] = np.einsum("nk,nkc->nc", w, onehot(yr[i])) / w.sum(1, keepdims=True)
        tag = f"{name}_dw" if weighted else f"{name}_k{k}"
        for c, cl in enumerate(C.CLASSES):
            out[f"rp_{tag}_{cl}"] = P[:, c]
    return pd.DataFrame(out, index=Q.index)


def port_share(S_ref, y_ref, Q, self_in_ref):
    key = lambda X: (np.floor(X["home_lat"] / PORT_DEG) * 100000 + np.floor(X["home_lon"] / PORT_DEG)).to_numpy()
    kr, kq = key(S_ref), key(Q)
    g = onehot(y_ref).mean(0)
    Y = pd.DataFrame(onehot(y_ref)).groupby(kr).sum()
    M = Y.reindex(kq).to_numpy()
    n = np.nan_to_num(M.sum(1))
    M = np.nan_to_num(M)
    if self_in_ref:
        M, n = M - onehot(y_ref), n - 1
    P = (M + ALPHA * g) / (n[:, None] + ALPHA)
    P[np.isnan(kq)] = np.nan
    out = {f"rp_port_{cl}": P[:, c] for c, cl in enumerate(C.CLASSES)}
    out["rp_port_n"] = np.where(np.isnan(kq), np.nan, n)
    return pd.DataFrame(out, index=Q.index)


class Cells:
    """선박 × 격자 조업 시간 비중 (행 합 1)"""

    def __init__(self, D, rids):
        D = D[D["rfid_id"].isin(rids)]
        self.cells = {c: j for j, c in enumerate(np.unique(D["cell"]))}
        pos = {r: i for i, r in enumerate(rids)}
        W = sparse.csr_matrix((D["hours"].to_numpy(float), (D["rfid_id"].map(pos).to_numpy(),
                                                            D["cell"].map(self.cells).to_numpy())),
                              shape=(len(rids), len(self.cells)))
        s = np.asarray(W.sum(1)).ravel()
        self.W = sparse.diags(np.where(s > 0, 1 / np.maximum(s, 1e-9), 0)) @ W


def grid_share(Wref, y_ref, Wq, self_in_ref):
    """질의 선박 격자 비중 × (격자별 학습 선박 업종 질량, 전체 비율로 평활) ─ 자기 기여 제외"""
    Yr = onehot(y_ref)
    M = np.asarray((Wref.T @ Yr))                     # 격자 × 업종 질량
    tot = np.asarray(Wref.sum(0)).ravel()             # 격자 총 질량
    g = Yr.mean(0)
    Q = Wq.tocoo()
    w, i, j = Q.data, Q.row, Q.col
    num, den = M[j] + ALPHA * g, tot[j] + ALPHA
    if self_in_ref:
        num, den = num - w[:, None] * Yr[i], den - w
    P = np.zeros((Wq.shape[0], NCL))
    np.add.at(P, i, w[:, None] * num / den[:, None])
    has = np.asarray(Wq.sum(1)).ravel() > 0
    P[~has] = np.nan
    out = {f"rp_grid_{cl}": P[:, c] for c, cl in enumerate(C.CLASSES)}
    out["rp_grid_ncell"] = np.asarray((Wq > 0).sum(1)).ravel()
    return pd.DataFrame(out)


def extra(parts, Sref, yref, Q, Wref, Wq, self_in_ref):
    F = [region_prior(Sref, yref, Q, self_in_ref=self_in_ref).reset_index(drop=True)]  # 1단계 k=30 (기준)
    if "A" in parts:
        F += [knn_share(Sref, yref, Q, k, self_in_ref).reset_index(drop=True) for k in (10, 100)]
    if "B" in parts:
        F.append(port_share(Sref, yref, Q, self_in_ref).reset_index(drop=True))
    if "C" in parts:
        F.append(grid_share(Wref, yref, Wq, self_in_ref))
    if "D" in parts:
        F.append(knn_share(Sref, yref, Q, 100, self_in_ref, weighted=True).reset_index(drop=True))
    return pd.concat(F, axis=1)


def main(a):
    if a.build_cells:
        return build_cells()
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
    ov_mask = T["_rid"].isin(lab).to_numpy()
    ov = T[ov_mask].reset_index(drop=True)
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(S["rfid_id"].astype(str), fold_id))).to_numpy()
    drop = set(shift_feats(S, T, pool_cols))
    g = pd.read_csv(C.RESULT_DIR / "feature_ranking_B.csv")
    g = g[g["redundant_folds"] <= 2].sort_values("final_rank")["feature"].tolist()
    D = pd.read_parquet(C.FEAT_DIR / "vessel_cells.parquet")
    Ws = Cells(D[D["split"] == "train"], S["rfid_id"].to_numpy())
    Wt_al = _align(Cells(D[D["split"] == "test"], T["rfid_id"].to_numpy()), Ws)  # 테스트 격자 → 학습 격자 번호
    Ws.W = sparse.hstack([Ws.W, sparse.csr_matrix((Ws.W.shape[0], 1))]).tocsr()  # 열 수를 맞추는 빈 열
    Wov = Wt_al[np.flatnonzero(ov_mask)]
    print(f"학습 {len(y)}척, 테스트 {len(T)}척(학습에도 있는 {len(ov)}척), 격자 {len(Ws.cells):,}개 "
          f"({(time.time() - t0) / 60:.1f}분)", flush=True)

    rows = []
    for name in a.configs:
        parts = CONFIGS[name]
        cfg = f"r3_{name}"
        oof = np.zeros((len(y), NCL))
        cross_u = np.zeros((len(ov), NCL))
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), va)
            feats = [c for c in R["B"]["folds"][f] if c not in drop][:k]
            hyp = [c for c in R["B"]["folds"][f] if V.is_hyp(c) and c not in drop][:30]
            Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
            ovf_i = np.flatnonzero(ov_fold == f)
            ovf = ov.iloc[ovf_i].reset_index(drop=True)
            Wtr = Ws.W[tr]
            E = [extra(parts, Str, y[tr], Str, Wtr, Wtr, True), extra(parts, Str, y[tr], Sva, Wtr, Ws.W[va], False),
                 extra(parts, Str, y[tr], ovf, Wtr, Wov[ovf_i], False)]
            cat = lambda X, e: pd.concat([X.reset_index(drop=True), e], axis=1)
            Xs = [cat(X[feats], e) for X, e in zip([Str, Sva, ovf], E)]
            Fs = [cat(X[pool_cols], e) for X, e in zip([Str, Sva, ovf], E)]
            Pva, Pov = fit_core(Xs[0], y[tr], Xs[1:], Fs[0], Fs[1:], hyp, "none")
            oof[va], cross_u[ov_fold == f] = Pva, Pov
            print(f"  [{name}] fold {f}: acc {accuracy_score(y[va], Pva.argmax(1)):.4f} "
                  f"({(time.time() - t0) / 60:.1f}분)", flush=True)
        feats = [c for c in g if c not in drop][:k]
        hyp = [c for c in g if V.is_hyp(c) and c not in drop][:30]
        Ea, Et = extra(parts, S, y, S, Ws.W, Ws.W, True), extra(parts, S, y, T, Ws.W, Wt_al, False)
        cat = lambda X, e: pd.concat([X.reset_index(drop=True), e], axis=1)
        (Ptest,) = fit_core(cat(S[feats], Ea), y, [cat(T[feats], Et)], cat(S[pool_cols], Ea), [cat(T[pool_cols], Et)],
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


def _align(Wq_obj, Wref_obj):
    """질의 격자 행렬을 참조 격자 번호로 맞춤. 참조에 없는 격자는 마지막 빈 열 하나로 모음 (참조 질량 0 → 전체 비율)"""
    inv = np.array(list(Wq_obj.cells))
    Q = Wq_obj.W.tocoo()
    n = len(Wref_obj.cells)
    j = np.array([Wref_obj.cells.get(c, n) for c in inv[Q.col]], dtype=int)
    return sparse.csr_matrix((Q.data, (Q.row, j)), shape=(Q.shape[0], n + 1))


def score(cfg, O, Pseen, CU, y, y_ov, fold_id):
    w = TC.fit_class_weights(O, y, score=accuracy_score)
    adj = TC.adjust_lofo(O, y, fold_id, score=accuracy_score)
    return dict(config=cfg, cv=accuracy_score(y, O.argmax(1)), cv_adj=accuracy_score(y, adj.argmax(1)),
                cv_macro=V.macro_f1(y, O.argmax(1)),
                cross_unseen=accuracy_score(y_ov, CU.argmax(1)) if CU is not None else np.nan,
                cross_seen=accuracy_score(y_ov, Pseen.argmax(1)),
                cross_seen_adj=accuracy_score(y_ov, (Pseen * w).argmax(1)),
                weights=json.dumps([round(float(v), 3) for v in w]))


def save(rows):
    p = C.RESULT_DIR / "step1_results.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["config"])
    new = pd.DataFrame(rows)
    pd.concat([old[~old["config"].isin(new["config"])], new]).round(4).to_csv(p, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--build_cells", action="store_true")
    main(ap.parse_args())
