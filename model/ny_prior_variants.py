"""
[NY-6] 지역 사전확률(진우 step1) 만드는 방식 비교 ─ 나연
  기준: 피처 전체 − 연도 차이 2개 + 지역 사전확률(k=30, 모항·조업지) / LightGBM 가중 없음 (val25 0.604)
  바꿔 보는 것
    · 이웃 수 k        : 10 / 30 / 50 / 100
    · 다중 규모        : k=10 과 k=50 을 같이
    · 거리 가중 커널   : 가까운 배일수록 크게 (가우시안, 대역폭 5km / 15km)
    · 항구 단위        : 모항에서 가장 가까운 공용 항구 칸이 같은 배들의 업종 비율 (전체 비율로 평활)
    · 기준점           : 모항만 / 조업지만
  규칙: 학습 2026 만, 이웃도 그 fold 학습 선박만 (학습 선박 자신은 제외). 평가 = 검증 선박 2025 항적 정확도
실행: python ny_prior_variants.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import BallTree

import cv_common as V
import fe_config as C
from ny_common import LGB, load

R_KM = 6371.0
SETS = {"home": ["home_lat", "home_lon"], "fish": ["fishing_lat", "fishing_lon"]}
SHIFT = ["mv_lowfreq_ratio_mean", "mv_lowfreq_ratio_median"]


def knn_prior(S_ref, y_ref, Q, self_in, name, cols, k=30, bw=None, tag=""):
    ref = S_ref[cols].to_numpy(float)
    ok = ~np.isnan(ref).any(1)
    tree, yr = BallTree(np.radians(ref[ok]), metric="haversine"), y_ref[ok]
    q = Q[cols].to_numpy(float)
    qok = ~np.isnan(q).any(1)
    P, dist = np.full((len(Q), 4), np.nan), np.full(len(Q), np.nan)
    d, i = tree.query(np.radians(q[qok]), k=k + int(self_in))
    if self_in:
        d, i = d[:, 1:], i[:, 1:]
    dk = d * R_KM
    w = np.ones_like(dk) if bw is None else np.exp(-(dk / bw) ** 2) + 1e-6
    P[qok] = np.stack([((yr[i] == c) * w).sum(1) / w.sum(1) for c in range(4)], 1)
    dist[qok] = dk.mean(1)
    out = {f"pr{tag}_{name}_{c}": P[:, j] for j, c in enumerate(C.CLASSES)}
    out[f"pr{tag}_{name}_dist"] = dist
    if bw is not None:
        out[f"pr{tag}_{name}_neff"] = np.where(qok, 0, np.nan)
        out[f"pr{tag}_{name}_neff"][qok] = w.sum(1)
    return pd.DataFrame(out, index=Q.index)


def port_prior(S_ref, y_ref, Q, self_in, ports, m=10):
    tree = BallTree(np.radians(ports[["lat", "lon"]].to_numpy()), metric="haversine")

    def cell(D):
        x = D[["home_lat", "home_lon"]].to_numpy(float)
        x = np.nan_to_num(x)
        return tree.query(np.radians(x), k=1)[1][:, 0]

    cr, cq = cell(S_ref), cell(Q)
    glob = np.bincount(y_ref, minlength=4) / len(y_ref)
    cnt = np.zeros((len(ports), 4))
    np.add.at(cnt, (cr, y_ref), 1)
    c = cnt[cq].copy()
    if self_in:  # 학습 선박: 자기 라벨 빼기
        c[np.arange(len(Q)), y_ref] -= 1
    n = c.sum(1, keepdims=True)
    P = (c + m * glob) / (n + m)
    out = {f"prport_{cl}": P[:, j] for j, cl in enumerate(C.CLASSES)}
    out["prport_n"] = n[:, 0]
    return pd.DataFrame(out, index=Q.index)


def make(kind, S_ref, y_ref, Q, self_in, ports):
    parts = []
    if kind == "k30+port":
        parts = [knn_prior(S_ref, y_ref, Q, self_in, n, c, k=30) for n, c in SETS.items()]
        parts.append(port_prior(S_ref, y_ref, Q, self_in, ports))
    elif kind.startswith("k"):  # k30, k10, ...
        k = int(kind[1:])
        parts = [knn_prior(S_ref, y_ref, Q, self_in, n, c, k=k) for n, c in SETS.items()]
    elif kind == "multi":
        parts = [knn_prior(S_ref, y_ref, Q, self_in, n, c, k=k, tag=str(k)) for n, c in SETS.items() for k in (10, 50)]
    elif kind.startswith("bw"):
        bw = float(kind[2:])
        parts = [knn_prior(S_ref, y_ref, Q, self_in, n, c, k=150, bw=bw) for n, c in SETS.items()]
    elif kind == "port":
        parts = [port_prior(S_ref, y_ref, Q, self_in, ports)]
    elif kind == "home_only":
        parts = [knn_prior(S_ref, y_ref, Q, self_in, "home", SETS["home"])]
    elif kind == "fish_only":
        parts = [knn_prior(S_ref, y_ref, Q, self_in, "fish", SETS["fish"])]
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=Q.index)


KINDS = {"사전확률 없음": "none", "k=30 (진우 step1)": "k30", "k=10": "k10", "k=50": "k50", "k=100": "k100",
         "다중 규모 k=10+50": "multi", "거리 가중 5km": "bw5", "거리 가중 15km": "bw15",
         "항구 단위": "port", "k=30 + 항구 단위": "k30+port", "모항만 k=30": "home_only", "조업지만 k=30": "fish_only"}


def main():
    t0 = time.time()
    S, T25, _ = load()
    S, T25 = S.reset_index(drop=True), T25.reset_index(drop=True)
    yo = {c: i for i, c in enumerate(C.CLASSES)}
    y, y25 = S["fishing_code"].map(yo).to_numpy(), T25["fishing_code"].map(yo).to_numpy()
    F = V.folds(S, y)
    ports = pd.read_csv(C.FEAT_DIR / "port_cells.csv")
    fs = [c for c in V.feature_pool(S, "B") if c in T25.columns and c not in SHIFT + ["year"]]
    rows, oof = [], {}
    for name, kind in KINDS.items():
        P26, P25 = np.zeros((len(S), 4)), np.zeros((len(T25), 4))
        for tr, va in F:
            Str, Sva = S.iloc[tr], S.iloc[va]
            mv = T25["rfid_id"].isin(Sva["rfid_id"]).to_numpy()
            Q25 = T25[mv]
            Xtr, Xva, X25 = Str[fs], Sva[fs], Q25[fs]
            if kind != "none":
                Xtr = Xtr.join(make(kind, Str, y[tr], Str, True, ports))
                Xva = Xva.join(make(kind, Str, y[tr], Sva, False, ports))
                X25 = X25.join(make(kind, Str, y[tr], Q25, False, ports))
            m = lgb.LGBMClassifier(**(LGB | dict(class_weight=None))).fit(Xtr, y[tr])
            P26[va], P25[mv] = m.predict_proba(Xva), m.predict_proba(X25)
        p26, p25 = P26.argmax(1), P25.argmax(1)
        rows.append(dict(variant=name, acc_cv26=accuracy_score(y, p26), acc_val25=accuracy_score(y25, p25),
                         macro_val25=f1_score(y25, p25, average="macro"),
                         **{f"f1_{c}_25": v for c, v in zip(C.CLASSES, f1_score(y25, p25, average=None))}))
        oof[name] = P25
        print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False, header=len(rows) == 1),
              f" ({(time.time() - t0) / 60:.1f}분)", flush=True)
    rng = np.random.default_rng(0)
    base = (oof["k=30 (진우 step1)"].argmax(1) == y25).astype(float)
    idx = [rng.integers(0, len(y25), len(y25)) for _ in range(1000)]
    for r in rows:
        d = (oof[r["variant"]].argmax(1) == y25) - base
        bs = [d[i].mean() for i in idx]
        r["d_vs_k30"], r["ci"] = d.mean(), f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
    R = pd.DataFrame(rows)
    R.round(4).to_csv(C.RESULT_DIR / "ny_prior_variants.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(C.RESULT_DIR / "ny_prior_variants_oof.npz", y=y25, **{f"v{i}": v for i, v in enumerate(oof.values())})
    print("\n", R[["variant", "acc_cv26", "acc_val25", "d_vs_k30", "ci", "macro_val25", "f1_통발_25"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
