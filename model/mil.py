"""
[MIL] 다중 인스턴스 학습 ─ 선박 1척 = 출어일(선박-일 L2 피처) 묶음, 라벨은 선박 단위

왜: 선박 단위 평균/표준편차 집계는 "업종이 드러나는 소수의 날"을 정박·이동일과 섞어 희석시킴.
    MIL 은 하루 하나하나를 인코딩한 뒤 attention 으로 정보가 있는 날에 가중치를 줘서 묶음 → 선박 판정.
모델: 일 인코더(MLP) → gated attention pooling (Ilse et al., ICML 2018) ⊕ mean pooling → 분류기
학습: 선박 기준 동일 fold (results/fold_rankings.json), 학습 때 배마다 8~128일 무작위 표본 (길이 증강)
      손실 = class-weighted CE (가중치 √(balanced): CatBoost 실험에서 Balanced 과보정 확인)
출력: results/oof_{pool}_mil.npy, mil_metrics.csv, mil_length_curve.csv, mil_attention.parquet

실행: python mil.py                       (A, B / CPU)
      python mil.py --device cuda --seeds 5
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

import cv_common as V
import fe_config as C
from train_classifiers import ENS_MEMBERS, adjust_lofo, metrics_row

NC = len(C.CLASSES)
DAY_LOC = ["lon_c", "lat_c", "lon_start", "lat_start", "lon_end", "lat_end", "mv_lon_c", "mv_lat_c"]
DAY_DROP = ["rfid_id", "part_dt", "fishing_code"]
LENGTHS = [7, 14, 30, 60]


class MIL(nn.Module):
    def __init__(self, d_in, d=128, p=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.att_v, self.att_u, self.att_w = nn.Linear(d, 64), nn.Linear(d, 64), nn.Linear(64, 1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, NC))

    def forward(self, x, mask):
        h = self.enc(x)                                                      # (B, T, d)
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        a = torch.softmax(a.masked_fill(~mask, -1e9), dim=1)                 # 날짜별 attention
        z_att = (a.unsqueeze(-1) * h).sum(1)
        m = mask.unsqueeze(-1).float()
        z_mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([z_att, z_mean], -1)), a


class Bags:
    """선박별 출어일 인덱스 (starts/counts) 위에서 배치 생성"""

    def __init__(self, Z, starts, counts, dev):
        self.Z, self.starts, self.counts, self.dev = Z, starts, counts, dev

    def batch(self, vids, rng=None, cap=None, n_days=None):
        rows = []
        for v in vids:
            idx = np.arange(self.starts[v], self.starts[v] + self.counts[v])
            m = len(idx)
            if n_days is not None:
                m = min(m, n_days)
            elif rng is not None:  # 학습: 길이 증강 (8 ~ min(cap, 전체) 일)
                hi = min(m, cap)
                m = int(rng.integers(min(8, hi), hi + 1))
            if m < len(idx):
                idx = np.sort((rng or np.random.default_rng(0)).choice(idx, m, replace=False))
            rows.append(idx)
        T = max(len(r) for r in rows)
        x = np.zeros((len(vids), T, self.Z.shape[1]), np.float32)
        mask = np.zeros((len(vids), T), bool)
        for i, r in enumerate(rows):
            x[i, : len(r)] = self.Z[r]
            mask[i, : len(r)] = True
        return torch.from_numpy(x).to(self.dev), torch.from_numpy(mask).to(self.dev), rows


def preprocess(X, train_rows):
    """train 선박의 날짜로만 분위수 변환 적합 → 전체 변환. 결측은 0(=중앙값) + 결측 지시자"""
    fit = train_rows if len(train_rows) <= 200_000 else np.random.default_rng(0).choice(train_rows, 200_000, False)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, subsample=200_000, random_state=0)
    qt.fit(X[fit])
    Z = np.clip(np.nan_to_num(qt.transform(X), nan=0.0), -5, 5).astype(np.float32)
    na = np.isnan(X)
    keep = na[train_rows].mean(0) > 0.01
    return np.hstack([Z, na[:, keep].astype(np.float32)])


def predict(model, bags, vids, n_days=None, reps=1, seed=0):
    model.eval()
    P = np.zeros((len(vids), NC))
    with torch.no_grad():
        for r in range(reps):
            rng = np.random.default_rng(seed + r) if n_days else None
            for i in range(0, len(vids), 256):
                x, m, _ = bags.batch(vids[i:i + 256], rng=rng, n_days=n_days)
                P[i:i + 256] += torch.softmax(model(x, m)[0], -1).cpu().numpy()
    return P / reps


def train_one(bags, tr, y, seed, dev, epochs=80, patience=12, cap=128, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t_in, v_in = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
    model = MIL(bags.Z.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cw = np.sqrt(len(t_in) / (NC * np.bincount(y[t_in], minlength=NC)))
    cw = torch.tensor(cw / cw.mean(), dtype=torch.float32, device=dev)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(t_in)
        for i in range(0, len(order), bs):
            vids = order[i:i + bs]
            x, m, _ = bags.batch(vids, rng=rng, cap=cap)
            loss = Fn.cross_entropy(model(x, m)[0], torch.tensor(y[vids], device=dev), weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = V.macro_f1(y[v_in], predict(model, bags, v_in).argmax(1))
        if f1 > best + 1e-4:
            best, bad = f1, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model, best, ep + 1


def main(a):
    torch.set_num_threads(C.N_JOBS)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    S, y = V.load_ship()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    val_idx = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(val_idx):
        fold_id[v] = f

    D = pd.read_parquet(C.FEAT_DIR / "shipday.parquet")
    vmap = pd.Series(np.arange(len(S)), index=S["rfid_id"].to_numpy())
    D = D[D["rfid_id"].isin(vmap.index)].copy()
    D["_v"] = vmap.loc[D["rfid_id"]].to_numpy()
    D = D.sort_values(["_v", "part_dt"]).reset_index(drop=True)
    vday = D["_v"].to_numpy()
    starts = np.searchsorted(vday, np.arange(len(S)))
    counts = np.bincount(vday, minlength=len(S))
    assert counts.min() > 0, "출어일이 없는 선박 존재"
    print(f"선박 {len(S):,} / 출어일 {len(D):,} / 선박당 일수 중앙값 {np.median(counts):.0f}, 최대 {counts.max()}  device={dev}")

    rows, curves, att_rows = [], [], []
    for pool in a.pools:
        cols = [c for c in D.columns if c not in DAY_DROP + ["_v"] and pd.api.types.is_numeric_dtype(D[c])]
        if pool == "A":
            cols = [c for c in cols if c not in DAY_LOC]
        X = D[cols].to_numpy(np.float64)
        print(f"\n━━ Pool {pool}: 일 피처 {len(cols)}개 ━━")
        P = np.zeros((len(y), NC))
        Pl = {n: np.zeros((len(y), NC)) for n in LENGTHS}
        for f, va in enumerate(val_idx):
            t = time.time()
            tr = np.setdiff1d(np.arange(len(y)), va)
            tr_rows = np.concatenate([np.arange(starts[v], starts[v] + counts[v]) for v in tr])
            bags = Bags(preprocess(X, tr_rows), starts, counts, dev)
            info = []
            for s in range(a.seeds):  # 시드 앙상블
                model, best, ep = train_one(bags, tr, y, seed=100 * f + s, dev=dev, epochs=a.epochs)
                P[va] += predict(model, bags, va) / a.seeds
                for n in LENGTHS:  # 기간 길이별: 배마다 n일만 무작위로 남기고 예측 (3회 평균)
                    Pl[n][va] += predict(model, bags, va, n_days=n, reps=3, seed=7) / a.seeds
                info.append(f"s{s}:{best:.3f}@{ep}ep")
                if s == 0 and pool == a.pools[-1]:  # attention 저장 (해석용)
                    model.eval()
                    with torch.no_grad():
                        for i in range(0, len(va), 256):
                            x, m, rws = bags.batch(va[i:i + 256])
                            att = model(x, m)[1].cpu().numpy()
                            for j, r in enumerate(rws):
                                att_rows.append(pd.DataFrame({"row": r, "att": att[j, : len(r)]}))
            print(f"  fold {f}: val F1 {V.macro_f1(y[va], P[va].argmax(1)):.4f}  ({' '.join(info)}, {time.time() - t:.0f}s)",
                  flush=True)
        np.save(C.RESULT_DIR / f"oof_{pool}_mil.npy", P)
        rows.append(metrics_row("mil", pool, y, P, fold_id))
        rows.append(metrics_row("mil", pool, y, adjust_lofo(P, y, fold_id), fold_id) | {"model": "mil+adj"})
        # 기존 모델과 결합 (late fusion, 단순 평균)
        oof = {m: np.load(p) for m in ENS_MEMBERS + ["catboost"] if (p := C.RESULT_DIR / f"oof_{pool}_{m}.npy").exists()}
        ens = np.mean([oof[m] for m in ENS_MEMBERS if m in oof], axis=0)
        for name, Q in [("ensemble(기존)", ens), ("ensemble+mil", (ens + P) / 2), ("catboost+mil", (oof["catboost"] + P) / 2)]:
            rows.append(metrics_row(name, pool, y, Q, fold_id))
            rows.append(metrics_row(name, pool, y, adjust_lofo(Q, y, fold_id), fold_id) | {"model": name + "+adj"})
        for n in LENGTHS + ["전체"]:
            Q = P if n == "전체" else Pl[n]
            curves.append(dict(pool=pool, n_days=n, macro_f1=V.macro_f1(y, Q.argmax(1))))
        for r in rows[-8:]:
            print(f"  {r['model']:<20s} macro-F1 {r['macro_f1']:.4f}  (fold ±{r['f1_fold_std']:.4f})")
        print("  기간 길이별:", {c["n_days"]: round(c["macro_f1"], 4) for c in curves if c["pool"] == pool})

    M = pd.DataFrame(rows)
    M.round(4).to_csv(C.RESULT_DIR / "mil_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(C.RESULT_DIR / "mil_length_curve.csv", index=False, encoding="utf-8-sig")
    if att_rows:  # 날짜별 attention + 그날의 상태 → 모델이 어떤 날을 보는가
        A = pd.concat(att_rows, ignore_index=True)
        info = D.loc[A["row"].to_numpy(), ["rfid_id", "part_dt", "n_moving_hours", "in_port_ratio", "dist_day_km",
                                           "fish_share", "drift_share", "transit_share"]].reset_index(drop=True)
        A = pd.concat([A, info], axis=1)
        A.to_parquet(C.RESULT_DIR / "mil_attention.parquet", index=False)
        A["moving_day"] = A["n_moving_hours"] >= 1
        A["n"] = A.groupby("rfid_id")["att"].transform("size")
        A["att_rel"] = A["att"] * A["n"]  # 1 = 균등 가중
        print("\nattention (1=균등): 이동일", round(A.loc[A.moving_day, "att_rel"].mean(), 3),
              "/ 비이동일", round(A.loc[~A.moving_day, "att_rel"].mean(), 3))
    show = ["pool", "model", "macro_f1", "f1_fold_std"] + [f"f1_{c}" for c in C.CLASSES] + ["gill_vs_coast_f1"]
    print("\n" + M[show].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=["A", "B"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    main(ap.parse_args())
