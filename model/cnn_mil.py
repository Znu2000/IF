"""
[CNN-MIL] 하루 항적(24시간 × 시간 피처)을 1D-CNN 으로 인코딩 → 출어일 attention 으로 선박 판정  (GPU 서버용)

왜: 지금 모델은 하루를 평균·분위수로 요약해서 "직선 투망(B)·선회(D)·표류-재배치·하루 리듬(E)" 같은
    시간 순서와 모양을 잃음. 하루 24시간을 순서 그대로 1D-CNN 에 넣어 모양을 직접 배우게 함.
    여러 날 중 업종이 드러나는 날에 가중치를 주는 gated attention MIL (Ilse et al., 2018) 은 기존 mil.py 와 같음.
입력: features/anom_hourly.parquet (학습) / anom_hourly_test.parquet (테스트)  ← anomaly_prep.py 결과
      features/ship_features.parquet (라벨·선박 순서), results/fold_rankings.json (기존과 같은 5 fold)
      절대 위치(위경도)는 쓰지 않음 (항구·모항까지 거리만 사용)
학습: 선박 단위 5-fold (기존 OOF 와 같은 fold) × seed, 손실 = 일반 CE (리더보드 micro-F1 기준)
      학습 때 배마다 출어일 최대 48일 무작위 표본, 각 fold 학습 선박의 10% 로 조기 종료
출력: results/oof_A_cnnmil.npy, oof_B_cnnmil.npy (같은 값, 앙상블 합류용)
      results/cnnmil_test.csv (테스트 선박별 4업종 확률 = fold 모델 평균) / results/cnnmil_metrics.json

실행: python cnn_mil.py --device cuda --seeds 3        (서버, 약 1~2시간 예상)
      python cnn_mil.py --smoke                         (로컬 동작 확인: 400척, 1 fold, 2 epoch)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import accuracy_score, f1_score

import fe_config as C

NC = len(C.CLASSES)
HOUR_FEATS = ["ks_speed_med", "sp_mean", "sp_max", "sp_std", "frac_stop", "frac_fast", "ks_frac_drift",
              "heading_R", "ks_heading_R", "turn_mean", "uturn_ratio", "loops", "straightness", "rg_m", "dist_m",
              "n_stays", "stay_frac", "heading_autocorr", "drift_cycles", "port_km", "home_km", "moving",
              "n_points", "jump_prev_km"]
LOG_FEATS = ["sp_max", "rg_m", "dist_m", "port_km", "home_km", "n_points", "jump_prev_km", "loops", "n_stays"]
N_STATE = 5


def build(tag, ids=None, stats=None):
    """선박-일 텐서 X (일수, 24, F), 선박별 일 범위"""
    H = pd.read_parquet(C.FEAT_DIR / f"anom_hourly{tag}.parquet",
                        columns=["rfid_id", "part_dt", "hour", "state"] + HOUR_FEATS)
    if ids is not None:
        H = H[H["rfid_id"].isin(ids)]
    H = H.sort_values(["rfid_id", "part_dt", "hour"]).reset_index(drop=True)
    F = H[HOUR_FEATS].astype(np.float32)
    F[LOG_FEATS] = np.log1p(F[LOG_FEATS].clip(lower=0))
    if stats is None:
        stats = (F.mean().to_numpy(np.float32), F.std().to_numpy(np.float32) + 1e-6)
    F = np.clip(((F.to_numpy(np.float32) - stats[0]) / stats[1]), -5, 5)
    F = np.nan_to_num(F)
    st = np.eye(N_STATE, dtype=np.float32)[H["state"].clip(0, N_STATE - 1).to_numpy()]
    feat = np.concatenate([F, st, np.ones((len(H), 1), np.float32)], 1)  # 마지막 채널 = 기록 있음
    day = H.groupby(["rfid_id", "part_dt"], sort=False).ngroup().to_numpy()
    X = np.zeros((day.max() + 1, 24, feat.shape[1]), np.float16)
    X[day, H["hour"].to_numpy().clip(0, 23)] = feat
    d = H.drop_duplicates(["rfid_id", "part_dt"])["rfid_id"].to_numpy()
    vid, starts, counts = np.unique(d, return_index=True, return_counts=True)
    return X, dict(zip(vid, zip(starts, counts))), stats


class DayEnc(nn.Module):
    """하루 24시간 시퀀스 → 128차원. 팽창 합성곱으로 1·3·7시간 단위 패턴(투망·선회·이동 덩어리)"""

    def __init__(self, f, d=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(f, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=4, dilation=4), nn.GELU())
        self.out = nn.Sequential(nn.Linear(256, d), nn.LayerNorm(d))

    def forward(self, x):                      # x (N, 24, f)
        m = x[..., -1:].transpose(1, 2)        # 기록 있는 시간만 풀링
        h = self.conv(x.transpose(1, 2))
        mean = (h * m).sum(-1) / m.sum(-1).clamp(min=1)
        mx = h.masked_fill(m == 0, -1e4).amax(-1)
        return self.out(torch.cat([mean, mx], -1))


class CNNMIL(nn.Module):
    def __init__(self, f, d=128, p=0.2):
        super().__init__()
        self.enc = DayEnc(f, d)
        self.att_v, self.att_u, self.att_w = nn.Linear(d, 64), nn.Linear(d, 64), nn.Linear(64, 1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, NC))

    def forward(self, x, mask):                # x (B, T, 24, f)
        B, T = x.shape[:2]
        h = self.enc(x.reshape(B * T, *x.shape[2:])).reshape(B, T, -1)
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        a = torch.softmax(a.masked_fill(~mask, -1e9), 1)
        z_att = (a.unsqueeze(-1) * h).sum(1)
        mm = mask.unsqueeze(-1).float()
        z_mean = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
        return self.head(torch.cat([z_att, z_mean], -1))


def make_batch(X, rng_map, vids, dev, cap, rng=None):
    sel = []
    for v in vids:
        s, c = rng_map.get(v, (0, 0))
        idx = np.arange(s, s + c)
        if c > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False)) if rng is not None else idx[np.linspace(0, c - 1, cap).astype(int)]
        sel.append(idx)
    T = max(1, max(len(i) for i in sel))
    x = np.zeros((len(vids), T, *X.shape[1:]), np.float32)
    mask = np.zeros((len(vids), T), bool)
    for b, i in enumerate(sel):
        if len(i):
            x[b, :len(i)] = X[i]
            mask[b, :len(i)] = True
        else:
            mask[b, 0] = True                  # 기록 없는 배: 빈 하루 1개
    return torch.from_numpy(x).to(dev), torch.from_numpy(mask).to(dev)


def predict(model, X, rng_map, vids, dev, cap=256, bs=16):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(vids), bs):
            x, m = make_batch(X, rng_map, vids[b:b + bs], dev, cap)
            out.append(torch.softmax(model(x, m), -1).float().cpu().numpy())
    return np.concatenate(out)


def train_one(X, rng_map, ids, y, tr, dev, seed, epochs, cap=48, bs=32, patience=6):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = rng.permutation(tr)
    n_es = max(1, len(tr) // 10)
    es, fit = tr[:n_es], tr[n_es:]
    model = CNNMIL(X.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(fit)
        for b in range(0, len(order), bs):
            i = order[b:b + bs]
            x, m = make_batch(X, rng_map, ids[i], dev, cap, rng)
            loss = Fn.cross_entropy(model(x, m), torch.tensor(y[i], device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = accuracy_score(y[es], predict(model, X, rng_map, ids[es], dev).argmax(1))
        if acc > best:
            best, bad, best_state = acc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1, best


def main(a):
    t0 = time.time()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code"])
    S = S[S["fishing_code"].isin(C.CLASSES)].reset_index(drop=True)  # cv_common.load_ship 과 같은 순서
    y = S["fishing_code"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()
    ids = S["rfid_id"].to_numpy()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    if a.smoke:
        keep = np.random.default_rng(0).choice(len(S), 400, replace=False)
        folds = [np.intersect1d(folds[0], keep)]
        a.seeds, a.epochs = 1, 2
        X, rng_map, stats = build("", ids=ids[keep])
    else:
        X, rng_map, stats = build("")
    print(f"선박-일 {len(X):,} × 24시간 × {X.shape[-1]}채널  device={dev}  ({(time.time() - t0) / 60:.1f}분)", flush=True)

    oof = np.zeros((len(S), NC))
    models, log = [], []
    for f, va in enumerate(folds):
        tr = np.setdiff1d(keep if a.smoke else np.arange(len(S)), va)
        P = 0
        for s in range(a.seeds):
            m, ep, es_acc = train_one(X, rng_map, ids, y, tr, dev, seed=s, epochs=a.epochs)
            P = P + predict(m, X, rng_map, ids[va], dev) / a.seeds
            models.append(m)
            log.append(dict(fold=f, seed=s, epochs=ep, es_acc=round(es_acc, 4)))
            print(f"  fold {f} seed {s}: {ep} epoch, 조기종료 정확도 {es_acc:.4f}  ({(time.time() - t0) / 60:.1f}분)", flush=True)
        oof[va] = P
    va_all = np.concatenate(folds)
    res = dict(micro_f1=accuracy_score(y[va_all], oof[va_all].argmax(1)),
               macro_f1=f1_score(y[va_all], oof[va_all].argmax(1), average="macro"), log=log)
    print(f"OOF micro-F1 {res['micro_f1']:.4f} / macro-F1 {res['macro_f1']:.4f}", flush=True)
    if a.smoke:
        print("✅ smoke OK")
        return
    for pool in "AB":
        np.save(C.RESULT_DIR / f"oof_{pool}_cnnmil.npy", oof)
    (C.RESULT_DIR / "cnnmil_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    del X
    Xt, map_t, _ = build("_test", stats=stats)
    tid = np.array(sorted(map_t))
    Pt = np.mean([predict(m, Xt, map_t, tid, dev) for m in models], axis=0)
    out = pd.DataFrame(Pt, columns=[f"p_{c}" for c in C.CLASSES])
    out.insert(0, "rfid_id", tid)
    out.to_csv(C.RESULT_DIR / "cnnmil_test.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분 → oof_*_cnnmil.npy, cnnmil_test.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args())
