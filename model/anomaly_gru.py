"""
[ANOM-4] 예측오차 이상 점수 ─ 직전 12개 시간 기록으로 이번 시간 피처를 예측하는 GRU (자기지도)
  정상 항적으로 "다음 시간이 어떨지"를 배우고, 예측이 크게 빗나간 시간 = 이상
  입력/목표 : anomaly_model.prep_X 피처 (학습 분포로 표준화, ±5 클리핑)
  학습 : 학습 선박 시간대에서 40만 개 표본, 3 epoch (CPU 로 충분)
  출력 : features/anom_gru.parquet, anom_gru_test.parquet  (anom_hourly* 와 같은 행 순서, gru_err 1열)
실행: python anomaly_gru.py [--device cuda]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import fe_config as C
from anomaly_model import prep_X

SEQ = 12


class GRUForecast(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.gru = nn.GRU(d, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.head(self.gru(x)[0][:, -1])


def load(tag, mu=None, sd=None):
    H = pd.read_parquet(C.FEAT_DIR / f"anom_hourly{tag}.parquet")
    X = prep_X(H).to_numpy(np.float32)
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6
    X = np.clip((X - mu) / sd, -5, 5).astype(np.float32)
    rid = H["rfid_id"].to_numpy()
    pos = H.groupby("rfid_id").cumcount().to_numpy()  # 선박 안에서 몇 번째 기록인지
    return X, rid, pos, mu, sd


def windows(X, idx):
    """idx 행을 목표로, 바로 앞 SEQ 개 기록을 입력으로 (선박 시작 부분은 앞쪽을 0 으로 채움)"""
    off = np.arange(-SEQ, 0)
    return X[np.clip(idx[:, None] + off, 0, None)]


def masked(X, pos, idx):
    W = windows(X, idx)
    W[(pos[idx][:, None] + np.arange(-SEQ, 0)) < 0] = 0  # 다른 선박 기록이 섞이지 않게
    return W


def main(a):
    torch.manual_seed(0)
    t0 = time.time()
    X, rid, pos, mu, sd = load("")
    rng = np.random.default_rng(C.RANDOM_STATE)
    cand = np.flatnonzero(pos >= 3)
    tr = rng.choice(cand, size=min(400_000, len(cand)), replace=False)
    dev = torch.device(a.device)
    m = GRUForecast(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(3):
        rng.shuffle(tr)
        tot = 0
        m.train()
        for b in range(0, len(tr), 1024):
            i = tr[b:b + 1024]
            x = torch.from_numpy(masked(X, pos, i)).to(dev)
            y = torch.from_numpy(X[i]).to(dev)
            loss = nn.functional.smooth_l1_loss(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
        print(f"epoch {ep}: loss {tot / len(tr):.4f}  ({(time.time() - t0) / 60:.1f}분)", flush=True)

    m.eval()
    for tag in ["", "_test"]:
        if tag:
            X, rid, pos, _, _ = load(tag, mu, sd)
        err = np.empty(len(X), np.float32)
        with torch.no_grad():
            for b in range(0, len(X), 16384):
                i = np.arange(b, min(b + 16384, len(X)))
                p = m(torch.from_numpy(masked(X, pos, i)).to(dev)).cpu().numpy()
                err[i] = np.abs(p - X[i]).mean(1)
        err[pos < 3] = np.nan  # 앞 기록이 거의 없으면 예측 불가
        pd.DataFrame({"gru_err": err}).to_parquet(C.FEAT_DIR / f"anom_gru{tag}.parquet", index=False)
        print(f"✅ anom_gru{tag}.parquet  ({(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    main(ap.parse_args())
