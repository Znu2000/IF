"""
[ANOM-8] 복원오차 이상 점수 ─ 직전 12개 기록 + 현재 기록(13시간 창)을 압축했다 되살리는 1D-CNN Autoencoder (자기지도)
  GRU(다음 시간 예측)와의 차이: GRU 는 "갑자기 바뀌었나", AE 는 "이 몇 시간의 모양 자체가 평소 항적과 다른가"
  실시간 : 창은 현재 시각까지만 포함 (미래 없음), 입력 피처는 anomaly_gru --rt 와 같음 (gap_next_h 제외)
  점수   : 현재 시각(창의 마지막 칸) 복원오차 + 창 전체 복원오차의 평균
  학습   : 학습 선박 시간대 40만 개 표본, 3 epoch (CPU 로 수 분)
  출력   : features/anom_ae.parquet, anom_ae_test.parquet  (anom_hourly* 와 같은 행 순서, ae_err 1열)
실행: python anomaly_ae.py [--device cuda]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import fe_config as C
from anomaly_gru import load

SEQ = 13  # 직전 12개 + 현재


class ConvAE(nn.Module):
    def __init__(self, d, h=32, z=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(d, h, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.ReLU(),
                                 nn.Conv1d(h, z, 1))  # 시간축은 유지하고 채널을 z 로 압축 (병목)
        self.dec = nn.Sequential(nn.Conv1d(z, h, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(h, h, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(h, d, 1))

    def forward(self, x):  # x: (B, T, d)
        return self.dec(self.enc(x.transpose(1, 2))).transpose(1, 2)


def win(X, pos, idx):
    """idx 행을 마지막 칸으로 하는 SEQ 길이 창 (선박 시작 이전 칸은 0, mask 로 손실에서 제외)"""
    off = np.arange(-SEQ + 1, 1)
    W = X[np.clip(idx[:, None] + off, 0, None)]
    m = (pos[idx][:, None] + off) >= 0
    W[~m] = 0
    return W, m


def main(a):
    torch.manual_seed(0)
    t0 = time.time()
    X, rid, pos, mu, sd = load("", rt=True)
    rng = np.random.default_rng(C.RANDOM_STATE)
    cand = np.flatnonzero(pos >= 3)
    tr = rng.choice(cand, size=min(400_000, len(cand)), replace=False)
    dev = torch.device(a.device)
    m = ConvAE(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(3):
        rng.shuffle(tr)
        tot = 0
        m.train()
        for b in range(0, len(tr), 1024):
            W, mk = win(X, pos, tr[b:b + 1024])
            x, mk = torch.from_numpy(W).to(dev), torch.from_numpy(mk).to(dev)
            e = nn.functional.smooth_l1_loss(m(x), x, reduction="none").mean(2)
            loss = (e * mk).sum() / mk.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(W)
        print(f"epoch {ep}: loss {tot / len(tr):.4f}  ({(time.time() - t0) / 60:.1f}분)", flush=True)

    m.eval()
    for tag in ["", "_test"]:
        if tag:
            X, rid, pos, _, _ = load(tag, mu, sd, rt=True)
        err = np.empty(len(X), np.float32)
        with torch.no_grad():
            for b in range(0, len(X), 16384):
                i = np.arange(b, min(b + 16384, len(X)))
                W, mk = win(X, pos, i)
                e = np.abs(m(torch.from_numpy(W).to(dev)).cpu().numpy() - W).mean(2)
                err[i] = 0.5 * e[:, -1] + 0.5 * (e * mk).sum(1) / mk.sum(1)
        err[pos < 3] = np.nan
        name = f"anom_ae{tag}.parquet"
        pd.DataFrame({"ae_err": err}).to_parquet(C.FEAT_DIR / name, index=False)
        print(f"✅ {name}  ({(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    main(ap.parse_args())
