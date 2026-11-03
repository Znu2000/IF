"""
[STEP6] 아직 안 해 본 두 입력: (1) 시간별 좌표 시퀀스 통째로 → 1D CNN, (2) 항적을 지도 이미지로 → 2D CNN
  seq : 바다에 있던 시간을 시간순으로 최대 512개 (넘으면 학습 때 무작위 구간, 예측 때 여러 구간 평균)
        채널 13 = 위경도(전체 표준화, 절대 위치), 주 조업지 기준 상대 위치(km), 속력·최대 속력, 상태(정지·표류·조업·이동),
                  시각 sin/cos, 직전 바다 시간과의 간격(log), 항구 거리(log)
  img : 10분 점(b_lon0~5)을 64×64 로 그림. 채널 7 = 한반도 전체 지도(절대 위치) 1 + 주 조업지 중심 80km·20km·5km × (조업 상태 점, 이동 점)
  학습 : 1단계와 같은 5 fold. fold 안에서 10% 를 떼어 조기 종료 → OOF / 테스트(5 fold 평균) 확률
  비교 : 단독 성능 + 현재 제출 r4_rp 와 로지스틱 스태킹 (step5 와 같은 방식)
  출력 : features/deep_{seq,img}{,_test}.npy, results/step6_deep_eval.csv, results/step1_{oof,test}_r6_*.npy
실행: python step6_deep.py --build
      python step6_deep.py --kind seq img [--epochs 30] [--device cpu]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import cv_common as V
import fe_config as C
from step3_region import save, score

L_SEQ = 512
IMG = 64
BOX = (123.0, 131.6, 32.0, 39.9)        # 한반도 주변 (경도 min, max, 위도 min, max)
SCALES_KM = (80, 20, 5)
FISH = (1, 2, 3)
torch.set_num_threads(8)


# ───────────────────────── 입력 만들기 ─────────────────────────
def _hourly(tag):
    H = pd.read_parquet(C.FEAT_DIR / f"anom_hourly{tag}.parquet",
                        columns=["rfid_id", "ts", "at_sea", "state", "lon_c", "lat_c", "sp_mean", "sp_max", "port_km"])
    H = H[(H.at_sea == 1)].dropna(subset=["lon_c", "lat_c"]).sort_values(["rfid_id", "ts"])
    return H


def _centers(H, ids):
    F = H[H.state.isin(FISH)]
    c = F.groupby("rfid_id")[["lon_c", "lat_c"]].median()
    c2 = H.groupby("rfid_id")[["lon_c", "lat_c"]].median()
    c = c.reindex(c2.index).fillna(c2)
    return c.reindex(pd.Index(ids, name="rfid_id"))


def build_seq(tag, ids, cen, stats):
    H = _hourly(tag)
    n = len(ids)
    X = np.zeros((n, 13, L_SEQ), np.float16)
    L = np.zeros(n, np.int32)
    pos = {r: i for i, r in enumerate(ids)}
    full = {}
    for v, h in H.groupby("rfid_id"):
        i = pos.get(v)
        if i is None:
            continue
        lon, lat = h.lon_c.to_numpy(), h.lat_c.to_numpy()
        c = cen.loc[v]
        t = (h.ts.astype("int64") // 3_600_000_000_000).to_numpy()
        hr = h.ts.dt.hour.to_numpy()
        gap = np.r_[0, np.diff(t)]
        st = h.state.to_numpy()
        f = np.stack([(lon - stats[0]) / stats[1], (lat - stats[2]) / stats[3],
                      (lon - c.lon_c) * 111.32 * np.cos(np.radians(c.lat_c)) / 20, (lat - c.lat_c) * 110.57 / 20,
                      h.sp_mean.fillna(0).to_numpy() / 5, h.sp_max.fillna(0).to_numpy() / 10,
                      st == 1, st == 2, st == 3, st == 4,
                      np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), np.log1p(gap) / 3], 0).astype(np.float32)
        f[2:4] = np.clip(f[2:4], -10, 10)
        full[i] = f
        L[i] = f.shape[1]
        X[i, :, :min(L_SEQ, L[i])] = f[:, :L_SEQ]
    return X, L, full


def build_img(tag, ids, cen):
    n = len(ids)
    pos = pd.Series(np.arange(n), index=pd.Index(ids))
    S = pd.read_parquet(C.FEAT_DIR / f"anom_hourly{tag}.parquet", columns=["rfid_id", "ts", "at_sea", "state"])
    S = S[S.at_sea == 1]
    S["part_dt"] = S.ts.dt.strftime("%Y%m%d").astype(np.int32)
    S["hour"] = S.ts.dt.hour.astype(np.int8)
    S = S[["rfid_id", "part_dt", "hour", "state"]]
    cols = ["part_dt", "rfid_id", "hour"] + [f"b_lon{j}" for j in range(6)] + [f"b_lat{j}" for j in range(6)]
    pre = "test_hourly" if tag else "hourly"
    acc = np.zeros((n, 7, IMG, IMG), np.float32)
    clon = cen.lon_c.to_numpy()
    clat = cen.lat_c.to_numpy()
    for m in range(1, 7):
        P = pd.read_parquet(C.FEAT_DIR / f"{pre}_{m:02d}.parquet", columns=cols).merge(S, on=["rfid_id", "part_dt", "hour"])
        vi = P.rfid_id.map(pos).to_numpy()
        ok = ~np.isnan(vi)
        P, vi = P[ok], vi[ok].astype(np.int64)
        fish = P.state.isin(FISH).to_numpy()
        for j in range(6):
            lon, lat = P[f"b_lon{j}"].to_numpy(), P[f"b_lat{j}"].to_numpy()
            g = ~np.isnan(lon)
            v, lo, la, fi = vi[g], lon[g], lat[g], fish[g]
            # 절대 지도
            x = ((lo - BOX[0]) / (BOX[1] - BOX[0]) * IMG).astype(int)
            y = ((BOX[3] - la) / (BOX[3] - BOX[2]) * IMG).astype(int)
            k = (x >= 0) & (x < IMG) & (y >= 0) & (y < IMG)
            np.add.at(acc, (v[k], 0, y[k], x[k]), 1)
            # 주 조업지 중심 상대 지도 (3 크기 × 조업/이동)
            dx = (lo - clon[v]) * 111.32 * np.cos(np.radians(clat[v]))
            dy = (clat[v] - la) * 110.57
            for s, km in enumerate(SCALES_KM):
                x = ((dx / km + 0.5) * IMG).astype(int)
                y = ((dy / km + 0.5) * IMG).astype(int)
                k = (x >= 0) & (x < IMG) & (y >= 0) & (y < IMG)
                for ch, sel in ((1 + 2 * s, fi), (2 + 2 * s, ~fi)):
                    kk = k & sel
                    np.add.at(acc, (v[kk], ch, y[kk], x[kk]), 1)
        print(f"    {pre}_{m:02d} 점 {len(P) * 6:,}", flush=True)
    acc = np.log1p(acc)
    mx = acc.reshape(n, 7, -1).max(2)[:, :, None, None]
    return (acc / np.maximum(mx, 1e-6) * 255).astype(np.uint8)    # 배·채널마다 최대 = 255


def build():
    S, _ = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    Htr = _hourly("")
    stats = (Htr.lon_c.mean(), Htr.lon_c.std(), Htr.lat_c.mean(), Htr.lat_c.std())
    for tag, ids in [("", S.rfid_id.to_numpy()), ("_test", T.rfid_id.to_numpy())]:
        t0 = time.time()
        cen = _centers(_hourly(tag), ids)
        X, L, full = build_seq(tag, ids, cen, stats)
        np.save(C.FEAT_DIR / f"deep_seq{tag}.npy", X)
        np.save(C.FEAT_DIR / f"deep_seqlen{tag}.npy", L)
        obj = np.empty(len(ids), dtype=object)
        for i, f in full.items():
            obj[i] = f
        np.save(C.FEAT_DIR / f"deep_seqfull{tag}.npy", obj, allow_pickle=True)
        print(f"✅ seq{tag} {X.shape}, 바다 기록 없는 배 {(L == 0).sum()} ({(time.time() - t0) / 60:.1f}분)", flush=True)
        I = build_img(tag, ids, cen.fillna(cen.median()))
        np.save(C.FEAT_DIR / f"deep_img{tag}.npy", I)
        print(f"✅ img{tag} {I.shape} ({(time.time() - t0) / 60:.1f}분)", flush=True)


# ───────────────────────── 모델 ─────────────────────────
class SeqNet(nn.Module):
    def __init__(self, cin=13, h=64, nc=4):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.GELU(),
                               nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.GELU(),
                               nn.Conv1d(h, h, 5, padding=8, dilation=4), nn.BatchNorm1d(h), nn.GELU(),
                               nn.Conv1d(h, h, 5, padding=16, dilation=8), nn.BatchNorm1d(h), nn.GELU())
        self.o = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * h, nc))

    def forward(self, x, m):
        z = self.c(x)
        mf = m[:, None, :]
        mean = (z * mf).sum(2) / mf.sum(2).clamp(min=1)
        mx = z.masked_fill(mf == 0, -1e4).amax(2)
        logn = torch.log1p(mf.sum(2)).expand(-1, z.shape[1])
        return self.o(torch.cat([mean, mx, logn], 1))


class ImgNet(nn.Module):
    def __init__(self, cin=7, nc=4):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(2))
        self.c = nn.Sequential(blk(cin, 32), blk(32, 64), blk(64, 96), blk(96, 128))
        self.o = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 2, nc))

    def forward(self, x, m=None):
        z = self.c(x)
        return self.o(torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1))


def _seq_batch(full, idx, rng, train):
    X = np.zeros((len(idx), 13, L_SEQ), np.float32)
    M = np.zeros((len(idx), L_SEQ), np.float32)
    for b, i in enumerate(idx):
        f = full[i]
        if f is None:
            continue
        n = f.shape[1]
        s = rng.integers(0, n - L_SEQ + 1) if (train and n > L_SEQ) else 0
        k = min(n, L_SEQ)
        X[b, :, :k] = f[:, s:s + k]
        M[b, :k] = 1
    return torch.from_numpy(X), torch.from_numpy(M)


def _predict_seq(model, full, idx, bs=256):
    """512 넘는 배는 겹치지 않는 구간들 평균"""
    model.eval()
    out = np.zeros((len(idx), 4), np.float32)
    with torch.no_grad():
        jobs = []
        for b, i in enumerate(idx):
            f = full[i]
            n = 0 if f is None else f.shape[1]
            starts = [0] if n <= L_SEQ else list(range(0, n - L_SEQ + 1, L_SEQ)) + [n - L_SEQ]
            jobs += [(b, i, s) for s in starts]
        cnt = np.zeros(len(idx))
        for j in range(0, len(jobs), bs):
            ch = jobs[j:j + bs]
            X = np.zeros((len(ch), 13, L_SEQ), np.float32)
            M = np.zeros((len(ch), L_SEQ), np.float32)
            for q, (b, i, s) in enumerate(ch):
                f = full[i]
                if f is not None:
                    k = min(f.shape[1], L_SEQ)
                    X[q, :, :k] = f[:, s:s + k]
                    M[q, :k] = 1
            p = torch.softmax(model(torch.from_numpy(X), torch.from_numpy(M)), 1).numpy()
            for q, (b, _, _) in enumerate(ch):
                out[b] += p[q]
                cnt[b] += 1
    return out / cnt[:, None]


def fit_predict(kind, data, y, tr, queries, a, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr = rng.permutation(tr)
    nv = max(1, len(tr) // 10)
    es, fit_i = tr[:nv], tr[nv:]
    model = SeqNet() if kind == "seq" else ImgNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * (len(fit_i) // a.bs + 1))
    lossf = nn.CrossEntropyLoss()
    yt = torch.from_numpy(y)

    def pred(src, idx):
        if kind == "seq":
            return _predict_seq(model, src, idx)
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.softmax(model(torch.from_numpy(src[idx[j:j + 512]].astype(np.float32) / 255)), 1).numpy()
                                   for j in range(0, len(idx), 512)])

    best, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(fit_i)
        for j in range(0, len(perm), a.bs):
            b = perm[j:j + a.bs]
            if kind == "seq":
                X, M = _seq_batch(data["tr"], b, rng, True)
                logit = model(X, M)
            else:
                X = torch.from_numpy(data["tr"][b].astype(np.float32) / 255)
                logit = model(X)
            loss = lossf(logit, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = accuracy_score(y[es], pred(data["tr"], es).argmax(1))
        if acc > best:
            best, bad, best_state = acc, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return [pred(data[src], idx) for src, idx in queries], best, ep + 1


# ───────────────────────── 평가 ─────────────────────────
def main(a):
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    lab = dict(zip(S.rfid_id.astype(str), y))
    ov_mask = T.rfid_id.astype(str).isin(lab).to_numpy()
    y_ov = T.rfid_id.astype(str)[ov_mask].map(lab).to_numpy()
    Bo = np.load(C.RESULT_DIR / f"step1_oof_{a.base}.npy")
    Bt = np.load(C.RESULT_DIR / f"step1_test_{a.base}.npy")
    lg = lambda P: np.log(np.clip(P, 1e-6, 1))
    allte = np.arange(len(T))
    rows, preds = [], {}
    for kind in a.kind:
        if kind == "seq":
            data = {"tr": np.load(C.FEAT_DIR / "deep_seqfull.npy", allow_pickle=True),
                    "te": np.load(C.FEAT_DIR / "deep_seqfull_test.npy", allow_pickle=True)}
            has_tr = np.array([f is not None for f in data["tr"]])
            has_te = np.array([f is not None for f in data["te"]])
        else:
            data = {"tr": np.load(C.FEAT_DIR / "deep_img.npy"), "te": np.load(C.FEAT_DIR / "deep_img_test.npy")}
            has_tr = data["tr"].reshape(len(y), -1).max(1) > 0
            has_te = data["te"].reshape(len(T), -1).max(1) > 0
        oof = np.zeros((len(y), 4))
        te = np.zeros((len(T), 4))
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), va)
            prior = np.bincount(y[tr], minlength=4) / len(tr)
            (pv, pt), es_acc, ep = fit_predict(kind, data, y, tr[has_tr[tr]], [("tr", va), ("te", allte)], a, seed=f)
            pv[~has_tr[va]], pt[~has_te] = prior, prior     # 바다 기록 없는 배 → 학습 비율
            oof[va] = pv
            te += pt / len(folds)
            print(f"  [{kind}] fold {f}: acc {accuracy_score(y[va], pv.argmax(1)):.4f} (조기종료 검증 {es_acc:.4f}, "
                  f"{ep} epoch, {(time.time() - t0) / 60:.1f}분)", flush=True)
        preds[kind] = (oof, te)
        np.save(C.RESULT_DIR / f"step6_oof_{kind}.npy", oof)
        np.save(C.RESULT_DIR / f"step6_test_{kind}.npy", te)
        rows.append(dict(model=f"{kind} 단독", cv=accuracy_score(y, oof.argmax(1)), cv_macro=V.macro_f1(y, oof.argmax(1)),
                         cross_seen=accuracy_score(y_ov, te[ov_mask].argmax(1))))
        print(rows[-1], flush=True)

    cv5 = StratifiedKFold(5, shuffle=True, random_state=1)
    combos = {"base": []} | {f"+{k}": [k] for k in preds}
    if len(preds) > 1:
        combos["+both"] = list(preds)
    res = []
    for name, parts in combos.items():
        Xo = np.hstack([lg(Bo)] + [lg(preds[p][0]) for p in parts])
        Xt = np.hstack([lg(Bt)] + [lg(preds[p][1]) for p in parts])
        st = LogisticRegression(max_iter=3000, C=0.3)
        O = cross_val_predict(st, Xo, y, cv=cv5, method="predict_proba")
        P = st.fit(Xo, y).predict_proba(Xt)
        cfg = f"r6_{name.strip('+')}"
        r = score(cfg, O, P[ov_mask], None, y, y_ov, fold_id)
        res.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "weights"}, flush=True)
        if name != "base":
            np.save(C.RESULT_DIR / f"step1_oof_{cfg}.npy", O)
            np.save(C.RESULT_DIR / f"step1_test_{cfg}.npy", P)
    E = pd.DataFrame(res)
    E["d_cv"], E["d_cross_seen_adj"] = E.cv - E.cv.iloc[0], E.cross_seen_adj - E.cross_seen_adj.iloc[0]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(E.drop(columns="weights").round(4).to_string(index=False))
    pd.concat([pd.DataFrame(rows), E]).round(4).to_csv(C.RESULT_DIR / "step6_deep_eval.csv", index=False,
                                                       encoding="utf-8-sig")
    save(res[1:])
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--kind", nargs="*", default=["seq", "img"])
    ap.add_argument("--base", default="r4_rp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    build() if a.build else main(a)
