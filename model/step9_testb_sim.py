"""
[STEP9-2] Test B 모의 채점기 + 사고 유형 흔적 탐지기
  공지된 규칙 : 예측 시각이 실제 사고 시각에 가까울수록 높은 점수, 6시간 초과 = 0점, 사고 없다고 보면 빈칸
  추정한 규칙 : 사고 없는 배를 빈칸 = 1점 / 채우면 0점. 사고 배를 빈칸 = 0점.
                시각 점수 = max(0, 1 − |오차|/6h) (선형, 가정) ─ 계단형(±6h 안이면 1)도 같이 봄
                사고 선박 비율 20% (모두 빈칸 제출 시 B ≈ 0.80 이 나온 총점에서 역산)
  모의 세트   : 학습(2026) 선박에서 사고 선박(라벨 v2 확실 등급) 20척 + 사고 없는 선박 80척을 2,000번 뽑아 채점
  탐지기      : 시간대마다 "바다에서 이동·조업하다 갑자기 정지·표류"(기동 불능 흔적)를 찾아 점수화 (사후 판정이라 앞뒤 기록 사용)
                onset  = 바다 정지·표류 시작 + 직전 4시간 안에 바다 이동·조업
                drift  = 이후 12시간 안 바다 정지·표류 시간 수
                tow    = 이후 12시간 안 저속(2~6노트) 직진(방향 일관성 ≥ 0.8) = 예인·저속 귀항
                ret    = 이후 12시간 안 항구 도착,  rare = 그 배가 평소 안 가는 칸 (cell_rarity)
  전략        : 배 최고 점수 상위 k 척만 시각(최고 점수 시각)을 채우고 나머지 빈칸, k = 0 ~ 30
  출력 : results/step9_testb_detect.csv, step9_testb_sim.csv
실행: python step9_testb_sim.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fe_config as C

ACC_FRAC, N_SET, N_DRAW, SEED = 0.20, 100, 2000, 0
SURE = ["AS", "A", "S"]
COLS = ["rfid_id", "ts", "at_sea", "state", "ks_speed_med", "ks_heading_R", "cell_rarity"]


def fwd_any(v, cond, ts, hours, steps=12):
    """같은 배의 다음 기록들 중 hours 시간 안에 cond 가 있는지 / 개수"""
    n = len(cond)
    cnt = np.zeros(n, np.int16)
    for k in range(1, steps + 1):
        same = np.r_[v[k:] == v[:-k], np.zeros(k, bool)]
        dt = np.r_[(ts[k:] - ts[:-k]) / np.timedelta64(1, "h"), np.full(k, np.inf)]
        c = np.r_[cond[k:], np.zeros(k, bool)]
        cnt += (same & (dt <= hours) & c).astype(np.int16)
    return cnt


def bwd_any(v, cond, ts, hours, steps=4):
    n = len(cond)
    out = np.zeros(n, bool)
    for k in range(1, steps + 1):
        same = np.r_[np.zeros(k, bool), v[k:] == v[:-k]]
        dt = np.r_[np.full(k, np.inf), (ts[k:] - ts[:-k]) / np.timedelta64(1, "h")]
        c = np.r_[np.zeros(k, bool), cond[:-k]]
        out |= same & (dt <= hours) & c
    return out


def events(H):
    v = H["rid"].to_numpy()
    ts = H["ts"].to_numpy()
    sea = H["at_sea"].fillna(0).to_numpy() == 1
    st = H["state"].to_numpy()
    sp = H["ks_speed_med"].fillna(0).to_numpy()
    hr = H["ks_heading_R"].fillna(0).to_numpy()
    stop = sea & np.isin(st, (1, 2))
    move = sea & np.isin(st, (3, 4))
    prev_stop = np.r_[False, (v[1:] == v[:-1]) & stop[:-1]]
    onset = stop & ~prev_stop & bwd_any(v, move, ts, 4)
    drift = fwd_any(v, stop, ts, 12)
    tow = fwd_any(v, sea & (sp >= 2) & (sp < 6) & (hr >= 0.8), ts, 12) > 0
    ret = fwd_any(v, st == 0, ts, 12) > 0
    rare = H["cell_rarity"].fillna(0).to_numpy()
    d = 1 + np.minimum(drift, 12) / 6
    return {
        "onset": onset * d,
        "onset_tow": onset * d * (1 + tow + ret),
        "onset_tow_rare": onset * d * (1 + tow + ret) * (0.5 + rare),
        "stop_rare": stop * (0.5 + rare),
    }


def time_score(err_h, kind):
    e = np.abs(err_h)
    return np.where(e > 6, 0.0, 1 - e / 6) if kind == "linear" else (e <= 6).astype(float)


def main():
    t0 = time.time()
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=COLS)
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.sort_values(["rid", "ts"], kind="stable").reset_index(drop=True)
    E = events(H)
    print(f"시간대 {len(H):,}행, 탐지 점수 계산 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["acc_time", "label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    pos = L[L["tier"].isin(SURE)].drop_duplicates("rid")
    neg_ids = np.array(sorted(set(H["rid"]) - touched))
    print(f"사고 선박(확실) {len(pos)}척 (바다 {int((pos['group'] != 'port').sum())}, 항구 {int((pos['group'] == 'port').sum())}) / "
          f"사고 없는 선박 {len(neg_ids):,}척", flush=True)

    rows, vs = [], {}
    for name, s in E.items():
        Hs = H[["rid", "ts"]].assign(s=s)
        top = Hs.loc[Hs.groupby("rid")["s"].idxmax()].set_index("rid")
        vs[name] = top
        p = top.reindex(pos["rid"])
        err = (p["ts"].to_numpy() - pos["label_ts"].to_numpy()) / np.timedelta64(1, "h")
        has = p["s"].fillna(0).to_numpy() > 0
        sea = (pos["group"] != "port").to_numpy()
        ys = np.r_[np.ones(len(pos)), np.zeros(len(neg_ids))]
        sc = np.r_[p["s"].fillna(0).to_numpy(), top["s"].reindex(neg_ids).fillna(0).to_numpy()]
        rows.append(dict(detector=name,
                         hit6_all=float(np.mean(has & (np.abs(err) <= 6))),
                         hit6_sea=float(np.mean((has & (np.abs(err) <= 6))[sea])),
                         hit6_port=float(np.mean((has & (np.abs(err) <= 6))[~sea])),
                         time_score_mean=float(np.nanmean(np.where(has, time_score(err, "linear"), 0))),
                         vessel_auc=roc_auc_score(ys, sc)))
    D = pd.DataFrame(rows)
    D.round(4).to_csv(C.RESULT_DIR / "step9_testb_detect.csv", index=False, encoding="utf-8-sig")
    print(D.round(3).to_string(index=False), flush=True)

    # 모의 채점: 사고 20척 + 정상 80척, 배 최고 점수 상위 k 척만 채움
    rng = np.random.default_rng(SEED)
    n_pos = int(round(N_SET * ACC_FRAC))
    sims = []
    for name, top in vs.items():
        ps = top["s"].reindex(pos["rid"]).fillna(0).to_numpy()
        perr = (top["ts"].reindex(pos["rid"]).to_numpy() - pos["label_ts"].to_numpy()) / np.timedelta64(1, "h")
        ns = top["s"].reindex(neg_ids).fillna(0).to_numpy()
        for kind in ("linear", "step"):
            tsc = np.nan_to_num(time_score(perr, kind))
            res = {k: [] for k in range(0, 31)}
            for _ in range(N_DRAW):
                ip = rng.choice(len(ps), n_pos, replace=len(ps) < n_pos)
                ineg = rng.choice(len(ns), N_SET - n_pos, replace=False)
                s = np.r_[ps[ip] + rng.random(n_pos) * 1e-9, ns[ineg] + rng.random(N_SET - n_pos) * 1e-9]
                is_pos = np.r_[np.ones(n_pos, bool), np.zeros(N_SET - n_pos, bool)]
                gain = np.r_[tsc[ip], np.zeros(N_SET - n_pos)]
                order = np.argsort(-s)
                for k in res:
                    fill = np.zeros(N_SET, bool)
                    fill[order[:k]] = True
                    score = np.where(fill, gain, (~is_pos).astype(float)).mean()
                    res[k].append(score)
            for k, v in res.items():
                v = np.array(v)
                sims.append(dict(detector=name, time_rule=kind, k=k, B_mean=v.mean(), B_p05=np.quantile(v, 0.05),
                                 B_p95=np.quantile(v, 0.95)))
    Sm = pd.DataFrame(sims)
    Sm.round(4).to_csv(C.RESULT_DIR / "step9_testb_sim.csv", index=False, encoding="utf-8-sig")
    best = Sm.loc[Sm.groupby(["detector", "time_rule"])["B_mean"].idxmax()]
    print(Sm[Sm["k"].isin([0, 1, 3, 5, 10, 20])].pivot_table(index=["detector", "time_rule"], columns="k",
                                                                values="B_mean").round(4).to_string())
    print("최적 k:\n" + best[["detector", "time_rule", "k", "B_mean", "B_p05", "B_p95"]].round(4).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
