"""
[STEP14-3] 미탐 중심 재평가 ─ 경보 예산(정상 배 바다 시간의 q)을 늘릴 때 사고를 얼마나 잡고, 오탐은 얼마나 느는지
  점수 (step13 의 시각 t 까지 점수 rt, 사후 포함 post)
    none / soft : 공통 기준, 1단계 확률 혼합 (합산)
    OR          : 항목별 백분위의 최댓값. 한 항목만 튀어도 경보 (합산에 묻히는 미탐을 줄이려는 시도)
    업종별      : 예측 업종 안에서 백분위로 바꿔 업종마다 같은 예산 (경보가 적은 업종의 미탐을 줄이려는 시도)
  임계값은 지역 안 사고 무관 선박 바다 시간으로만 정함. 라벨로 고르지 않음
  평가
    A·AS : 사고 ±6h 안 경보. 위약은 같은 배의 ±7·14·21·28일 중 그 시각 바다에 있던 때
    묶음 : 사고 3km 안 같은 업종 배 중 ±6h 경보 비율, 시각을 옮겨 다시 고른 묶음과 비교. 초과척 = 차이 * 묶음 크기
    오탐 : 정상 배 1척당 월 경보 수 (6시간 안 연속 경보는 1건)
  비용: 미탐 1건 = 오탐 C 건일 때 C * (놓친 사고) + 오탐 건수 최소인 예산. 재현율은 A·AS 기준
  출력: results/step14_recall_<시각>.csv, results/step14_cost_<시각>.csv (rfid_id 없음)
실행: python step14_recall.py
"""
import time

import numpy as np
import pandas as pd

import fe_config as C
import step13_stage2 as M
import step9_detect as D
import step9_labels as L9

STAMP = time.strftime("%m%d_%H%M")
BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
SHIFTS = [-28, -21, -14, -7, 7, 14, 21, 28]
COSTS = [10, 30, 50, 100]
N_ACC = [10, 20, 40]
FOCUS = ["none rt", "soft rt", "OR rt", "soft rt 업종별", "OR rt 업종별", "soft post", "OR post"]


def pct(s, ref):
    r = np.sort(s[ref])
    return (np.searchsorted(r, s, side="left") / max(len(r), 1)).astype(np.float32)


def pct_by(s, ref, grp):
    out = np.zeros(len(s), np.float32)
    for k in np.unique(grp):
        m = grp == k
        r = np.sort(s[m & ref])
        out[m] = np.searchsorted(r, s[m], side="left") / max(len(r), 1)
    return out


def window_any(al, ts, sea, t, h, a, e):
    rel = np.abs((ts[a:e] - t) / np.timedelta64(1, "h"))
    return bool((al[a:e] & (rel <= h)).any()), bool((sea[a:e] & (rel <= 1)).any())


def alarms(al, v, ts, mask):
    """정상 배 경보 건수: 같은 배 6시간 안 연속 경보는 1건"""
    i = np.flatnonzero(al & mask)
    if len(i) == 0:
        return 0
    new = np.r_[True, (v[i][1:] != v[i][:-1]) | ((ts[i][1:] - ts[i][:-1]) / np.timedelta64(1, "h") > 6)]
    return int(new.sum())


def main():
    t0 = time.time()
    H, S, y, fold, P = M.load()
    X = M.context(H)
    row_of = pd.Series(np.arange(len(S)), index=S["rid"]).reindex(X["v"]).to_numpy()
    g, fold_row, P_row = y[row_of], fold[row_of], P[row_of]
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    ref_ok = ~np.isin(X["v"], list(touched))
    sl = M.slices(X["v"])
    pos = L[(L["group"] != "port") & L["tier"].isin(["A", "AS"]) & L["rid"].isin(sl)].drop_duplicates("rid")
    acc = pd.read_excel(L9.ACC)
    H_ = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "home_lat", "home_lon"])
    box = (H_["home_lat"].between(acc["위도(변환)"].min() - 0.2, acc["위도(변환)"].max() + 0.2)
           & H_["home_lon"].between(acc["경도(변환)"].min() - 0.2, acc["경도(변환)"].max() + 0.2))
    neg_r = np.array(sorted((set(sl) - touched) & set(H_.loc[box, "rfid_id"].astype(str))))
    ref = np.isin(X["v"], neg_r) & X["sea"]
    print(f"A·AS {len(pos)}척 / 정상 비교군 {len(neg_r):,}척 / 바다 {ref.sum():,}시간 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    b = M.bits_all(X, g, P_row, fold_row, ref_ok)
    hard = (P_row * M.W_HARD).argmax(1)
    F = D.build(H)
    S_ = {"급정지": np.where(F["onset"], 1 + F["f_drift"] / 6, 0.0)}
    for m in ("none", "soft"):
        rt = sum(b[m][k] for k in M.PRE)
        S_[f"{m} rt"] = rt
        S_[f"{m} post"] = rt + sum(b[m][k] for k in M.POST)
    pc = {k: pct(b["soft"][k], ref) for k in M.PRE + M.POST}
    S_["OR rt"] = np.max([pc[k] for k in M.PRE], 0)
    S_["OR post"] = np.max([pc[k] for k in M.PRE + M.POST], 0)
    S_["soft rt 업종별"] = pct_by(S_["soft rt"], ref, hard)
    S_["OR rt 업종별"] = np.max([pct_by(b["soft"][k], ref, hard) for k in M.PRE], 0)
    del b, pc
    print(f"점수 계산 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    Bg = M.bag_candidates()
    Bg = Bg[(Bg["group"] != "port") & Bg["rid"].isin(sl)]
    sea, ts, v = X["sea"], X["ts"], X["v"]
    rng = np.random.default_rng(0)
    rows = []
    n_ves = len(neg_r)
    for name, s in S_.items():
        s = s * sea
        for q in BUDGETS:
            thr = np.quantile(s[ref], 1 - q)
            al = (s > thr) & sea
            rec = dict(점수=name, 예산=q, 실제경보율=al[ref].mean(), 오탐건=alarms(al, v, ts, ref))
            rec["척당월경보"] = rec["오탐건"] / n_ves / 6
            hit, plc = [], []
            for p in pos.itertuples():
                a, e = sl[p.rid]
                t = np.datetime64(p.label_ts)
                hit.append(window_any(al, ts, sea, t, 6, a, e)[0])
                ph = []
                for d in SHIFTS:
                    tt = t + np.timedelta64(d, "D")
                    h_, at_sea = window_any(al, ts, sea, tt, 6, a, e)
                    if at_sea:
                        ph.append(h_)
                plc.append(np.mean(ph) if ph else np.nan)
            rec["AAS_적중"] = int(np.sum(hit))
            rec["AAS_재현"] = np.mean(hit)
            rec["AAS_위약"] = np.nanmean(plc)
            vals = np.zeros(len(Bg))
            for j, c in enumerate(Bg.itertuples()):
                a, e = sl[c.rid]
                vals[j] = window_any(al, ts, sea, np.datetime64(c.t), 6, a, e)[0]
            G = Bg.assign(hit=vals).groupby(["acc_idx", "shift"])["hit"].agg(["mean", "size"])
            real = G.xs(0, level="shift")
            pl = G[G.index.get_level_values("shift") != 0].groupby("acc_idx")["mean"].mean()
            idx = real.index.intersection(pl.index)
            d = (real.loc[idx, "mean"] - pl.loc[idx]).to_numpy()
            rec |= dict(묶음_사고=len(idx), 묶음_실제=real.loc[idx, "mean"].mean(), 묶음_위약=pl.loc[idx].mean(), 묶음_차이=d.mean(), 묶음_p=M.sign_flip(d, rng),
                        묶음_초과척=(d * real.loc[idx, "size"].to_numpy()).mean())
            rows.append(rec)
        print(f"{name} 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    R = pd.DataFrame(rows)
    R.round(4).to_csv(C.RESULT_DIR / f"step14_recall_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(R.round(3).to_string(index=False), flush=True)

    rows = []
    for name in ["급정지"] + FOCUS:
        G = R[R["점수"] == name]
        for n_acc in N_ACC:
            for c in COSTS:
                cost = c * n_acc * (1 - G["AAS_재현"].to_numpy()) + G["오탐건"].to_numpy()
                base = c * n_acc * (1 - G["AAS_위약"].to_numpy()) + G["오탐건"].to_numpy()
                k = cost.argmin()
                rows.append(dict(점수=name, 사고건=n_acc, 미탐비용=c, 최적예산=G["예산"].iloc[k], 재현=G["AAS_재현"].iloc[k], 위약재현=G["AAS_위약"].iloc[k],
                                 오탐건=G["오탐건"].iloc[k], 비용=cost[k], 무경보비용=c * n_acc, 무작위동일예산비용=base[k]))
    Co = pd.DataFrame(rows)
    Co.round(4).to_csv(C.RESULT_DIR / f"step14_cost_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Co[Co["사고건"] == 20].round(3).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
