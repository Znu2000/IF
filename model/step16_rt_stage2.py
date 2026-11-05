"""
[STEP16-4] 실시간 1단계 확률로 2단계 다시 계산 ─ 시각 t 에 실제로 알 수 있는 업종 확률만 쓰면 이상탐지 평가가 어떻게 바뀌나
  기존 step13: 배마다 전체 항적으로 낸 확률 1개를 모든 시간에 사용 (rt 점수에 미래 정보가 섞임)
  여기: 그 시각까지 경과 일수 e 로 쓸 수 있는 가장 긴 잘린 항적 N ≤ e 의 확률. e < 3일은 업종 사전 분포 (전체 라벨 비율)
  비교: 공통 기준 / 전체 항적 3모델(step13) / 전체 항적 LGB / 실시간 matched / 실시간 aug / 실제 업종
  평가: step13 alert_eval(경보 쏠림, 상위 0.1%), time_eval(±6h 적중). 경과 일수 구간별 쏠림도 계산
  한계: 기준표와 상태 판정의 모항은 그대로 전체 기간. 작은 누수가 남을 수 있음
  출력: results/step16_stage2_<시각>.csv (rfid_id 없음)
실행: python step16_rt_stage2.py (step16_rt_class.py 먼저)
"""
import time

import numpy as np
import pandas as pd

import fe_config as C
import step13_stage2 as M
import step16_area as AR
import step16_rt_class as RC

STAMP = time.strftime("%m%d_%H%M")
PERIODS = {"0-14일": (0, 14), "14-90일": (14, 90), "90일 이후": (90, 1e9)}


def rt_probs(mode, elapsed, row_of, y):
    prior = np.bincount(y, minlength=M.NG) / len(y)
    P = np.tile(prior, (len(elapsed), 1)).astype(np.float32)
    for n in RC.CUTS:
        Pn = np.load(C.RESULT_DIR / f"step16_oof_{mode}_{n}.npy")
        m = elapsed >= n
        P[m] = Pn[row_of[m]]
    return P


def skew(s, neg_mask, g, thr, mask=None):
    m = neg_mask if mask is None else neg_mask & mask
    r = np.array([(s[m & (g == k)] > thr).mean() * 1000 for k in range(M.NG)])
    return r, r.max() / max(r.min(), 1e-3)


def main():
    t0 = time.time()
    H, S, y, fold, P3 = M.load()
    X = M.context(H)
    row_of = pd.Series(np.arange(len(S)), index=S["rid"]).reindex(X["v"]).to_numpy()
    g, fold_row = y[row_of], fold[row_of]
    first = H.groupby("rid")["ts"].transform("min").to_numpy()
    el = ((X["ts"] - first) / np.timedelta64(1, "D")).astype(np.float32)
    sea_el = el[X["sea"]]
    print("바다 시간 경과일 분포:", {k: round(float(((sea_el >= a) & (sea_el < b)).mean()), 3) for k, (a, b) in PERIODS.items()}, flush=True)
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    tch = AR.touched()
    ref_ok = ~np.isin(X["v"], list(tch))
    sl = M.slices(X["v"])
    sea_l = L[(L["group"] != "port") & L["rid"].isin(sl)]
    pos = {"A·AS(거리)": sea_l[sea_l["tier"].isin(["A", "AS"])].drop_duplicates("rid"),
           "A·AS·S(순환 포함)": sea_l[sea_l["tier"].isin(["A", "AS", "S"])].drop_duplicates("rid")}
    neg = np.array(sorted(set(sl) - tch))
    neg_mask = np.isin(X["v"], neg) & X["sea"]

    src = {"전체 항적 3모델": P3[row_of],
           "전체 항적 LGB": np.load(C.RESULT_DIR / "step16_oof_full_full.npy")[row_of],
           "실시간 full학습": rt_probs("full", el, row_of, y),
           "실시간 matched": rt_probs("matched", el, row_of, y),
           "실시간 aug": rt_probs("aug", el, row_of, y)}
    rows, tm = [], []
    extra_done = False
    for name, Pr in src.items():
        acc_h = (Pr[X["sea"]].argmax(1) == g[X["sea"]]).mean()
        b = M.bits_all(X, g, Pr, fold_row, ref_ok)
        S_ = {name: sum(b["soft"][k] for k in M.PRE) * X["sea"]}
        if not extra_done:
            S_["공통 기준"] = sum(b["none"][k] for k in M.PRE) * X["sea"]
            S_["실제 업종"] = sum(b["oracle"][k] for k in M.PRE) * X["sea"]
            extra_done = True
        del b
        for sn, s in S_.items():
            thr = np.quantile(s[neg_mask], M.ALERT_Q)
            r, sk = skew(s, neg_mask, g, thr)
            rec = dict(점수=sn, 바다시간_정확도=acc_h if sn == name else np.nan, 쏠림=sk, **{f"천시간당_{gr}": x for gr, x in zip(M.GEARS, r)})
            for pn, (a_, b_) in PERIODS.items():
                _, sk_p = skew(s, neg_mask, g, thr, (el >= a_) & (el < b_))
                rec[f"쏠림_{pn}"] = sk_p
            rows.append(rec)
            for lab, gp in pos.items():
                r = []
                for p in gp.itertuples():
                    a, e = sl[p.rid]
                    m = X["sea"][a:e]
                    ss = s[a:e][m]
                    if len(ss) == 0:
                        continue
                    rel = np.abs((X["ts"][a:e][m] - np.datetime64(p.label_ts)) / np.timedelta64(1, "h"))
                    near = rel <= 1
                    r.append(dict(hit=rel[ss.argmax()] <= 6, rand=(rel <= 6).mean(), pct=(ss < ss[near].max()).mean() if near.any() else np.nan))
                r = pd.DataFrame(r)
                tm.append(dict(점수=sn, 라벨=lab, n=len(r), 적중6h=int(r["hit"].sum()), 무작위기대=r["rand"].sum(), 순위백분위_중앙=r["pct"].median()))
        print(f"{name} 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    E, T = pd.DataFrame(rows), pd.DataFrame(tm)
    E.round(4).to_csv(C.RESULT_DIR / f"step16_stage2_{STAMP}.csv", index=False, encoding="utf-8-sig")
    T.round(4).to_csv(C.RESULT_DIR / f"step16_stage2_time_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(E.round(3).to_string(index=False), flush=True)
    print(T.round(3).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
