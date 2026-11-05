"""
[STEP16-2] 보고 빈도 편향 점검 ─ 경보가 보고 빈도가 낮은 배·시간에 몰리는가
  배경: step15_diag 에서 ks_speed_med 가 시간당 점 수에 좌우됨을 확인 (같은 실제 속도에서 30점 이하 0.64 / 60점 초과 1.78).
        상태 판정이 이 속도를 쓰므로 이상 점수도 영향받을 수 있음
  시간당 점 수 구간별 (사고 범위 박스 안 사고 무관 선박 바다 시간)
    · 경보율 (전체 분위 기준 0.1·1·5%) : 구간 최대/최소
    · 배 단위 : 배의 평소 시간당 점 수 vs 경보율 (Spearman)
  편향이 2배를 넘으면 시험: 점 수 구간 안 백분위로 바꿔 구간마다 같은 예산을 주는 보정 (정상 기준으로만 임계값 계산)
    보정 전후 비교: 구간·업종 쏠림, A·AS 재현 vs 같은 배 위약 (step14_recall 과 같은 방식)
  출력: results/step16_rate_<시각>.csv (rfid_id 없음)
실행: python step16_rate.py  (step16_area.py 의 점수 캐시 재사용, 없으면 계산)
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fe_config as C
import step13_stage2 as M
import step14_recall as R14
import step16_area as AR
import step9_labels as L9

STAMP = time.strftime("%m%d_%H%M")
BINS = [0, 10, 20, 30, 60, 100, 1e9]
LABS = ["≤10", "11-20", "21-30", "31-60", "61-100", ">100"]
QS = [0.001, 0.01, 0.05]
BUDGETS = [0.01, 0.05, 0.1, 0.2]
SHIFTS = [-28, -21, -14, -7, 7, 14, 21, 28]


def table(name, s, al_q, bin_i, g, ref):
    rows = []
    for q in al_q:
        thr = np.quantile(s[ref], 1 - q)
        rates = np.array([(s[ref & (bin_i == b)] > thr).mean() if (ref & (bin_i == b)).any() else np.nan for b in range(len(LABS))])
        g_rates = np.array([(s[ref & (g == k)] > thr).mean() for k in range(4)])
        rec = dict(점수=name, 예산=q, **{f"점수구간{l}": r / q for l, r in zip(LABS, rates)})
        rec["구간_최대최소"] = np.nanmax(rates) / max(np.nanmin(rates), 1e-6)
        rec["업종_최대최소"] = g_rates.max() / max(g_rates.min(), 1e-6)
        rows.append(rec)
    return rows


def recall(s, sea, ts, pos, sl, ref, name):
    rows = []
    for q in BUDGETS:
        al = (s > np.quantile(s[ref], 1 - q)) & sea
        hit, plc = [], []
        for p in pos.itertuples():
            a, e = sl[p.rid]
            t = np.datetime64(p.label_ts)
            hit.append(R14.window_any(al, ts, sea, t, 6, a, e)[0])
            ph = [h for h, on in (R14.window_any(al, ts, sea, t + np.timedelta64(d, "D"), 6, a, e) for d in SHIFTS) if on]
            plc.append(np.mean(ph) if ph else np.nan)
        rows.append(dict(점수=name, 예산=q, 적중=int(np.sum(hit)), 사고=len(pos), 재현=np.mean(hit), 위약=np.nanmean(plc)))
    return rows


def main():
    t0 = time.time()
    H, S, y, fold, P = M.load()
    Z = AR.scores(H, S, y, fold, P)
    np_ = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "ts", "n_points"])
    np_["rid"] = np_["rfid_id"].astype(str)
    np_["ts"] = pd.to_datetime(np_["ts"])
    n = H[["rid", "ts"]].merge(np_[["rid", "ts", "n_points"]], on=["rid", "ts"], how="left")["n_points"].to_numpy()
    print(f"n_points 결측 {np.isnan(n).mean():.4f} ({(time.time() - t0) / 60:.1f}분)", flush=True)

    row_of = pd.Series(np.arange(len(S)), index=S["rid"]).reindex(H["rid"].to_numpy()).to_numpy()
    g = y[row_of]
    sea = H["at_sea"].fillna(0).to_numpy() == 1
    acc = AR.load_acc()
    H_ = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "home_lat", "home_lon"])
    box = (H_["home_lat"].between(acc["위도(변환)"].min() - 0.2, acc["위도(변환)"].max() + 0.2)
           & H_["home_lon"].between(acc["경도(변환)"].min() - 0.2, acc["경도(변환)"].max() + 0.2))
    neg = set(H_.loc[box, "rfid_id"].astype(str)) - AR.touched()
    v = H["rid"].to_numpy()
    ref = np.isin(v, list(neg)) & sea
    bin_i = np.digitize(np.nan_to_num(n, nan=1), BINS[1:-1], right=True)
    # 바다 시간 분포
    dist = pd.Series(bin_i[ref]).value_counts(normalize=True).sort_index()
    print("바다 시간 점수 구간 비율:", {LABS[i]: round(x, 3) for i, x in dist.items()}, flush=True)

    S_ = {"soft rt": Z["soft_rt"], "soft post": Z["soft_post"], "none rt": Z["none_rt"], "급정지": np.where(Z["onset"], 1.0, 0.0)}
    rows = []
    for name, s in S_.items():
        if name == "급정지":
            base = s[ref].mean()
            rates = np.array([s[ref & (bin_i == b)].mean() for b in range(len(LABS))])
            rec = dict(점수=name, 예산=base, **{f"점수구간{l}": r / base for l, r in zip(LABS, rates)})
            rec["구간_최대최소"] = rates.max() / max(rates.min(), 1e-6)
            gr = np.array([s[ref & (g == k)].mean() for k in range(4)])
            rec["업종_최대최소"] = gr.max() / max(gr.min(), 1e-6)
            rows.append(rec)
        else:
            rows += table(name, s, QS, bin_i, g, ref)
    # 구간 내 백분위 보정 (정상 기준)
    adj = R14.pct_by(Z["soft_rt"], ref, bin_i)
    rows += table("soft rt 구간보정", adj, QS, bin_i, g, ref)
    # 구간 × 예측 업종 안 백분위 (경보 쏠림을 두 축 모두 줄이는지)
    hard = (P[row_of] * M.W_HARD).argmax(1)
    adj2 = R14.pct_by(Z["soft_rt"], ref, bin_i * 10 + hard)
    rows += table("soft rt 구간·업종보정", adj2, QS, bin_i, g, ref)
    Tb = pd.DataFrame(rows)
    Tb.round(4).to_csv(C.RESULT_DIR / f"step16_rate_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Tb.round(3).to_string(index=False), flush=True)

    # 배 단위: 평소 시간당 점 수 vs 경보율
    thr = np.quantile(Z["soft_rt"][ref], 0.99)
    D_ = pd.DataFrame({"v": v[ref], "n": np.nan_to_num(n, nan=1)[ref], "al": Z["soft_rt"][ref] > thr})
    G = D_.groupby("v").agg(n=("n", "median"), rate=("al", "mean"), h=("al", "size"))
    G = G[G["h"] >= 24]
    rho, pv = spearmanr(G["n"], G["rate"])
    print(f"배 단위 (바다 24시간 이상 {len(G):,}척) 평소 점 수 vs 1% 경보율 Spearman {rho:.3f} (p={pv:.1e})", flush=True)
    q = pd.qcut(G["n"], 5, duplicates="drop")
    print(G.groupby(q, observed=True).agg(척=("rate", "size"), 경보율평균=("rate", "mean"), 점수중앙=("n", "median")).round(4).to_string(), flush=True)

    # 재현: 보정 전후
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    sl = M.slices(v)
    pos = L[(L["group"] != "port") & L["tier"].isin(["A", "AS"]) & L["rid"].isin(sl)].drop_duplicates("rid")
    ts = H["ts"].to_numpy()
    rc = recall(Z["soft_rt"] * sea, sea, ts, pos, sl, ref, "soft rt") + recall(adj * sea, sea, ts, pos, sl, ref, "soft rt 구간보정") + recall(adj2 * sea, sea, ts, pos, sl, ref, "soft rt 구간·업종보정")
    Rc = pd.DataFrame(rc)
    Rc.round(4).to_csv(C.RESULT_DIR / f"step16_rate_recall_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Rc.round(3).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
