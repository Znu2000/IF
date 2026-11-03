"""
[STEP9-7] 사고 167건 전체 활용 ─ 배를 특정하지 않는 두 가지 방법
  (1) 통계 사전확률: 바다 사고(항구 제외)의 시간대·월·해역(0.1도)·업종 분포 ÷ 같은 조건의 바다 시간 수(노출량) = 상대 위험
      상대 위험 = (사고 수 + k) / (노출량 기준 기대 사고 수 + k), k = 5. 사고 단위 5-fold 로 평가 사고를 빼고 계산 (누수 방지)
      점수 = 급정지 점수 × 상대 위험
  (2) 후보 묶음 평가: 사고 위치 3km·±90분 안의 같은 업종 배 전체를 후보로 두고, 후보 중 몇 %가 사고 ±6h 안을 집어내는지 측정
      같은 위치·업종에서 시각만 ±7·14일 옮긴 위약과 비교 → 라벨 확정 없이 탐지기 신호를 잼 (사고 단위 부트스트랩)
      위약은 시간대·월이 거의 같아 시간 사전확률의 효과는 차이에 나타나지 않음 → 실제 시각 절대값도 같이 봄
  S 등급은 흔적(정지)으로 고른 라벨이라 급정지 평가가 순환 → 묶음 평가에서 따로 표시
  출력: results/step9_priors_stat.csv, step9_bag_eval.csv, step9_prior_eval.csv (rfid_id 없음)
실행: python step9_priors.py
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.model_selection import KFold

import fe_aggregate as A
import fe_config as C
import step9_detect as D
import step9_labels as L9

SHIFTS = [-14, -7, 0, 7, 14]
SURE = ["AS", "A", "S"]
BAG_KM, SMOOTH, CELL, N_FOLD, N_BOOT = 3.0, 5.0, 0.1, 5, 2000
VARIANTS = {"급정지": [], "+시간대·월": ["hod", "month"], "+해역": ["cell"], "+시간대·월·해역": ["hod", "month", "cell"]}


def cell_of(lat, lon):
    return np.floor(np.asarray(lat, float) / CELL).astype(np.int64) * 10000 + np.floor(np.asarray(lon, float) / CELL).astype(np.int64)


def rel_risk(acc_x, expo):
    """expo: 조건별 노출량 Series, acc_x: 사고별 조건값"""
    n = pd.Series(acc_x).value_counts().reindex(expo.index, fill_value=0)
    e = len(acc_x) * expo / expo.sum()
    return (n + SMOOTH) / (e + SMOOTH), n, e


def candidates(P, a, t, gear):
    W = P[(P["gear"] == gear) & (P["ts"] >= t.floor("h") - pd.Timedelta(hours=2)) & (P["ts"] <= t.floor("h") + pd.Timedelta(hours=2))]
    if W.empty:
        return []
    lon = W[[f"b_lon{j}" for j in range(6)]].to_numpy()
    lat = W[[f"b_lat{j}" for j in range(6)]].to_numpy()
    tm = W["ts"].to_numpy()[:, None] + (np.arange(6) * 10 + 5).astype("timedelta64[m]")
    ok = np.abs((tm - np.datetime64(t)) / np.timedelta64(1, "m")) <= L9.NEAR_MIN
    d = np.where(ok & ~np.isnan(lon), A.hav_km(lon, lat, a["경도(변환)"], a["위도(변환)"]), np.inf)
    per = pd.Series(d.min(1), index=W["rid"].to_numpy()).groupby(level=0).min()
    return per[per <= BAG_KM].index.tolist()


def main():
    t0 = time.time()
    acc = pd.read_excel(L9.ACC)
    acc["group"] = acc.apply(L9.group, axis=1)
    acc["gear"] = acc["어업면허"].map(lambda g: L9.GEAR.get(g, g))
    acc["hod"] = acc["발생일시"].dt.hour
    acc["month"] = acc["발생일시"].dt.month
    acc["cell"] = cell_of(acc["위도(변환)"], acc["경도(변환)"])
    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    acc["tier"] = L.set_index("acc_idx")["tier"].reindex(acc.index)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))

    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code", "home_lat", "home_lon"])
    S["rid"] = S["rfid_id"].astype(str)
    gear_of = dict(zip(S["rid"], S["fishing_code"]))

    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=D.COLS + ["lat_c", "lon_c"])
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.sort_values(["rid", "ts"], kind="stable").reset_index(drop=True)
    F = D.build(H)
    F["hod"] = H["ts"].dt.hour.to_numpy()
    F["month"] = H["ts"].dt.month.to_numpy()
    F["cell"] = cell_of(H["lat_c"].fillna(0), H["lon_c"].fillna(0))
    F["gear"] = F["rid"].map(gear_of)
    sea = F[F["sea"]]
    expo = {k: sea.groupby(k).size() for k in ["hod", "month", "cell", "gear"]}
    print(f"바다 시간 {len(sea):,}개 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    # (1) 전체 분포 (보고서용)
    A_sea = acc[acc["group"] != "port"].copy()
    rows = []
    for k in ["hod", "month", "gear", "cell"]:
        x = A_sea[k][A_sea[k].isin(expo[k].index)]
        rr, n, e = rel_risk(x.to_numpy(), expo[k])
        keep = (n > 0) | (k != "cell")
        p = chisquare(n, e * n.sum() / e.sum()).pvalue if k != "cell" else np.nan
        print(f"{k}: 사고 {len(x)}건 / 노출 대비 분포 차이 p={p:.3g}")
        rows.append(pd.DataFrame({"차원": k, "값": n.index[keep], "사고": n[keep].to_numpy(), "노출비율": (expo[k] / expo[k].sum())[keep].to_numpy(),
                                  "기대사고": e[keep].to_numpy(), "상대위험": rr[keep].to_numpy(), "분포차이_p": p}))
    pd.concat(rows).round(4).to_csv(C.RESULT_DIR / "step9_priors_stat.csv", index=False, encoding="utf-8-sig")

    # 사고 단위 fold 별 점수 → 선박별 1위 시각·최고 점수
    A_sea["fold"] = -1
    for f, (_, te) in enumerate(KFold(N_FOLD, shuffle=True, random_state=0).split(A_sea)):
        A_sea.iloc[te, A_sea.columns.get_loc("fold")] = f
    E = F[F["onset"]].copy()
    E["s0"] = 1 + E["f_drift"] / 6
    tops, vmaxs = {}, {}
    for f in range(N_FOLD):
        tr = A_sea[A_sea["fold"] != f]
        rr = {k: rel_risk(tr[k][tr[k].isin(expo[k].index)].to_numpy(), expo[k])[0] for k in expo}
        base_cell = SMOOTH / SMOOTH  # 노출 없는 칸 = 사고도 없음 → 1
        for name, dims in VARIANTS.items():
            s = E["s0"].to_numpy().copy()
            for k in dims:
                s = s * E[k].map(rr[k]).fillna(base_cell).to_numpy()
            df = pd.DataFrame({"rid": E["rid"].to_numpy(), "ts": E["ts"].to_numpy(), "s": s})
            tops[f, name] = df.loc[df.groupby("rid")["s"].idxmax()].set_index("rid")["ts"]
            vmaxs[f, name] = df.groupby("rid")["s"].max()
            if name == "+시간대·월·해역":           # 선박 판별에는 업종 위험도 곱한 경우를 추가
                g = pd.Series(df["rid"].unique()).map(gear_of).map(rr["gear"]).fillna(1.0).to_numpy()
                vmaxs[f, "+전부+업종"] = vmaxs[f, name] * pd.Series(g, index=df["rid"].unique()).reindex(vmaxs[f, name].index)
    onset_ts = {r: g.to_numpy() for r, g in E.groupby("rid")["ts"]}
    print(f"점수 계산 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    # (2) 후보 묶음 평가 (실제 시각 vs 옮긴 시각)
    shifted = pd.concat([A_sea.assign(발생일시=A_sea["발생일시"] + pd.Timedelta(days=s)) for s in SHIFTS])
    shifted = shifted[shifted["발생일시"].between("2026-01-01", "2026-06-30 23:59")]
    P = L9.load_points(shifted)
    P["gear"] = P["rid"].map(gear_of)
    bag = []
    for s in SHIFTS:
        for i, a in A_sea.iterrows():
            t = a["발생일시"] + pd.Timedelta(days=s)
            if not pd.Timestamp("2026-01-01") <= t <= pd.Timestamp("2026-06-30 23:59"):
                continue
            cand = candidates(P, a, t, a["gear"])
            if not cand:
                continue
            tn = np.datetime64(t.floor("h"))
            near = lambda ts: np.abs((ts - tn) / np.timedelta64(1, "h")) <= 6
            ev = np.mean([near(onset_ts[r]).any() if r in onset_ts else False for r in cand])
            rec = dict(acc_idx=i, shift=s, group=a["group"], tier=a["tier"], n_cand=len(cand), 급정지있음=ev)
            for name in VARIANTS:
                top = tops[a["fold"], name]
                rec[name] = np.mean([bool(near(np.datetime64(top[r]))) if r in top.index else False for r in cand])
            bag.append(rec)
    B = pd.DataFrame(bag)
    rng = np.random.default_rng(0)
    cols = ["급정지있음"] + list(VARIANTS)
    out = []
    subsets = {"바다 전체": B, "B(애매)": B[B["tier"] == "B"], "A·AS(거리)": B[B["tier"].isin(["A", "AS"])], "S(흔적, 순환)": B[B["tier"] == "S"]}
    for name, g in subsets.items():
        real = g[g["shift"] == 0].set_index("acc_idx")
        plc = g[g["shift"] != 0].groupby("acc_idx")[cols].mean()
        idx = real.index.intersection(plc.index)
        for c in cols:
            d = (real.loc[idx, c] - plc.loc[idx, c]).to_numpy()
            bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)] if len(d) else [np.nan]
            out.append(dict(묶음=name, 지표=c, 사고수=len(idx), 후보수_평균=real.loc[idx, "n_cand"].mean(), 실제=real.loc[idx, c].mean(),
                            위약=plc.loc[idx, c].mean(), 차이=d.mean() if len(d) else np.nan,
                            ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
    Bt = pd.DataFrame(out)
    Bt.round(4).to_csv(C.RESULT_DIR / "step9_bag_eval.csv", index=False, encoding="utf-8-sig")
    print("\n[후보 묶음] 후보 중 사고 ±6h 를 집어낸 비율 (급정지있음 = ±6h 안에 급정지 사건이 하나라도)")
    print(Bt.round(3).to_string(index=False))

    # (3) 확실 라벨 기준 시각 적중 · 선박 판별 (평가 사고는 사전확률에서 제외)
    pos = L[L["tier"].isin(SURE) & (L["group"] != "port")].drop_duplicates("rid")[["acc_idx", "rid", "label_ts"]].copy()
    pos["fold"] = A_sea["fold"].reindex(pos["acc_idx"]).to_numpy()
    act = F.groupby("rid")["sea"].sum()
    box = (S["home_lat"].between(acc["위도(변환)"].min() - 0.2, acc["위도(변환)"].max() + 0.2)
           & S["home_lon"].between(acc["경도(변환)"].min() - 0.2, acc["경도(변환)"].max() + 0.2))
    in_box = set(S.loc[box, "rid"])
    all_neg = np.array(sorted(set(F["rid"]) - touched))
    reg_neg = np.array([r for r in all_neg if r in in_box])
    res = []
    for name in list(VARIANTS) + ["+전부+업종"]:
        rec = dict(점수=name, n=len(pos))
        if name in VARIANTS:
            for h in (1, 3, 6):
                rec[f"적중{h}h_건"] = sum(1 for p in pos.itertuples() if p.rid in tops[p.fold, name].index
                                       and abs((tops[p.fold, name][p.rid] - p.label_ts) / pd.Timedelta(hours=1)) <= h)
        for lab, neg in [("전체", all_neg), ("지역", reg_neg)]:
            aucs, ws = [], []
            for f, g in pos.groupby("fold"):
                vid = np.r_[g["rid"].to_numpy(), neg]
                y = np.r_[np.ones(len(g)), np.zeros(len(neg))]
                v = vmaxs[f, name].reindex(vid).fillna(0).to_numpy()
                a = act.reindex(vid).fillna(0).to_numpy()
                aucs.append(D.strat_auc(y, v, a, nbin=5))
                ws.append(len(g))
            rec[f"활동량보정AUC_{lab}"] = np.average(aucs, weights=ws)
        res.append(rec)
    Rt = pd.DataFrame(res)
    Rt.round(4).to_csv(C.RESULT_DIR / "step9_prior_eval.csv", index=False, encoding="utf-8-sig")
    print("\n[확실 라벨, 바다] 시각 적중 건수 · 활동량 보정 선박 판별")
    print(Rt.round(3).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
