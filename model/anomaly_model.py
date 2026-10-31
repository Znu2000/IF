"""
[ANOM-3] 시간대 이상 점수 ─ 정상 패턴 학습(비지도) + 사고 패턴 점수 → 테스트B 제출
  구성 요소 (모두 0~1 백분위로 바꿔서 가중 평균)
    if_type  : 업종별 Isolation Forest (학습 선박 시간대로 업종마다 정상 분포 적합)
               테스트 선박 업종 = 과제1 예측 (test_pred_all.csv)
    own_dev  : 그 배의 평소 대비 이탈 (선박별 중앙값·MAD 로 로버스트 z, 큰 3개 평균)
    event    : 사고 패턴 규칙 ─ 바다 위 급정지, 비정상 고속(예인), 바다에서 신호 끊김, 위치 튐, 급귀항
    gru      : 직전 12시간으로 다음 시간을 예측한 오차 (anomaly_gru.py 결과가 있으면 사용)
  사고 직후 신호(예인·귀항·신호 끊김)를 반영하는 [t, t+k] 구간 최대값도 후보 (k 는 검증으로 고름)
  검증 : 사고현황 매칭 라벨(anomaly_labels.py, 등급 A/B) ─ 그 배의 전체 시간 중 사고 ±1시간이
         몇 위인지 (백분위, top-24 적중률). 가중은 거친 격자에서만 고름 (라벨 88건이라 과적합 주의)
  출력 : results/anom_scores_test.parquet, results/submission.xlsx (테스트A 예측 + 테스트B 이상탐지)
실행: python anomaly_model.py
"""
import itertools

import numpy as np
import openpyxl
import pandas as pd
from sklearn.ensemble import IsolationForest

import fe_aggregate as A
import fe_config as C

FEATS = ["n_points", "spike_ratio", "n_jumps", "rg_m", "dist_m", "straightness", "sp_mean", "sp_std", "sp_p90",
         "sp_max", "sp_change_mean", "frac_stop", "frac_fast", "n_stop_move_switch", "heading_R", "turn_mean",
         "uturn_ratio", "ks_speed_med", "ks_frac_drift", "ks_heading_R", "port_km", "home_km", "gap_prev_h",
         "gap_next_h", "d_speed", "jump_prev_km", "home_km_delta", "hod_rarity", "cell_rarity", "hour"]
OWN = ["ks_speed_med", "sp_max", "dist_m", "rg_m", "d_speed", "home_km", "home_km_delta", "jump_prev_km",
       "frac_stop", "turn_mean"]
COMP = ["if_type", "own_dev", "event", "gru"]
SAMPLE = C.PROJECT_ROOT / "문제 데이터_수정_1027" / "테스트셋 및 결과제출 샘플" / "제출샘플.xlsx"


def pct(x):
    return pd.Series(x).rank(pct=True).fillna(0.5).to_numpy()


def prep_X(H):
    X = H[FEATS].astype(np.float32).copy()
    X["gap_prev_h"] = np.log1p(X["gap_prev_h"])
    X["gap_next_h"] = np.log1p(X["gap_next_h"])
    X["hour_sin"], X["hour_cos"] = np.sin(X["hour"] * np.pi / 12), np.cos(X["hour"] * np.pi / 12)
    return X.drop(columns="hour").fillna(0)


def fit_if(Htr, types):
    models = {}
    for t in types:
        X = prep_X(Htr[Htr["fishing_code"] == t])
        X = X.sample(min(len(X), 300_000), random_state=C.RANDOM_STATE)
        models[t] = IsolationForest(n_estimators=200, max_samples=4096, random_state=C.RANDOM_STATE,
                                    n_jobs=C.N_JOBS).fit(X)
    return models


def score_if(H, models, typ):
    s = np.zeros(len(H), np.float32)
    for t, m in models.items():
        idx = np.flatnonzero(typ == t)
        if len(idx):
            s[idx] = -m.score_samples(prep_X(H.iloc[idx]))
    return s


def own_dev(H):
    g = H.groupby("rfid_id")[OWN]
    med = g.transform("median")
    mad = (H[OWN] - med).abs().groupby(H["rfid_id"]).transform("median")
    # 대부분 0인 피처는 MAD=0 → 작은 값도 무한대 z 가 되므로 전체 표준편차의 10% 를 하한으로
    scale = (1.4826 * mad).clip(lower=0.1 * H[OWN].std(), axis=1)
    z = ((H[OWN] - med).abs() / scale).clip(upper=20).fillna(0).to_numpy()
    return np.sort(z, axis=1)[:, -3:].mean(axis=1)


def event(H):
    """사고 패턴 규칙 점수 (각 항목 백분위 평균)"""
    sea = H["at_sea"].to_numpy() == 1
    drop = -H["d_speed"].fillna(0).to_numpy()                       # 급정지 (속도 감소)
    stop_at_sea = sea & np.isin(H["state"].to_numpy(), [A.SEA_STOP, A.DRIFT])
    loss_at_sea = sea * np.log1p(H["gap_next_h"].clip(upper=200).to_numpy())  # 바다에서 신호 끊김
    fast = H["sp_max"].fillna(0).to_numpy()                         # 예인 등 비정상 고속
    jump = H["jump_prev_km"].fillna(0).to_numpy()
    back = -H["home_km_delta"].fillna(0).to_numpy()                 # 급귀항
    parts = [pct(drop) * (0.5 + 0.5 * stop_at_sea), pct(loss_at_sea), pct(fast), pct(jump), pct(back)]
    return np.mean(parts, axis=0)


def forward_max(H, s, k=2):
    """시각 t 점수 = 같은 배 [t, t+k] 시간 범위 안 기록 점수의 최대 (사고 직후 신호 반영)"""
    out = s.copy()
    rid, t = H["rfid_id"].to_numpy(), H["ts"].to_numpy().astype("datetime64[h]").astype(np.int64)
    for j in range(1, k + 1):
        nxt = np.r_[s[j:], np.zeros(j)]
        ok = np.r_[(rid[j:] == rid[:-j]) & (t[j:] - t[:-j] <= k), np.zeros(j, bool)]
        out = np.where(ok, np.maximum(out, nxt), out)
    return out


def components(H, models, typ):
    D = pd.DataFrame(index=H.index)
    D["if_type"] = pct(score_if(H, models, typ))
    D["own_dev"] = pct(own_dev(H))
    D["event"] = pct(event(H))
    D["gru"] = pct(H["gru_err"].to_numpy()) if "gru_err" in H else 0.5
    return D


def evaluate(H, score, L):
    """라벨 선박마다 사고 ±1시간 최고 점수의 그 배 내부 순위"""
    # 동점은 평균 순위 (method="min" 이면 동점 최상위가 모두 1위로 잡혀 과대평가됨)
    r = pd.Series(score, index=H.index).groupby(H["rfid_id"]).rank(ascending=False, method="average")
    res = []
    for _, l in L.iterrows():
        m = (H["rfid_id"] == l["rfid_id"]).to_numpy()
        if not m.any():
            continue
        tt = H.loc[m, "ts"]
        win = (tt >= l["label_ts"] - pd.Timedelta(hours=1)) & (tt <= l["label_ts"] + pd.Timedelta(hours=1))
        if not win.any():
            continue
        best = r[m][win.to_numpy()].min()
        res.append(dict(rank=best, n=m.sum(), pct=1 - best / m.sum()))
    R = pd.DataFrame(res)
    return dict(n=len(R), pct_med=R["pct"].median(), top24=(R["rank"] <= 24).mean(), top5=(R["rank"] <= 5).mean())


def main():
    Htr = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet")
    Hte = pd.read_parquet(C.FEAT_DIR / "anom_hourly_test.parquet")
    for H, tag in [(Htr, ""), (Hte, "_test")]:
        p = C.FEAT_DIR / f"anom_gru{tag}.parquet"
        if p.exists():
            H["gru_err"] = pd.read_parquet(p)["gru_err"].to_numpy()
    print(f"GRU 오차 사용: {'gru_err' in Htr}")
    types = list(C.CLASSES)
    models = fit_if(Htr, types)

    # 검증 (학습 2026 + 사고 매칭 라벨)
    L = pd.read_csv(C.RESULT_DIR / "anom_labels.csv", parse_dates=["label_ts"])
    L = L[L["tier"].isin(["A", "B"]) & L["label_ts"].notna()]
    lab_ids = L["rfid_id"].unique()
    Hv = Htr[Htr["rfid_id"].isin(lab_ids)].reset_index(drop=True)
    Dv = components(Hv, models, Hv["fishing_code"].to_numpy())
    comps = [c for c in COMP if c != "gru" or "gru_err" in Htr]
    rows = []
    for c in comps:
        for fw in (0, 2):
            s = Dv[c].to_numpy()
            rows.append(dict(weights=c, fwd=fw, **evaluate(Hv, forward_max(Hv, s, fw) if fw else s, L)))
    grid = [w for w in itertools.product([0, 1, 2], repeat=len(comps)) if sum(w) > 0]
    for w, fw in itertools.product(grid, (0, 1, 2)):
        s = (Dv[comps].to_numpy() * np.array(w)).sum(1) / sum(w)
        rows.append(dict(weights=str(dict(zip(comps, w))), fwd=fw,
                         **evaluate(Hv, forward_max(Hv, s, fw) if fw else s, L)))
    E = pd.DataFrame(rows)
    E.round(4).to_csv(C.RESULT_DIR / "anom_validation.csv", index=False, encoding="utf-8-sig")
    print(E.head(len(comps) * 2).round(3).to_string(index=False))
    # 제출은 배마다 1위 시각을 고르므로 상위 적중률(top24)과 전체 순위(백분위)를 같이 봄
    G = E.iloc[len(comps) * 2:].assign(sel=lambda d: d["pct_med"] + d["top24"])
    best = G.sort_values("sel", ascending=False).iloc[0]
    print("\n최고 조합:", best.to_dict())
    print(f"(무작위 기준: 백분위 중앙값 ≈ 0.5, top24 ≈ {24 / Hv.groupby('rfid_id').size().median():.3f})")
    w = np.array([eval(best["weights"])[c] for c in comps], float)
    fw = int(best["fwd"])

    # 테스트
    pred = pd.read_csv(C.RESULT_DIR / "test_pred_all.csv", dtype={"rfid_id": str})
    typ = Hte["rfid_id"].astype(str).map(pred.set_index("rfid_id")["pred"]).fillna("연안복합").to_numpy()
    Dt = components(Hte, models, typ)
    s = (Dt[comps].to_numpy() * w).sum(1) / w.sum()
    Hte["score"] = (forward_max(Hte, s, fw) if fw else s).astype(np.float32)
    Hte[["rfid_id", "ts", "score"]].to_parquet(C.RESULT_DIR / "anom_scores_test.parquet", index=False)
    write_submission(Hte)


def write_submission(Hte):
    src = C.RESULT_DIR / "submission_testA.xlsx"
    dst = C.RESULT_DIR / "submission.xlsx"
    wb = openpyxl.load_workbook(src if src.exists() else SAMPLE)
    ws = wb.worksheets[1]
    Hte = Hte.assign(rid=Hte["rfid_id"].astype(str))
    G = {k: v.set_index("ts")["score"] for k, v in Hte.groupby("rid")}
    fmt = lambda t: t.strftime("%Y%m%d %H")
    n = 0
    for row in ws.iter_rows(min_row=2):
        rid = str(row[0].value).strip()
        if rid not in G:
            continue
        s = G[rid]
        t0 = s.idxmax()
        vals = [fmt(t0), round(float(s.max()), 4)]
        # 시각1~3 = 사고시각 직전에 실제 점수를 계산한 기록 3개 (시간순). 대회 설명: "이상항적 점수를 계산한 추적 시간"
        #   24시간보다 오래된 기록은 같은 항적 추적이 아니므로 제외
        prev = s[(s.index < t0) & (s.index >= t0 - pd.Timedelta(hours=24))].sort_index().tail(3)
        for t, v in prev.items():
            vals += [fmt(t), round(float(v), 4)]
        for cell, v in itertools.zip_longest(row[1:9], vals):  # 직전 기록이 3개보다 적으면 빈칸
            cell.value = v
        n += 1
    wb.save(dst)
    print(f"\n테스트B {n}척 채움 → {dst.name}")


if __name__ == "__main__":
    main()
