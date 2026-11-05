"""
[STEP13-2] "확신 없음" 5번째 클래스 ─ 1단계 최대 확률이 임계값 이상이면 그 업종 기준표, 미만이면 미정 기준표로 이상 점수 계산
  미정 기준표 2가지
    공통 : 업종 무시 전체 기준표 (step13 의 none)
    집단 : 다른 fold 의 미정 선박들로만 만든 기준표 (미정 배들끼리의 평소)
  임계값은 라벨로 고르지 않고 전부 보고. 점수는 rt (시각 t 까지만) 만 사용
  출력: results/step13_abstain_{cover,time,alert,vessel,bag}_<시각>.csv (rfid_id 없음)
실행: python step13_abstain.py
"""
import time

import numpy as np
import pandas as pd

import fe_config as C
import step13_stage2 as M
import step9_labels as L9

TAUS = [0.5, 0.6, 0.7, 0.8]
STAMP = time.strftime("%m%d_%H%M")


def coverage(P, y):
    pick, conf = P.argmax(1), P.max(1)
    rows = []
    for t in TAUS:
        m = conf >= t
        rec = dict(임계값=t, 확신비율=m.mean(), 확신_정확도=(pick[m] == y[m]).mean(), 미정_정확도=(pick[~m] == y[~m]).mean())
        for k, g in enumerate(M.GEARS):
            rec[f"확신비율_{g}"] = m[y == k].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def abstain_scores(X, g, P_row, fold_row, ref_ok):
    pi = P_row / P_row.sum(1, keepdims=True)
    pick, conf = pi.argmax(1), pi.max(1)
    n = len(X["v"])
    b_pick, b_pool = np.zeros(n, np.float32), np.zeros(n, np.float32)
    b_grp = {t: np.zeros(n, np.float32) for t in TAUS}
    for f in np.unique(fold_row):
        i = np.flatnonzero(fold_row == f)
        ref = (fold_row != f) & ref_ok
        L = M.lookup(M.tables(X, ref, g), X, i)
        for k in M.PRE:
            b_pick[i] += -np.log2(L[k][np.arange(len(i)), pick[i]])
            b_pool[i] += -np.log2(L[k][:, M.NG])
        for t in TAUS:
            Lu = M.lookup(M.tables(X, ref & (conf < t), g), X, i)
            for k in M.PRE:
                b_grp[t][i] += -np.log2(Lu[k][:, M.NG])
    out = {}
    for t in TAUS:
        out[f"미정공통{t} rt"] = np.where(conf >= t, b_pick, b_pool) * X["sea"]
        out[f"미정집단{t} rt"] = np.where(conf >= t, b_pick, b_grp[t]) * X["sea"]
    return out


def main():
    t0 = time.time()
    H, S, y, fold, P = M.load()
    X = M.context(H)
    row_of = pd.Series(np.arange(len(S)), index=S["rid"]).reindex(X["v"]).to_numpy()
    g, fold_row, P_row = y[row_of], fold[row_of], P[row_of]
    Cv = coverage(P / P.sum(1, keepdims=True), y)
    Cv.round(4).to_csv(C.RESULT_DIR / f"step13_abstain_cover_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Cv.round(3).to_string(index=False), flush=True)

    L = pd.read_csv(C.RESULT_DIR / "anom_labels_v2.csv", parse_dates=["label_ts"])
    L["rid"] = pd.to_numeric(L["rfid_id"], errors="coerce").astype("Int64").astype(str)
    v1 = pd.read_csv(C.RESULT_DIR / "anom_labels.csv")
    touched = set(L["rid"]) | set(pd.to_numeric(v1["rfid_id"], errors="coerce").dropna().astype("int64").astype(str))
    ref_ok = ~np.isin(X["v"], list(touched))
    sl = M.slices(X["v"])
    sea_l = L[(L["group"] != "port") & L["rid"].isin(sl)]
    pos = {"A·AS(거리)": sea_l[sea_l["tier"].isin(["A", "AS"])].drop_duplicates("rid"),
           "A·AS·S(순환 포함)": sea_l[sea_l["tier"].isin(["A", "AS", "S"])].drop_duplicates("rid")}
    neg = np.array(sorted(set(sl) - touched))
    acc = pd.read_excel(L9.ACC)
    H_ = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "home_lat", "home_lon"])
    box = (H_["home_lat"].between(acc["위도(변환)"].min() - 0.2, acc["위도(변환)"].max() + 0.2)
           & H_["home_lon"].between(acc["경도(변환)"].min() - 0.2, acc["경도(변환)"].max() + 0.2))
    neg_r = np.array(sorted(set(neg) & set(H_.loc[box, "rfid_id"].astype(str))))

    b = M.bits_all(X, g, P_row, fold_row, ref_ok)
    S_ = {f"{m} rt": sum(b[m][k] for k in M.PRE) * X["sea"] for m in M.METHODS}
    S_ |= abstain_scores(X, g, P_row, fold_row, ref_ok)
    print(f"점수 계산 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    Al = M.alert_eval(S_, X, g, neg, pos["A·AS·S(순환 포함)"], sl)
    Al.round(4).to_csv(C.RESULT_DIR / f"step13_abstain_alert_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Al.round(3).to_string(index=False), flush=True)
    Tm = M.time_eval(S_, X, pos, sl)
    Tm.round(4).to_csv(C.RESULT_DIR / f"step13_abstain_time_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Tm.round(3).to_string(index=False), flush=True)
    Ve = M.vessel_eval(S_, X, pos, neg, sl).merge(M.vessel_eval(S_, X, pos, neg_r, sl), on=["점수", "라벨", "n"], suffixes=("", "_지역"))
    Ve.round(4).to_csv(C.RESULT_DIR / f"step13_abstain_vessel_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Ve.round(3).to_string(index=False), flush=True)
    B = M.bag_candidates()
    B["group2"] = np.where(B["group"] == "port", "port", "sea")
    Bg = M.bag_eval(S_, X, B, sl, np.random.default_rng(0))
    Bg.round(4).to_csv(C.RESULT_DIR / f"step13_abstain_bag_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Bg.round(3).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
