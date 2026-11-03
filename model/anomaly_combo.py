"""
[ANOM-9] 이상 점수 조합 비교 ─ 결합 방식 × 구성 요소 전부 같은 라벨·같은 지표로 비교하고, 선박 단위 중첩 검증으로 정직하게 고름
  구성 요소 (모두 실시간: 시각 t 점수는 t 까지 기록만 사용, 학습 기준 선박으로 고정한 백분위 RefPct 로 0~1 변환)
    if_type : 업종별 Isolation Forest                      (anomaly_rt)
    own_dev : 그 배 직전까지 기준선 대비 이탈 robust z       (anomaly_rt)
    gru     : 직전 12시간으로 다음 시간 예측오차              (anomaly_gru --rt, 없으면 기존 gru_err)
    cpd     : 변화점 탐지 ─ 피처별 부호 있는 z 의 양방향 CUSUM. 한 시간만 튀는 것보다 "행동이 바뀌어 계속되는" 구간에 반응
              S+ = max(0, S+ + z - K), S- = max(0, S- - z - K), 선박이 바뀌거나 기록이 GAP_RESET 시간 넘게 끊기면 0 으로
    ae      : 13시간 창 복원오차 1D-CNN Autoencoder          (anomaly_ae, 있으면)
  결합 방식 : 평균 / 최대(하나라도 강하면) / 기하평균(모두 높아야) × 구성 요소 부분집합 전부
  지표 : anomaly_eval.metrics (AUROC, 선박 내 AUROC, AUPRC, top-1 적중) + 선박 판별 AUROC
         선박 판별 = 라벨 선박(사고 있음) vs 라벨 없는 학습 선박 1,000척의 "배 최고 점수" 구분력 (빈칸 제출 판단용, 라벨 없는 배에도
         미확인 사고가 있을 수 있어 하한 추정)
  선택 : 선박 5-fold 중첩 검증, 학습 fold 의 (AUROC + 선박 내 AUROC)/2 최대 조합 → 검증 fold 점수 (선택 편향 제거)
  제출 : 최종 조합으로 테스트B 채움. 배마다 최고 시각 = 1 이 되게 정규화 (나머지 시각 = 최고 점수 대비 비율)
         submission_combo.xlsx (100척 전부) / _top30, _top50 (배 최고 원점수 상위 30%·50% 만 채우고 나머지 빈칸)
  출력 : results/anom_combo_eval.csv, anom_combo_nested.csv, anom_scores_test_combo.parquet, submission_combo*.xlsx
실행: python anomaly_combo.py
"""
import itertools
import json
import time

import numba
import numpy as np
import openpyxl
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import fe_config as C
from anomaly_eval import labels, metrics
from anomaly_rt import RefPct, fit_if, load_labels, own_z_rt, raw_components, type_baseline

K, GAP_RESET = 1.0, 12      # CUSUM 허용 편차(z), 이 시간 넘게 끊기면 새 항차로 보고 누적 초기화
N_REF, N_NEG = 3000, 1000
COMBOS = ("mean", "max", "geo")
NAME = {"if_type": "IF", "own_dev": "z", "gru": "GRU", "cpd": "변화점", "ae": "AE"}


@numba.njit(cache=True)
def _cusum(z, reset):
    n, d = z.shape
    out = np.zeros((n, d), np.float32)
    sp, sn = np.zeros(d), np.zeros(d)
    for i in range(n):
        if reset[i]:
            sp[:] = 0.0
            sn[:] = 0.0
        for j in range(d):
            v = min(max(z[i, j], -5.0), 5.0)
            sp[j] = min(max(0.0, sp[j] + v - K), 50.0)
            sn[j] = min(max(0.0, sn[j] - v - K), 50.0)
            out[i, j] = max(sp[j], sn[j])
    return out


def cpd(H, typ, base):
    z = own_z_rt(H, typ, base)
    rid = H["rfid_id"].to_numpy()
    reset = np.r_[True, rid[1:] != rid[:-1]] | (H["gap_prev_h"].fillna(0).to_numpy() > GAP_RESET)
    S = _cusum(z.astype(np.float64), reset)
    return np.sort(S, axis=1)[:, -3:].mean(axis=1)


def comps(H, models, typ, base):
    D = raw_components(H, models, typ, base)
    if "gru_rt" not in H and "gru_err" in H:
        D["gru"] = H["gru_err"].to_numpy()
    D["cpd"] = cpd(H, typ, base)
    D["ae"] = H["ae_err"].to_numpy() if "ae_err" in H else np.nan
    return D[[c for c in D if not D[c].isna().all()]]


def combine(P, sub, how):
    M = np.column_stack([P[c] for c in sub])
    if how == "mean":
        return M.mean(1)
    if how == "max":
        return M.max(1)
    return np.exp(np.log(np.clip(M, 1e-3, 1)).mean(1))


def label(sub, how):
    return f"{'+'.join(NAME[c] for c in sub)} {dict(mean='평균', max='최대', geo='기하평균')[how]}"


def vessel_auc(Hv, sv, Hn, sn):
    a = pd.Series(sv).groupby(Hv["rfid_id"].to_numpy()).max()
    b = pd.Series(sn).groupby(Hn["rfid_id"].to_numpy()).max()
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])


def main():
    t0 = time.time()
    Htr = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet")
    wb = openpyxl.load_workbook(C.RESULT_DIR / "submission.xlsx", read_only=True)
    test_ids = [int(r[0]) for r in wb.worksheets[1].iter_rows(min_row=2, values_only=True) if r[0]]
    Hte = pd.read_parquet(C.FEAT_DIR / "anom_hourly_test.parquet")
    keep_te = Hte["rfid_id"].isin(test_ids).to_numpy()
    for H, tag, keep in [(Htr, "", None), (Hte, "_test", keep_te)]:
        for col, f in [("gru_rt", f"anom_gru_rt{tag}.parquet"), ("gru_err", f"anom_gru{tag}.parquet"),
                       ("ae_err", f"anom_ae{tag}.parquet")]:
            if (C.FEAT_DIR / f).exists():
                H[col] = pd.read_parquet(C.FEAT_DIR / f).iloc[:, 0].to_numpy()
    Hte = Hte[keep_te].reset_index(drop=True)
    gru_src = "실시간 gru_rt" if "gru_rt" in Htr else "기존 gru_err (gap_next_h 포함 학습)"
    print(f"테스트B {Hte['rfid_id'].nunique()}척, GRU: {gru_src}, AE: {'ae_err' in Htr}", flush=True)

    models, base = fit_if(Htr), type_baseline(Htr)
    labs = {n: L for n in ["anom_labels.csv", "anom_labels_raw.csv"] if (L := load_labels(n)) is not None}
    lab_ids = set().union(*[set(L["rfid_id"]) for L in labs.values()])
    rng = np.random.default_rng(C.RANDOM_STATE)
    pool = rng.permutation(np.setdiff1d(Htr["rfid_id"].unique(), list(lab_ids)))
    sub_of = lambda ids: Htr[Htr["rfid_id"].isin(ids)].reset_index(drop=True)
    Hr, Hn = sub_of(pool[:N_REF]), sub_of(pool[N_REF:N_REF + N_NEG])
    Dr = comps(Hr, models, Hr["fishing_code"].to_numpy(), base)
    ref = {c: RefPct(Dr[c].to_numpy()) for c in Dr}
    pct = lambda D: {c: ref[c](D[c].to_numpy()) for c in ref}
    Pn = pct(comps(Hn, models, Hn["fishing_code"].to_numpy(), base))
    cs = list(ref)
    cands = [(s, h) for r in range(1, len(cs) + 1) for s in itertools.combinations(cs, r)
             for h in (COMBOS if r > 1 else ("mean",))]
    print(f"구성 요소 {cs}, 후보 {len(cands)}개 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    rows, nested = [], []
    for name, L in labs.items():
        Hv = sub_of(L["rfid_id"].unique())
        Pv = pct(comps(Hv, models, Hv["fishing_code"].to_numpy(), base))
        y, y3 = labels(Hv, L, 1), labels(Hv, L, 3)
        S = {k: combine(Pv, *k) for k in cands}
        tag = dict(labels=name, n_acc=len(L), n_vessel=Hv["rfid_id"].nunique())
        rows.append(tag | dict(method="무작위") | metrics(Hv, rng.random(len(Hv)), y, y3))
        w = (2 * Pv["if_type"] + Pv["own_dev"] + 2 * Pv["gru"]) / 5
        rows.append(tag | dict(method="현재 제출 IF2:z1:GRU2 가중평균") | metrics(Hv, w, y, y3)
                    | dict(vessel_auc=vessel_auc(Hv, w, Hn, (2 * Pn["if_type"] + Pn["own_dev"] + 2 * Pn["gru"]) / 5)))
        for k in cands:
            rows.append(tag | dict(method=label(*k), parts=len(k[0]), how=k[1]) | metrics(Hv, S[k], y, y3)
                        | dict(vessel_auc=vessel_auc(Hv, S[k], Hn, combine(Pn, *k))))

        # 중첩 검증: 조합 선택은 학습 fold 선박으로만
        ids = Hv["rfid_id"].unique()
        crit = lambda s, m: np.mean([roc_auc_score(y[m], s[m]), metrics(Hv[m].reset_index(drop=True), s[m], y[m], y3[m])["auroc_vessel_mean"]])
        oof = np.zeros(len(Hv))
        for f, (tr_i, va_i) in enumerate(KFold(5, shuffle=True, random_state=0).split(ids)):
            tr = Hv["rfid_id"].isin(ids[tr_i]).to_numpy()
            k = max(cands, key=lambda k: crit(S[k], tr))
            oof[~tr] = S[k][~tr]
            nested.append(dict(labels=name, fold=f, chosen=label(*k)))
        rows.append(tag | dict(method="중첩 검증 선택 (편향 제거)") | metrics(Hv, oof, y, y3))
        print(f"[{name}] 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    E = pd.DataFrame(rows)
    E.round(4).to_csv(C.RESULT_DIR / "anom_combo_eval.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nested).to_csv(C.RESULT_DIR / "anom_combo_nested.csv", index=False, encoding="utf-8-sig")
    E["sel"] = (E["auroc"] + E["auroc_vessel_mean"]) / 2
    cols = ["method", "auroc", "auroc_vessel_mean", "auprc", "hit1_3h", "vessel_auc", "sel"]
    for name in labs:
        e = E[E["labels"] == name]
        print(f"\n[{name}] 상위 15 (sel = (AUROC + 선박 내 AUROC)/2)")
        print(e.sort_values("sel", ascending=False)[cols].head(15).round(3).to_string(index=False))
        print(e[e["method"].isin(["무작위", "중첩 검증 선택 (편향 제거)", "현재 제출 IF2:z1:GRU2 가중평균"])
                | (e["parts"] == 1)][cols].round(3).to_string(index=False))
    print(pd.DataFrame(nested).to_string(index=False))

    # 최종 조합: 시각(sel)과 선박 판별(vessel_auc)을 같이 봄 (빈칸 판단에도 같은 점수를 씀, 라벨 v2 있으면 평균)
    #   1위와 0.005 안 차이는 잡음 수준 → 그 안에서 구성 요소가 가장 적은 조합
    E["final"] = (E["sel"] + E["vessel_auc"]) / 2
    g = E.dropna(subset=["parts"]).groupby("method").agg(final=("final", "mean"), parts=("parts", "first"))
    g = g[g["final"] >= g["final"].max() - 0.005].sort_values(["parts", "final"], ascending=[True, False])
    best = next(k for k in cands if label(*k) == g.index[0])
    print(f"\n최종 조합: {g.index[0]} (final {g['final'].iloc[0]:.3f}, 후보 {g.index.tolist()})")
    Pt = pct(comps(Hte, models, Hte["rfid_id"].map(
        pd.read_csv(C.RESULT_DIR / "test_pred_all.csv").set_index("rfid_id")["pred"]).fillna("연안복합").to_numpy(), base))
    Hte["score"] = combine(Pt, *best).astype(np.float32)
    Hte[["rfid_id", "ts", "score"]].to_parquet(C.RESULT_DIR / "anom_scores_test_combo.parquet", index=False)
    for top in (None, 0.3, 0.5):
        write_submission(Hte, top)
    (C.RESULT_DIR / "anom_combo_config.json").write_text(json.dumps(dict(
        combo=g.index[0], parts=list(best[0]), how=best[1], gru=gru_src, cusum_k=K, gap_reset_h=GAP_RESET),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


def write_submission(Hte, top=None):
    """테스트B: 배마다 최고 시각 = 1 로 정규화. top 이 있으면 배 최고 원점수 상위 top 비율만 채우고 나머지는 빈칸"""
    wb = openpyxl.load_workbook(C.RESULT_DIR / "submission.xlsx")
    ws = wb.worksheets[1]
    G = {k: v.set_index("ts")["score"] for k, v in Hte.assign(rid=Hte["rfid_id"].astype(str)).groupby("rid")}
    peak = pd.Series({k: s.max() for k, s in G.items()})
    chosen = set(peak.nlargest(int(round(len(peak) * top))).index) if top else set(peak.index)
    fmt = lambda t: t.strftime("%Y%m%d %H")
    n = 0
    for row in ws.iter_rows(min_row=2):
        rid = str(row[0].value).strip()
        vals = []
        if rid in G and rid in chosen:
            s = G[rid] / G[rid].max()
            t0 = s.idxmax()
            vals = [fmt(t0), 1.0]
            prev = s[(s.index < t0) & (s.index >= t0 - pd.Timedelta(hours=24))].sort_index().tail(3)
            for t, v in prev.items():
                vals += [fmt(t), round(float(v), 4)]
            n += 1
        for cell, v in itertools.zip_longest(row[1:9], vals):
            cell.value = v
    name = f"submission_combo{f'_top{int(top * 100)}' if top else ''}.xlsx"
    wb.save(C.RESULT_DIR / name)
    print(f"테스트B {n}척 채움 → {name}")


if __name__ == "__main__":
    main()
