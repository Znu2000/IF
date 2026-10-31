"""
[ANOM-5] 이상탐지 검증 ─ AUROC / AUPRC / F1 (공식 지표 미공개 → 대표 지표 모두 측정)
  라벨   : anomaly_labels.py 매칭 사고 (등급 A/B). 양성 = 매칭 선박의 사고 ±1시간 기록
  대상   : 라벨 선박들의 전체 시간대 (선박당 수백 시간 중 사고 시각을 골라내는 테스트B 상황과 같음)
  지표
    AUROC / AUPRC        : 시간대 단위, 전체 풀링 (AUPRC 무작위 기준 = 양성 비율)
    AUROC (선박 내 평균) : 선박마다 따로 계산 후 평균 (그 배 안에서 사고 시각을 구분하는 능력)
    F1 (임계값)          : 학습 fold 에서 F1 최대 임계값을 정해 검증 fold 에 적용
    F1 (top-1)           : 제출 방식 그대로 배마다 1위 시각 1개만 양성으로 예측 → precision·recall·F1
    hit@1 ±1h / ±3h      : 배마다 1위 시각이 사고 ±1·±3시간 안인 비율
  가중 선택 편향 제거 : 선박 단위 5-fold 중첩 검증 (가중·임계값은 학습 fold 로만 고르고 검증 fold 에서 측정)
  출력 : results/anom_eval.csv
실행: python anomaly_eval.py
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold

import fe_config as C
from anomaly_model import COMP, components, fit_if, forward_max


def labels(H, L, tol=1):
    y = np.zeros(len(H), bool)
    for _, l in L.iterrows():
        m = (H["rfid_id"] == l["rfid_id"]).to_numpy()
        dt = (H["ts"] - l["label_ts"]).abs() <= pd.Timedelta(hours=tol)
        y |= m & dt.to_numpy()
    return y


def top1(H, s):
    """배마다 점수 1위 기록의 위치 (동점이면 앞 시각)"""
    return pd.Series(s, index=H.index).groupby(H["rfid_id"]).idxmax().to_numpy()


def metrics(H, s, y, y3, thr=None):
    out = dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), base_rate=y.mean())
    per = [roc_auc_score(y[m], s[m]) for _, m in H.groupby("rfid_id").indices.items()
           if 0 < y[m].sum() < len(m)]
    out["auroc_vessel_mean"] = np.mean(per)
    if thr is not None:
        out["f1_thr"] = f1_score(y, s >= thr)
    t = top1(H, s)
    pred = np.zeros(len(H), bool)
    pred[t] = True
    out["prec_top1"], out["rec_top1"] = y[pred].mean(), (pred & y).sum() / y.sum()
    out["f1_top1"] = f1_score(y, pred)
    out["hit1_1h"], out["hit1_3h"] = y[t].mean(), y3[t].mean()
    return out


def best_thr(s, y):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return t[np.argmax(f[:-1])]


def main():
    Htr = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet")
    p = C.FEAT_DIR / "anom_gru.parquet"
    if p.exists():
        Htr["gru_err"] = pd.read_parquet(p)["gru_err"].to_numpy()
    L = pd.read_csv(C.RESULT_DIR / "anom_labels.csv", parse_dates=["label_ts"])
    L = L[L["tier"].isin(["A", "B"]) & L["label_ts"].notna()]
    models = fit_if(Htr, list(C.CLASSES))
    Hv = Htr[Htr["rfid_id"].isin(L["rfid_id"].unique())].reset_index(drop=True)
    D = components(Hv, models, Hv["fishing_code"].to_numpy())
    comps = [c for c in COMP if c != "gru" or "gru_err" in Htr]
    y, y3 = labels(Hv, L, 1), labels(Hv, L, 3)
    print(f"라벨 선박 {Hv['rfid_id'].nunique()}척, 시간대 {len(Hv):,}개, 양성 {y.sum()} (비율 {y.mean():.4f})")

    rows = []
    rng = np.random.default_rng(0)
    rows.append(dict(method="무작위", subset="A+B") | metrics(Hv, rng.random(len(Hv)), y, y3))
    rows.append(dict(method="단순: 모항에서 가장 먼 시각", subset="A+B") | metrics(Hv, Hv["home_km"].fillna(0).to_numpy(), y, y3))
    rows.append(dict(method="단순: 최고 속도", subset="A+B") | metrics(Hv, Hv["sp_max"].fillna(0).to_numpy(), y, y3))
    for c in comps:
        rows.append(dict(method=c, subset="A+B") | metrics(Hv, D[c].to_numpy(), y, y3))
    w_sub = np.array([2, 1, 0, 2][:len(comps)], float)  # 현재 제출 조합 (같은 라벨로 고름 → 낙관적)
    s_sub = (D[comps].to_numpy() * w_sub).sum(1) / w_sub.sum()
    rows.append(dict(method="제출 조합 IF2:z1:GRU2 (선택 편향 포함)", subset="A+B") | metrics(Hv, s_sub, y, y3))

    # 중첩 검증: 선박 5-fold, 가중·전방창·임계값은 학습 fold 의 AUPRC/F1 로 선택
    ids = Hv["rfid_id"].unique()
    grid = [w for w in itertools.product([0, 1, 2], repeat=len(comps)) if sum(w) > 0]
    cand = {}
    for w, fw in itertools.product(grid, (0, 2)):
        s = (D[comps].to_numpy() * np.array(w)).sum(1) / sum(w)
        cand[(w, fw)] = forward_max(Hv, s, fw) if fw else s
    oof = np.zeros(len(Hv))
    thr_hit = np.zeros(len(Hv), bool)
    chosen = []
    for k, (tr_i, va_i) in enumerate(KFold(5, shuffle=True, random_state=0).split(ids)):
        tr = Hv["rfid_id"].isin(ids[tr_i]).to_numpy()
        va = ~tr
        key = max(cand, key=lambda c: average_precision_score(y[tr], cand[c][tr]))
        chosen.append(key)
        oof[va] = cand[key][va]
        thr_hit[va] = cand[key][va] >= best_thr(cand[key][tr], y[tr])
    r = dict(method="앙상블 (중첩 검증, 편향 제거)", subset="A+B") | metrics(Hv, oof, y, y3)
    r["f1_thr"] = f1_score(y, thr_hit)
    rows.append(r)
    print("fold 별 선택 조합:", [dict(zip(comps, c[0])) | {"fwd": c[1]} for c in chosen])

    # 확실한 라벨(A)만
    LA = L[L["tier"] == "A"]
    mA = Hv["rfid_id"].isin(LA["rfid_id"]).to_numpy()
    HA = Hv[mA].reset_index(drop=True)
    yA, y3A = labels(HA, LA, 1), labels(HA, LA, 3)
    for name, s in [("무작위", rng.random(mA.sum())), ("앙상블 (중첩 검증, 편향 제거)", oof[mA])] + \
                   [(c, D[c].to_numpy()[mA]) for c in comps]:
        rows.append(dict(method=name, subset="A만") | metrics(HA, s, yA, y3A))

    E = pd.DataFrame(rows)
    E.round(4).to_csv(C.RESULT_DIR / "anom_eval.csv", index=False, encoding="utf-8-sig")
    cols = ["method", "subset", "auroc", "auroc_vessel_mean", "auprc", "base_rate", "f1_thr", "f1_top1",
            "prec_top1", "rec_top1", "hit1_1h", "hit1_3h"]
    print(E[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
