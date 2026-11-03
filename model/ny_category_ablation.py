"""
[NY-4] 범주별 피처 구성 실험 (ablation) ─ 나연
  feature_taxonomy.py 의 대분류(13개) 단위로 피처 묶음을 통째로 빼거나 골라 넣어 비교
  → "어떤 종류의 행동 정보가 업종 구분에 필요한가" / "빼도 되는(연도에 흔들리는) 범주가 있는가"
  규칙: 학습은 2026 항적만. 2025 는 검증에만 사용.
  채점: 선박 단위 5-fold (팀과 같은 fold)
    · cv26 : 검증 선박의 2026 항적 (팀 기존 방식)
    · val25: 검증 선박의 2025 항적 (2026 라벨) ─ 테스트와 같은 연도
  모델: LightGBM (팀 lgbm_base 설정), 지표: 정확도(리더보드 micro-F1) / macro-F1
실행: python ny_category_ablation.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import cv_common as V
import fe_config as C
from ny_common import LGB, load

HYP = ["재방문·어장반복(A)", "경로형태·직선투망(B)", "하루 리듬·시각(E)", "일 유형·이질성(G)", "계절(H)"]
YEAR_SENSITIVE = ["계절(H)", "자연 주기 (v2)", "품질·기록량"]


def configs(cat):
    cats = sorted(cat.unique(), key=lambda c: -(cat == c).sum())
    allf = list(cat.index)
    out = {"전체 (413)": allf}
    for c in cats:
        out[f"− {c}"] = [f for f in allf if cat[f] != c]
    out["가설 범주만 (A·B·E·G·H)"] = [f for f in allf if cat[f] in HYP]
    out["가설 범주 + 위치"] = [f for f in allf if cat[f] in HYP + ["위치(LOC)"]]
    out["− 연도 민감 (계절·자연주기·품질)"] = [f for f in allf if cat[f] not in YEAR_SENSITIVE]
    out["− 연도 민감 − 위치"] = [f for f in allf if cat[f] not in YEAR_SENSITIVE + ["위치(LOC)"]]
    out["속도 범주만"] = [f for f in allf if cat[f] == "속도 구조(C)"]
    return out


def main():
    t0 = time.time()
    S, T25, _ = load()
    S = S.reset_index(drop=True)
    y_of = {c: i for i, c in enumerate(C.CLASSES)}
    y = S["fishing_code"].map(y_of).to_numpy()
    y25 = T25["fishing_code"].map(y_of).to_numpy()
    F = V.folds(S, y)
    cols = [c for c in V.feature_pool(S, "B") if c in T25.columns and c != "year"]
    cat = pd.Series({c: V.feat_meta(c)[1] for c in cols})
    print("범주별 피처 수:", cat.value_counts().to_dict())

    rows, oof = [], {}
    for name, fs in configs(cat).items():
        P26, P25 = np.zeros((len(S), 4)), np.zeros((len(T25), 4))
        for tr, va in F:
            m = lgb.LGBMClassifier(**LGB).fit(S.iloc[tr][fs], y[tr])
            P26[va] = m.predict_proba(S.iloc[va][fs])
            mv = T25["rfid_id"].isin(S["rfid_id"].iloc[va]).to_numpy()
            P25[mv] = m.predict_proba(T25.loc[mv, fs])
        p26, p25 = P26.argmax(1), P25.argmax(1)
        rows.append(dict(config=name, n_feats=len(fs),
                         acc_cv26=accuracy_score(y, p26), macro_cv26=f1_score(y, p26, average="macro"),
                         acc_val25=accuracy_score(y25, p25), macro_val25=f1_score(y25, p25, average="macro"),
                         **{f"f1_{c}_25": v for c, v in zip(C.CLASSES, f1_score(y25, p25, average=None))}))
        oof[name] = (P26, P25)
        print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False, header=len(rows) == 1),
              f" ({(time.time() - t0) / 60:.1f}분)", flush=True)

    # 전체 대비 차이의 부트스트랩 95% 신뢰구간 (선박 재표집)
    rng = np.random.default_rng(0)
    b26 = (oof["전체 (413)"][0].argmax(1) == y).astype(float)
    b25 = (oof["전체 (413)"][1].argmax(1) == y25).astype(float)
    i26 = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    i25 = [rng.integers(0, len(y25), len(y25)) for _ in range(1000)]
    for r in rows:
        d26 = (oof[r["config"]][0].argmax(1) == y) - b26
        d25 = (oof[r["config"]][1].argmax(1) == y25) - b25
        r["d_acc_cv26"], r["d_acc_val25"] = d26.mean(), d25.mean()
        bs26, bs25 = [d26[i].mean() for i in i26], [d25[i].mean() for i in i25]
        r["ci_cv26"] = f"[{np.percentile(bs26, 2.5):+.3f}, {np.percentile(bs26, 97.5):+.3f}]"
        r["ci_val25"] = f"[{np.percentile(bs25, 2.5):+.3f}, {np.percentile(bs25, 97.5):+.3f}]"
    R = pd.DataFrame(rows)
    R.round(4).to_csv(C.RESULT_DIR / "ny_category_ablation.csv", index=False, encoding="utf-8-sig")
    show = ["config", "n_feats", "acc_cv26", "d_acc_cv26", "ci_cv26", "acc_val25", "d_acc_val25", "ci_val25",
            "macro_val25"]
    print("\n", R[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
