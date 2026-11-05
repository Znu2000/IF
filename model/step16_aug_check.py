"""
[STEP16-5] 잘린 항적 증강 재확인 ─ step16_rt_class 의 aug 가 전체 항적 cv 도 올린 것(+0.0079)이 진짜인지
  앞 실험과 다른 시드(2, 3)로 같은 fold 를 다시 돌리고, 쓰지 않았던 2025 항적으로 확인 (학습에는 쓰지 않음)
    cv           : 2026 전체 항적, 처음 보는 배
    cross_unseen : fold 모델 → 그 fold 검증 선박의 2025 항적 (시트 선박 제외)
    cross_seen   : 학습 선박 전체 → 겹치는 배의 2025 항적 (시트 선박 제외)
  설정: base(증강 없음) / aug7(전체 + 6가지 길이) / aug_long(전체 + 90·150일) / aug_mid(전체 + 14·30·90일)
  모델 LGB_BASE 클래스 가중 없음. 3모델 재학습 전에 어떤 조합이 값어치 있는지 고르는 용도
  출력: results/step16_aug_<시각>.csv
실행: python step16_aug_check.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import cv_common as V
import fe_config as C
import step11_train26 as S11
import step16_rt_class as RC
import train_classifiers as TC
from step1_experiments import region_prior
from step4_final import attach

STAMP = time.strftime("%m%d_%H%M")
SEEDS = [2, 3]
CONFIGS = {"base": [], "aug_long": [90, 150], "aug_mid": [14, 30, 90], "aug7": RC.CUTS}


def fit_pred(Xa, ya, Xbs):
    params = dict(TC.LGB_BASE, class_weight=None)
    out = [np.zeros((len(x), len(C.CLASSES))) for x in Xbs]
    for sd in SEEDS:
        m = lgb.LGBMClassifier(**params, random_state=sd, n_jobs=C.N_JOBS).fit(Xa, ya)
        for i, x in enumerate(Xbs):
            out[i] += m.predict_proba(x) / len(SEEDS)
    return out


def main():
    t0 = time.time()
    S, y, cols, Sc = RC.load_all()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    attach(T, "_test")
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    sheets = pd.read_excel(S11.SAMPLE, sheet_name=None, dtype={"rfid_id": str})
    scored = set().union(*[set(d["rfid_id"].astype(str)) for d in sheets.values()])
    rid = S["rfid_id"].astype(str)
    lab = dict(zip(rid, y))
    T["_rid"] = T["rfid_id"].astype(str)
    ov = T[T["_rid"].isin(lab) & ~T["_rid"].isin(scored)].reset_index(drop=True)
    y_ov = ov["_rid"].map(lab).to_numpy()
    ov_fold = ov["_rid"].map(dict(zip(rid, fold_id))).to_numpy()
    print(f"참고 2025 {len(ov):,}척 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    def design(X, ref, yr, self_in):
        X = X.reset_index(drop=True)
        return pd.concat([X.reindex(columns=cols), region_prior(ref, yr, X, self_in_ref=self_in).reset_index(drop=True)], axis=1)

    res = {}
    for name, cuts in CONFIGS.items():
        oof, cu = np.zeros((len(y), len(C.CLASSES))), np.zeros((len(ov), len(C.CLASSES)))
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), va)
            Str, ytr = S.iloc[tr].reset_index(drop=True), y[tr]
            Xa = pd.concat([design(Str, Str, ytr, True)] + [design(Sc[n].iloc[tr], Str, ytr, True) for n in cuts], ignore_index=True)
            oof[va], cu[ov_fold == f] = fit_pred(Xa, np.tile(ytr, 1 + len(cuts)),
                                                 [design(S.iloc[va], Str, ytr, False), design(ov[ov_fold == f], Str, ytr, False)])
            print(f"{name} fold {f} ({(time.time() - t0) / 60:.1f}분)", flush=True)
        Xa = pd.concat([design(S, S, y, True)] + [design(Sc[n], S, y, True) for n in cuts], ignore_index=True)
        cs = fit_pred(Xa, np.tile(y, 1 + len(cuts)), [design(ov, S, y, False)])[0]
        res[name] = (oof, cu, cs)
        np.save(C.RESULT_DIR / f"step16_augcheck_oof_{name}.npy", oof)

    rng = np.random.default_rng(0)
    b_oof, b_cu, b_cs = res["base"]
    rows = []
    for name, (oof, cu, cs) in res.items():
        d = (oof.argmax(1) == y).astype(float) - (b_oof.argmax(1) == y)
        du = (cu.argmax(1) == y_ov).astype(float) - (b_cu.argmax(1) == y_ov)
        ds = (cs.argmax(1) == y_ov).astype(float) - (b_cs.argmax(1) == y_ov)
        boot = lambda x: x[rng.integers(0, len(x), (3000, len(x)))].mean(1).std()
        rows.append(dict(설정=name, cv=(oof.argmax(1) == y).mean(), macro=V.macro_f1(y, oof.argmax(1)), logloss=log_loss(y, oof), 차이=d.mean(), SE=boot(d),
                         fold차=" ".join(f"{d[fold_id == f].mean():+.3f}" for f in range(5)),
                         cross_unseen=(cu.argmax(1) == y_ov).mean(), 차이_u=du.mean(), SE_u=boot(du), ll_u=log_loss(y_ov, cu),
                         cross_seen=(cs.argmax(1) == y_ov).mean(), 차이_s=ds.mean(), SE_s=boot(ds), ll_s=log_loss(y_ov, cs)))
    E = pd.DataFrame(rows)
    E.round(4).to_csv(C.RESULT_DIR / f"step16_aug_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(E.round(4).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
