"""
[STEP5] MiniRocket / MultiRocket ─ 배마다 시계열 2종을 무작위 합성곱 특징으로 바꿔 릿지 분류, 1단계 최종과 스태킹
  daily   : 1~6월 181일 × 채널 9 (바다 시간, 정지·표류·조업·이동 시간, 평균 속력, 이동거리, 항구 거리, 전날까지 찾은 어장 재방문 비율)
  diurnal : 하루 24시간 × 채널 7 (시간대별 바다·이동 비율, 정지·표류·조업·이동 비율, 평균 속력)  ← 출어 시각·작업 리듬
  학습    : fold 별 로켓 특징 → 표준화 → RidgeClassifierCV → 결정값 (OOF / 테스트 평균)
  스태킹  : 기준(--base, 기본 r4_rp) OOF 로그 확률 + 로켓 결정값 → 로지스틱 (cv 는 5-fold 교차 예측)
  출력    : results/step5_rocket_eval.csv, results/step1_{oof,test}_r5_<설정>.npy, step1_results.csv 행 추가
실행: python step5_rocket.py --build        (시계열 생성, features/rocket_{daily,diurnal}{,_test}.npy)
      python step5_rocket.py [--kind mini multi] [--base r4_rp]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import cv_common as V
import fe_config as C
from step3_region import save, score

N_DAYS = 181
CELL = 0.005  # 500m


def _seq(tag, ids):
    H = pd.read_parquet(C.FEAT_DIR / f"anom_hourly{tag}.parquet",
                        columns=["rfid_id", "ts", "at_sea", "state", "moving", "sp_mean", "dist_m", "port_km", "lon_c", "lat_c"])
    ts = pd.to_datetime(H.ts)
    H["d"] = (ts.dt.dayofyear - 1).clip(upper=N_DAYS - 1)
    H["h"] = ts.dt.hour
    H["rid"] = H.rfid_id.astype(str)
    idx = {r: i for i, r in enumerate(ids)}
    H["v"] = H.rid.map(idx)
    H = H.dropna(subset=["v"])
    H["v"] = H.v.astype(int)
    n = len(ids)
    sea = H.at_sea.fillna(0).to_numpy()
    st = H.state.to_numpy()
    D = np.zeros((n, 9, N_DAYS), np.float32)
    np.add.at(D[:, 0], (H.v, H.d), sea)
    for j, s in enumerate((1, 2, 3, 4)):
        np.add.at(D[:, 1 + j], (H.v, H.d), (st == s) & (sea == 1))
    cnt = np.zeros((n, N_DAYS), np.float32)
    mv = (H.moving.fillna(0) == 1).to_numpy()
    np.add.at(cnt, (H.v[mv], H.d[mv]), 1)
    np.add.at(D[:, 5], (H.v[mv], H.d[mv]), H.sp_mean.fillna(0).to_numpy()[mv])
    D[:, 5] /= np.maximum(cnt, 1)
    np.add.at(D[:, 6], (H.v, H.d), np.log1p(H.dist_m.fillna(0).to_numpy() / 1000))
    np.maximum.at(D[:, 7], (H.v, H.d), np.log1p(H.port_km.fillna(0).clip(0, 300).to_numpy()))
    # 그날 조업 칸 중 지난 7일 안에 찾았던 칸 비율 (재방문 리듬)
    F = H[(sea == 1) & np.isin(st, (1, 2, 3))].dropna(subset=["lon_c"])
    F = F.assign(c=np.floor(F.lon_c / CELL).astype(np.int64) * 1_000_000 + np.floor(F.lat_c / CELL).astype(np.int64))
    U = F[["v", "d", "c"]].drop_duplicates().sort_values(["v", "c", "d"])
    same = U.v.eq(U.v.shift()) & U.c.eq(U.c.shift())
    U["rv"] = same & (U.d - U.d.shift() <= 7)
    g = U.groupby(["v", "d"]).rv.mean()
    D[g.index.get_level_values(0), 8, g.index.get_level_values(1)] = g.to_numpy()
    # 24시간 리듬
    Z = np.zeros((n, 7, 24), np.float32)
    nh = np.zeros((n, 24), np.float32)
    np.add.at(nh, (H.v, H.h), 1)
    np.add.at(Z[:, 0], (H.v, H.h), sea)
    np.add.at(Z[:, 1], (H.v, H.h), mv)
    for j, s in enumerate((1, 2, 3, 4)):
        np.add.at(Z[:, 2 + j], (H.v, H.h), (st == s) & (sea == 1))
    np.add.at(Z[:, 6], (H.v, H.h), H.sp_mean.fillna(0).to_numpy())
    Z /= np.maximum(nh, 1)[:, None, :]
    return D, Z


def build():
    S, _ = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    for tag, ids in [("", S.rfid_id.astype(str).tolist()), ("_test", T.rfid_id.astype(str).tolist())]:
        D, Z = _seq(tag, ids)
        np.save(C.FEAT_DIR / f"rocket_daily{tag}.npy", D)
        np.save(C.FEAT_DIR / f"rocket_diurnal{tag}.npy", Z)
        print(f"✅ rocket_daily{tag} {D.shape}, rocket_diurnal{tag} {Z.shape}", flush=True)


def norm(Xtr, *Xs):
    mu = Xtr.mean((0, 2), keepdims=True)
    sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    return [((X - mu) / sd).astype(np.float64) for X in (Xtr, *Xs)]


def rocket(kind, seed):
    if kind == "mini":
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate(num_kernels=10000, random_state=seed)
    from sktime.transformations.panel.rocket import MultiRocketMultivariate
    return MultiRocketMultivariate(num_kernels=6250, random_state=seed)


def main(a):
    t0 = time.time()
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    lab = dict(zip(S.rfid_id.astype(str), y))
    T["_rid"] = T.rfid_id.astype(str)
    ov_mask = T._rid.isin(lab).to_numpy()
    y_ov = T._rid[ov_mask].map(lab).to_numpy()
    views = {v: (np.load(C.FEAT_DIR / f"rocket_{v}.npy"), np.load(C.FEAT_DIR / f"rocket_{v}_test.npy"))
             for v in ("daily", "diurnal")}
    Bo = np.load(C.RESULT_DIR / f"step1_oof_{a.base}.npy")
    Bt = np.load(C.RESULT_DIR / f"step1_test_{a.base}.npy")
    lg = lambda P: np.log(np.clip(P, 1e-6, 1))
    print(f"기준 {a.base}: cv {accuracy_score(y, Bo.argmax(1)):.4f}, cross_seen {accuracy_score(y_ov, Bt[ov_mask].argmax(1)):.4f}",
          flush=True)

    dec = {}
    rows = []
    for kind in a.kind:
        for vname, (Xs, Xt) in views.items():
            oof = np.zeros((len(y), len(C.CLASSES)))
            te = np.zeros((len(T), len(C.CLASSES)))
            for f, va in enumerate(folds):
                tr = np.setdiff1d(np.arange(len(y)), va)
                Xtr, Xva, Xte = norm(Xs[tr], Xs[va], Xt)
                rk = rocket(kind, f).fit(Xtr)
                Ftr, Fva, Fte = (np.asarray(rk.transform(X), np.float32) for X in (Xtr, Xva, Xte))
                sc = StandardScaler().fit(Ftr)
                clf = RidgeClassifierCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(Ftr), y[tr])
                oof[va] = clf.decision_function(sc.transform(Fva))
                te += clf.decision_function(sc.transform(Fte)) / len(folds)
                print(f"  [{kind}/{vname}] fold {f}: acc {accuracy_score(y[va], oof[va].argmax(1)):.4f} "
                      f"(alpha {clf.alpha_:.3g}, {(time.time() - t0) / 60:.1f}분)", flush=True)
            dec[(kind, vname)] = (oof, te)
            rows.append(dict(model=f"{kind}/{vname} 단독", cv=accuracy_score(y, oof.argmax(1)),
                             cv_macro=V.macro_f1(y, oof.argmax(1)),
                             cross_seen=accuracy_score(y_ov, te[ov_mask].argmax(1))))
            print(rows[-1], flush=True)

    cv5 = StratifiedKFold(5, shuffle=True, random_state=1)
    combos = {"base": []} | {f"+{k}/{v}": [(k, v)] for k, v in dec} | {f"+{k}/both": [(k, "daily"), (k, "diurnal")] for k in a.kind}
    if len(a.kind) > 1:
        combos["+all"] = list(dec)
    res = []
    for name, parts in combos.items():
        Xo = np.hstack([lg(Bo)] + [dec[p][0] for p in parts])
        Xt = np.hstack([lg(Bt)] + [dec[p][1] for p in parts])
        st = LogisticRegression(max_iter=3000, C=0.3)
        O = cross_val_predict(st, Xo, y, cv=cv5, method="predict_proba")
        P = st.fit(Xo, y).predict_proba(Xt)
        cfg = f"r5_{name.strip('+').replace('/', '_')}"
        r = score(cfg, O, P[ov_mask], None, y, y_ov, fold_id)
        res.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "weights"}, flush=True)
        if name != "base":
            np.save(C.RESULT_DIR / f"step1_oof_{cfg}.npy", O)
            np.save(C.RESULT_DIR / f"step1_test_{cfg}.npy", P)
    E = pd.DataFrame(res)
    b = E.iloc[0]
    E["d_cv"], E["d_cross_seen_adj"] = E.cv - b.cv, E.cross_seen_adj - b.cross_seen_adj
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(E.drop(columns="weights").round(4).to_string(index=False))
    pd.concat([pd.DataFrame(rows), E]).round(4).to_csv(C.RESULT_DIR / "step5_rocket_eval.csv", index=False,
                                                       encoding="utf-8-sig")
    save([r for r in res[1:]])
    ok = E[(E.d_cross_seen_adj >= 0.005) & (E.d_cv >= 0)]
    print("채택 기준(스태킹 기준 대비 cross_seen_adj +0.005, cv 하락 없음) 통과:", ok.config.tolist() or "없음")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--kind", nargs="*", default=["mini"])
    ap.add_argument("--base", default="r4_rp")
    a = ap.parse_args()
    build() if a.build else main(a)
