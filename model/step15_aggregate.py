"""
[STEP15-3] 전처리 집계 변형 ─ fe_hourly 결과는 그대로 두고 상태 판정·항구 규칙만 바꿔 선박 피처를 다시 만들고, 빠른 LightGBM 으로 비교
  변형 (전처리 조건변경 판단.pdf 후보 C·D + 진단에서 찾은 보고 주기 편향)
    c_low / c_high : 5-step 속도 기준값 민감도 (0.2/1.5/4, 0.5/2.5/6)
    sp / sp_low    : 상태 속도를 시간 기준 10분 구간 sp_p50 로 교체. 보고 주기와 무관 (0.3/2/5, 0.2/1/3)
    d_port5        : 공용 항구 기준 10척 → 5척
    d_anchor       : 그 배가 5일 이상 밤(22~03시) 정박한 1km 칸도 정박지로 처리
  평가: fold_rankings 의 같은 fold, LGB_BASE (클래스 가중 없음) 2시드 평균, 피처 = 선박 피처 전체 + rv_·pd_ + 지역 사전확률
        2026 만 학습·검증. 채택 후보만 3모델·cross 로 다시 평가
  기존 파일은 건드리지 않음. 출력: features_v/<변형>/ship_features.parquet, results/step15_agg_<시각>.csv
실행: python step15_aggregate.py [--variants sp d_anchor] [--eval-only]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import BallTree

import cv_common as V
import fe_aggregate as A
import fe_config as C
import train_classifiers as TC
from step1_experiments import region_prior
from step4_final import attach

STAMP = time.strftime("%m%d_%H%M")
BASE_DIR = C.FEAT_DIR
VAR_ROOT = C.MODEL_ROOT / "features_v"
VARIANTS = {
    "c_low": dict(th=(0.2, 1.5, 4)),
    "c_high": dict(th=(0.5, 2.5, 6)),
    "sp": dict(speed="sp"),
    "sp_low": dict(speed="sp", th=(0.2, 1, 3)),
    "d_port5": dict(port_min=5),
    "d_anchor": dict(anchor=5),
    "t5": dict(hourly_dir=VAR_ROOT / "t5"),    # fe_hourly 를 SED_K_MINUTES=5 로 다시 만든 시간 기준 창 (5분 k-step, 3분 체류)
}
SEEDS = [0, 1]


def add_port_state(H, speed="ks", th=(0.3, 2, 5), port_min=10, anchor=0):
    """fe_aggregate.add_port_state 와 같은 흐름. speed·th·port_min·anchor 만 다름"""
    st = H["is_stationary"].to_numpy() == 1
    s = pd.DataFrame({"rid": H["rfid_id"].to_numpy()[st], "lon": H["lon_c"].to_numpy()[st], "lat": H["lat_c"].to_numpy()[st]})
    s["cell"] = A.grid(s["lon"], s["lat"], C.PORT_GRID_M)
    agg = s.groupby("cell").agg(n_vessels=("rid", "nunique"), lon=("lon", "median"), lat=("lat", "median"))
    ports = agg[agg["n_vessels"] >= port_min].reset_index()
    ports.to_csv(C.FEAT_DIR / "port_cells.csv", index=False)
    print(f"공용 항구 격자: {len(ports)}칸", flush=True)
    tree = BallTree(np.radians(ports[["lat", "lon"]].to_numpy()), metric="haversine")
    d, _ = tree.query(np.radians(H[["lat_c", "lon_c"]].to_numpy()), k=1)
    H["port_km"] = (d[:, 0] * A.R_KM).astype(np.float32)
    H["in_port"] = (st & (H["port_km"].to_numpy() < 0.5)).astype(np.float32)

    s["cell1k"] = A.grid(s["lon"], s["lat"], C.HOME_GRID_M)
    cnt = s.groupby(["rid", "cell1k"]).size().rename("n").reset_index()
    top = cnt.sort_values(["rid", "n"], ascending=[True, False]).drop_duplicates("rid")
    home = s.merge(top[["rid", "cell1k"]], on=["rid", "cell1k"]).groupby("rid")[["lon", "lat"]].median()
    home.columns = ["home_lon", "home_lat"]
    H = H.merge(home, left_on="rfid_id", right_index=True, how="left")
    first = H.groupby("rfid_id")[["lon_s", "lat_s"]].transform("first")
    H["home_lon"] = H["home_lon"].fillna(first["lon_s"])
    H["home_lat"] = H["home_lat"].fillna(first["lat_s"])
    H["home_km"] = A.hav_km(H["lon_c"], H["lat_c"], H["home_lon"], H["home_lat"]).astype(np.float32)

    port_stay = st & ((H["home_km"].to_numpy() < C.HOME_RADIUS_KM) | (H["in_port"].to_numpy() == 1))
    if anchor:
        hour = H["hour"].to_numpy().astype(int)
        night = st & np.isin(hour, [22, 23, 0, 1, 2, 3])
        c1k = A.grid(H["lon_c"].fillna(0), H["lat_c"].fillna(0), C.HOME_GRID_M)
        N = pd.DataFrame({"rid": H["rfid_id"].to_numpy()[night], "cell": c1k[night], "day": H["part_dt"].to_numpy()[night]})
        nights = N.groupby(["rid", "cell"])["day"].nunique()
        keys = nights[nights >= anchor].reset_index()
        print(f"정박지 칸 {len(keys):,} / 배 {keys['rid'].nunique():,}척", flush=True)
        key = pd.MultiIndex.from_frame(keys[["rid", "cell"]])
        in_anchor = pd.MultiIndex.from_arrays([H["rfid_id"].to_numpy(), c1k]).isin(key)
        port_stay = port_stay | (st & in_anchor)
    if speed == "sp":
        v = H["sp_p50"].fillna(H["ks_speed_med"]).fillna(0).to_numpy()
    else:
        v = H["ks_speed_med"].fillna(H["sp_p50"]).fillna(0).to_numpy()
    hr = H["ks_heading_R"].fillna(0).to_numpy()
    state = np.select([port_stay, st | (v < th[0]), v < th[1], v < th[2], hr >= 0.8],
                      [A.PORT, A.SEA_STOP, A.DRIFT, A.FISH, A.TRANSIT], A.FISH).astype(np.int8)
    H["state"] = state
    H["port_stay"] = port_stay.astype(np.float32)
    H["moving"] = ((H["frac_stop"].fillna(1) < 0.8) & (H["dist_m"] >= 200) & ~port_stay).astype(np.float32)
    print("상태 비율:", pd.Series(state).value_counts(normalize=True).sort_index().round(3).to_dict(), flush=True)
    return H


def build(name, cfg):
    out = VAR_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ship_features.parquet").exists():
        print(f"{name}: 있음, 건너뜀", flush=True)
        return
    t0 = time.time()
    cfg = dict(cfg)
    C.FEAT_DIR = cfg.pop("hourly_dir", BASE_DIR)
    H = A.load_hourly("hourly")
    C.FEAT_DIR = out                     # port_cells 를 변형 폴더에서 읽고 씀
    H = add_port_state(H, **cfg)
    D, cells1k = A.build_shipday(H)
    S, _ = A.build_ship(H, D, cells1k, None)
    S.to_parquet(out / "ship_features.parquet", index=False)
    C.FEAT_DIR = BASE_DIR
    print(f"{name}: 생성 {(time.time() - t0) / 60:.1f}분", flush=True)


def load_var(path):
    S = pd.read_parquet(path)
    S = S[S["fishing_code"].isin(C.CLASSES)].reset_index(drop=True)
    return S, S["fishing_code"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()


def evaluate(path, folds):
    S, y = load_var(path)
    pool = V.feature_pool(S, "B")
    attach(S, "")
    add = [c for c in S.columns if c.startswith(("rv_", "pd_"))]
    oof = np.zeros((len(y), len(C.CLASSES)))
    params = dict(TC.LGB_BASE, class_weight=None)
    for va in folds:
        tr = np.setdiff1d(np.arange(len(y)), va)
        Str, Sva = S.iloc[tr].reset_index(drop=True), S.iloc[va].reset_index(drop=True)
        Xa = pd.concat([Str[pool + add], region_prior(Str, y[tr], Str, self_in_ref=True)], axis=1)
        Xb = pd.concat([Sva[pool + add], region_prior(Str, y[tr], Sva)], axis=1)
        for sd in SEEDS:
            m = lgb.LGBMClassifier(**params, random_state=sd, n_jobs=C.N_JOBS).fit(Xa, y[tr])
            oof[va] += m.predict_proba(Xb) / len(SEEDS)
    return S["rfid_id"].astype(str).to_numpy(), y, oof


def main(a):
    t0 = time.time()
    names = a.variants or list(VARIANTS)
    if not a.eval_only:
        for n in names:
            build(n, VARIANTS[n])
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    ids0, y, P0 = evaluate(BASE_DIR / "ship_features.parquet", folds)
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    acc = lambda P: (P.argmax(1) == y).astype(float)
    a0 = acc(P0)
    rows = [dict(변형="base", cv=a0.mean(), macro=V.macro_f1(y, P0.argmax(1)), logloss=log_loss(y, P0))]
    print(f"base cv {a0.mean():.4f} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    rng = np.random.default_rng(0)
    for n in names:
        ids, yv, Pv = evaluate(VAR_ROOT / n / "ship_features.parquet", folds)
        assert (ids == ids0).all() and (yv == y).all()
        av = acc(Pv)
        d = av - a0
        boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        fd = [d[fold_id == f].mean() for f in range(len(folds))]
        rows.append(dict(변형=n, cv=av.mean(), 차이=d.mean(), 차이SE=boot.std(), 하한=np.quantile(boot, 0.025), 상한=np.quantile(boot, 0.975),
                         fold오름=int(sum(x > 0 for x in fd)), fold차=" ".join(f"{x:+.4f}" for x in fd), macro=V.macro_f1(y, Pv.argmax(1)), logloss=log_loss(y, Pv)))
        E = pd.DataFrame(rows)
        E.round(4).to_csv(C.RESULT_DIR / f"step15_agg_{STAMP}.csv", index=False, encoding="utf-8-sig")
        print(f"{n}: cv {av.mean():.4f} 차이 {d.mean():+.4f} SE {boot.std():.4f} fold {rows[-1]['fold차']} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--eval-only", action="store_true")
    main(ap.parse_args())
