"""
[NY-0] fe_aggregate.py 메모리 절약판 ─ 선박을 K개 묶음으로 나눠 집계 (RAM 8GB 환경용)
  fe_aggregate.main() 과 같은 결과를 내도록 '전체 데이터로 한 번만 정하는 것'은 따로 계산:
    · 공용 항구 격자(port_cells.csv) : 학습 전체 정박 시간대에서 1회 (원본 add_port_state 와 같은 규칙)
    · 출어일 유형 KMeans(daytype)     : 학습 전체 출어일에서 1회 (원본 build_ship 과 같은 규칙)
  선박-일·선박 피처는 배 하나 안에서만 계산되므로 묶음별 계산 결과를 이어 붙이면 원본과 같다.
실행: python ny_agg_sharded.py [--test] [--k 3]
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import fe_aggregate as A
import fe_config as C


def files(prefix):
    return sorted(C.FEAT_DIR.glob(f"{prefix}_[0-9][0-9].parquet"))


def load_shard(prefix, ids):
    H = pd.concat([pd.read_parquet(f, filters=[("rfid_id", "in", list(ids))]) for f in files(prefix)],
                  ignore_index=True)
    H = H[H["n_points"].notna()].reset_index(drop=True)
    H["fishing_code"] = H["fishing_code"].astype("object")
    fix = H["bearing_std"].isna() & (H["heading_R"] >= 0.999)
    H.loc[fix, "bearing_std"] = 0.0
    return H.sort_values(["rfid_id", "part_dt", "hour"], kind="stable").reset_index(drop=True)


def build_ports(prefix):
    cols = ["rfid_id", "lon_c", "lat_c", "is_stationary", "n_points"]
    H = pd.concat([pd.read_parquet(f, columns=cols) for f in files(prefix)], ignore_index=True)
    H = H[H["n_points"].notna() & (H["is_stationary"] == 1)]
    s = pd.DataFrame({"rid": H["rfid_id"].to_numpy(), "lon": H["lon_c"].to_numpy(), "lat": H["lat_c"].to_numpy()})
    s["cell"] = A.grid(s["lon"], s["lat"], C.PORT_GRID_M)
    agg = s.groupby("cell").agg(n_vessels=("rid", "nunique"), lon=("lon", "median"), lat=("lat", "median"))
    ports = agg[agg["n_vessels"] >= C.PORT_MIN_VESSELS].reset_index()
    ports.to_csv(C.FEAT_DIR / "port_cells.csv", index=False)
    print(f"공용 항구 격자 {len(ports)}칸 저장")
    return ports


def fit_daytype(D):
    Dm = D[D["n_moving_hours"] >= 1]
    X = Dm[A.DAY_TYPE_FEATS].astype(float)
    med = X.median()
    Xf = X.fillna(med)
    lo, hi = Xf.quantile(0.01), Xf.quantile(0.99)
    sc = StandardScaler().fit(Xf.clip(lo, hi, axis=1))
    km = KMeans(C.N_DAY_TYPES, random_state=C.RANDOM_STATE, n_init=4).fit(sc.transform(Xf.clip(lo, hi, axis=1)))
    return dict(med=med, lo=lo, hi=hi, scaler=sc, km=km)


def main(test, k):
    t0 = time.time()
    prefix, out = ("test_hourly", "test_") if test else ("hourly", "")
    ids = pd.concat([pd.read_parquet(f, columns=["rfid_id"]) for f in files(prefix)])["rfid_id"].unique()
    shards = np.array_split(np.sort(ids), k)
    ports = pd.read_csv(C.FEAT_DIR / "port_cells.csv") if test else build_ports(prefix)

    Ds, Cs = [], []
    for i, sid in enumerate(shards):
        H = A.add_port_state(load_shard(prefix, sid), ports)
        D, cells1k = A.build_shipday(H)
        Ds.append(D), Cs.append(cells1k)
        print(f"  [선박-일] 묶음 {i + 1}/{k}: 선박 {len(sid):,} / 시간 {len(H):,} → {len(D):,}일 "
              f"({(time.time() - t0) / 60:.1f}분)", flush=True)
        del H
    D = pd.concat(Ds, ignore_index=True)
    D.to_parquet(C.FEAT_DIR / f"{out}shipday.parquet", index=False)

    if test:
        daytype = joblib.load(C.FEAT_DIR / "daytype_model.joblib")
    else:
        daytype = fit_daytype(D)
        joblib.dump(daytype, C.FEAT_DIR / "daytype_model.joblib")

    Ss = []
    for i, sid in enumerate(shards):
        H = A.add_port_state(load_shard(prefix, sid), ports)
        S, _ = A.build_ship(H, Ds[i], Cs[i], daytype)
        Ss.append(S)
        print(f"  [선박] 묶음 {i + 1}/{k}: {len(S):,}척 ({(time.time() - t0) / 60:.1f}분)", flush=True)
        del H
    S = pd.concat(Ss, ignore_index=True)
    S.to_parquet(C.FEAT_DIR / f"{out}ship_features.parquet", index=False)
    print(f"✅ {out}ship_features {S.shape} / {out}shipday {D.shape}  {(time.time() - t0) / 60:.1f}분")
    print(S["fishing_code"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    main(a.test, a.k)
