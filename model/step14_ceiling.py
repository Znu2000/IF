"""
[STEP14-2] 라벨 상한 ─ 사고 배가 자료에 있고 그 시각 기록이 사고 위치 근처인 사고가 몇 건인지, 위치 오차를 가정하지 않고 추정
  사고마다 같은 업종 배 수 (사고 시각 ±90분, 반경 r 안)를 실제 시각과 ±7·14·21·28일 위약으로 셈
  사고 날은 원래 배가 많을 수 있음 → 10~20km 고리의 실제/위약 비로 위약을 보정
  초과 척수(r) = 실제(r) - 위약(r) * 고리 비. 반경을 늘려 포화되는 값이 연결 가능한 사고의 비율
  신뢰구간: 사고 단위 부트스트랩 2000회
  출력: results/step14_ceiling_<시각>.csv (rfid_id 없음)
실행: python step14_ceiling.py
"""
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C
import step9_labels as L9

STAMP = time.strftime("%m%d_%H%M")
SHIFTS = [-28, -21, -14, -7, 7, 14, 21, 28]
RADII = [0.5, 1, 2, 3, 5, 10]
RING = (10, 20)
N_BOOT = 2000


def main():
    t0 = time.time()
    acc = pd.read_excel(L9.ACC)
    acc["group"] = acc.apply(L9.group, axis=1)
    acc["gear"] = acc["어업면허"].map(lambda g: L9.GEAR.get(g, g))
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code"])
    gear_of = dict(zip(S["rfid_id"].astype(str), S["fishing_code"]))
    lim = lambda x: x.between("2026-01-01", "2026-06-30 23:59")
    shifted = pd.concat([acc.assign(발생일시=acc["발생일시"] + pd.Timedelta(days=s)) for s in [0] + SHIFTS])
    P = L9.load_points(shifted[lim(shifted["발생일시"])])
    P["gear"] = P["rid"].map(gear_of)
    P = P.sort_values("ts", kind="stable")
    pts, p_rid, p_gear = P["ts"].to_numpy(), P["rid"].to_numpy(), P["gear"].to_numpy()
    p_lon = P[[f"b_lon{j}" for j in range(6)]].to_numpy()
    p_lat = P[[f"b_lat{j}" for j in range(6)]].to_numpy()
    print(f"평활 점 {len(P):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    edges = RADII + [RING[0], RING[1]]
    rows = []
    for i, a in acc.iterrows():
        for s in [0] + SHIFTS:
            t = a["발생일시"] + pd.Timedelta(days=s)
            if not pd.Timestamp("2026-01-01") <= t <= pd.Timestamp("2026-06-30 23:59"):
                continue
            t_ = np.datetime64(t)
            f = np.datetime64(t.floor("h"))
            lo, hi = np.searchsorted(pts, f - np.timedelta64(2, "h")), np.searchsorted(pts, f + np.timedelta64(2, "h"), side="right")
            m = p_gear[lo:hi] == a["gear"]
            rec = dict(acc=i, s=s, group=a["group"])
            if m.any():
                tm = pts[lo:hi][m][:, None] + (np.arange(6) * 10 + 5).astype("timedelta64[m]")
                ok = np.abs((tm - t_) / np.timedelta64(1, "m")) <= L9.NEAR_MIN
                d = np.where(ok & ~np.isnan(p_lon[lo:hi][m]), A.hav_km(p_lon[lo:hi][m], p_lat[lo:hi][m], a["경도(변환)"], a["위도(변환)"]), np.inf)
                per = pd.Series(d.min(1), index=p_rid[lo:hi][m]).groupby(level=0).min().to_numpy()
            else:
                per = np.array([np.inf])
            for r in RADII:
                rec[f"r{r}"] = int((per <= r).sum())
            rec["ring"] = int(((per > RING[0]) & (per <= RING[1])).sum())
            rows.append(rec)
    D = pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    out = []
    for name, grp in [("바다", D[D["group"] != "port"]), ("항구", D[D["group"] == "port"])]:
        cols = [f"r{r}" for r in RADII] + ["ring"]
        real = grp[grp["s"] == 0].set_index("acc")[cols]
        plc = grp[grp["s"] != 0].groupby("acc")[cols].mean().reindex(real.index)
        R_, Q_ = real.to_numpy(float), plc.to_numpy(float)
        n = len(R_)
        boot = rng.integers(0, n, (N_BOOT, n))
        for j, r in enumerate(RADII):
            ratio = R_[:, -1].mean() / max(Q_[:, -1].mean(), 1e-9)
            ex = R_[:, j].mean() - Q_[:, j].mean() * ratio
            rb = R_[boot, -1].mean(1) / np.maximum(Q_[boot, -1].mean(1), 1e-9)
            eb = R_[boot, j].mean(1) - Q_[boot, j].mean(1) * rb
            out.append(dict(묶음=name, 사고수=n, 반경km=r, 실제=R_[:, j].mean(), 위약=Q_[:, j].mean(), 고리비=ratio, 초과척수=ex,
                            하한=np.quantile(eb, 0.025), 상한=np.quantile(eb, 0.975), 초과_보정없음=R_[:, j].mean() - Q_[:, j].mean()))
    Out = pd.DataFrame(out)
    Out["초과_사고건"] = Out["초과척수"] * Out["사고수"]
    Out.round(4).to_csv(C.RESULT_DIR / f"step14_ceiling_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(Out.round(3).to_string(index=False), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
