"""
[STEP9-8] 후보 묶음 급정지 신호 재검증 ─ step9_priors 의 "후보 배의 사고 ±6h 급정지 61% vs 위약 39%" 가 설정을 바꿔도 남는가
  바꾸는 것: 반경 2·3·5km / 위약 날짜 (±7·14일, ±3·7·14·21·28일) / 창 (−6~+6h, 사고 전 −6~0h, 사고 후 0~+6h)
  음성 대조: 항구 사고 (원래 서 있는 배라 바다 급정지 신호가 없어야 함)
  유의성: 사고별 (실제 − 위약) 차이의 부호 뒤집기 검정 + 조합 수가 많으므로 해석은 전체 경향으로
  출력: results/step9_bag_robust.csv (rfid_id 없음)
실행: python step9_bag_robust.py
"""
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C
import step9_detect as D
import step9_labels as L9

SHIFT_SETS = {"±7·14일": [-14, -7, 7, 14], "±3~28일": [-28, -21, -14, -7, -3, 3, 7, 14, 21, 28]}
ALL_SHIFTS = sorted({0} | {s for v in SHIFT_SETS.values() for s in v})
RADII = [2.0, 3.0, 5.0]
WINDOWS = {"−6~+6h": (-6, 6), "사고 전 −6~0h": (-6, 0), "사고 후 0~+6h": (0, 6)}
N_PERM = 10000


def main():
    t0 = time.time()
    acc = pd.read_excel(L9.ACC)
    acc["group"] = acc.apply(L9.group, axis=1)
    acc["gear"] = acc["어업면허"].map(lambda g: L9.GEAR.get(g, g))
    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet", columns=["rfid_id", "fishing_code"])
    gear_of = dict(zip(S["rfid_id"].astype(str), S["fishing_code"]))
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=D.COLS)
    H["rid"] = H["rfid_id"].astype(str)
    H["ts"] = pd.to_datetime(H["ts"])
    H = H.sort_values(["rid", "ts"], kind="stable").reset_index(drop=True)
    F = D.build(H)
    onset_ts = {r: g.to_numpy() for r, g in F.loc[F["onset"], ["rid", "ts"]].groupby("rid")["ts"]}

    shifted = pd.concat([acc.assign(발생일시=acc["발생일시"] + pd.Timedelta(days=s)) for s in ALL_SHIFTS])
    shifted = shifted[shifted["발생일시"].between("2026-01-01", "2026-06-30 23:59")]
    P = L9.load_points(shifted)
    P["gear"] = P["rid"].map(gear_of)
    print(f"준비 끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    rows = []
    for s in ALL_SHIFTS:
        for i, a in acc.iterrows():
            t = a["발생일시"] + pd.Timedelta(days=s)
            if not pd.Timestamp("2026-01-01") <= t <= pd.Timestamp("2026-06-30 23:59"):
                continue
            W = P[(P["gear"] == a["gear"]) & (P["ts"] >= t.floor("h") - pd.Timedelta(hours=2)) & (P["ts"] <= t.floor("h") + pd.Timedelta(hours=2))]
            if W.empty:
                continue
            lon = W[[f"b_lon{j}" for j in range(6)]].to_numpy()
            lat = W[[f"b_lat{j}" for j in range(6)]].to_numpy()
            tm = W["ts"].to_numpy()[:, None] + (np.arange(6) * 10 + 5).astype("timedelta64[m]")
            ok = np.abs((tm - np.datetime64(t)) / np.timedelta64(1, "m")) <= L9.NEAR_MIN
            d = np.where(ok & ~np.isnan(lon), A.hav_km(lon, lat, a["경도(변환)"], a["위도(변환)"]), np.inf)
            per = pd.Series(d.min(1), index=W["rid"].to_numpy()).groupby(level=0).min()
            per = per[per <= max(RADII)]
            tn = np.datetime64(t.floor("h"))
            for r, dist in per.items():
                rel = (onset_ts[r] - tn) / np.timedelta64(1, "h") if r in onset_ts else np.array([])
                rows.append(dict(acc_idx=i, shift=s, group=a["group"], dist=dist,
                                 **{w: bool(((rel >= lo) & (rel <= hi)).any()) for w, (lo, hi) in WINDOWS.items()}))
    C_ = pd.DataFrame(rows)
    print(f"후보 {len(C_):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    rng = np.random.default_rng(0)
    out = []
    for grp, gm in [("바다 전체", C_["group"] != "port"), ("바다 기동 불능", C_["group"] == "disabled"), ("항구(음성 대조)", C_["group"] == "port")]:
        for rad in RADII:
            for sname, shifts in SHIFT_SETS.items():
                g = C_[gm & (C_["dist"] <= rad)]
                per_acc = g.groupby(["acc_idx", "shift"])[list(WINDOWS)].mean()
                real = per_acc.xs(0, level="shift")
                plc = per_acc[per_acc.index.get_level_values("shift").isin(shifts)].groupby("acc_idx").mean()
                idx = real.index.intersection(plc.index)
                for w in WINDOWS:
                    d = (real.loc[idx, w] - plc.loc[idx, w]).to_numpy()
                    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
                    p = ((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (N_PERM + 1)
                    out.append(dict(사고묶음=grp, 반경km=rad, 위약=sname, 창=w, 사고수=len(idx), 실제=real.loc[idx, w].mean(),
                                    위약비율=plc.loc[idx, w].mean(), 차이=d.mean(), p=p))
    R = pd.DataFrame(out)
    R.round(4).to_csv(C.RESULT_DIR / "step9_bag_robust.csv", index=False, encoding="utf-8-sig")
    print(R.round(3).to_string(index=False))
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
