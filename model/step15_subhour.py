"""
[STEP15-4] 시간 내부 행동 피처 (전처리 조건변경 판단.pdf 후보 B) ─ 1시간 1상태로 뭉개는 정보를 10분 구간 속도로 되살림
  입력: hourly_MM.parquet 의 10분 평활 위치 (fe_hourly 재실행 없음), anom_hourly 의 바다 여부
  10분 속도 = 인접 구간 거리 * 6 (시간 안 5개 + 다음 시간이 연속이면 경계 1개)
  선박 피처 (바다 시간만, sb_ 접두사)
    share  : 속도 7구간 비율 (0~0.3~1~2~3~5~8~)
    tr     : 4상태(정지 <0.5, 저속 <3, 중속 <6, 고속) 10분 전이 확률 16개
    run    : 저속 연속 길이(분) 중앙값·90%, 시간당 블록 수, 고속 연속 길이 중앙값
    mix    : 한 시간 안에 저속과 고속이 같이 있는 시간 비율, 정지와 이동이 같이 있는 비율, 시간당 상태 전환 수
    turn   : 움직이는 구간 방향 변화 중앙값, 90도 넘는 비율
  평가: step15_aggregate.evaluate 와 같은 fold·모델로 base + sb_ 와 base 비교
  출력: features_v/subhour/ship_features.parquet (base + sb_), results/step15_sub_<시각>.csv
실행: python step15_subhour.py
"""
import json
import time

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C
import step15_aggregate as G

STAMP = time.strftime("%m%d_%H%M")
BANDS = [0.3, 1, 2, 3, 5, 8]
ST4 = [0.5, 3, 6]
OUT = G.VAR_ROOT / "subhour"


def run_lengths(mask, brk):
    """mask: 구간별 참/거짓, brk: 새 시퀀스 시작. 참 연속 길이 (시작 위치, 길이)"""
    start = mask & (brk | ~np.r_[False, mask[:-1]])
    idx = np.cumsum(start)
    n = np.bincount(idx[mask], minlength=idx.max() + 1)[1:]
    return np.flatnonzero(start), n


def build():
    t0 = time.time()
    cols = ["rfid_id", "part_dt", "hour"] + [f"b_lon{i}" for i in range(6)] + [f"b_lat{i}" for i in range(6)]
    P = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(C.FEAT_DIR.glob("hourly_*.parquet"))], ignore_index=True)
    H = pd.read_parquet(C.FEAT_DIR / "anom_hourly.parquet", columns=["rfid_id", "part_dt", "hour", "at_sea"])
    P = P.merge(H, on=["rfid_id", "part_dt", "hour"], how="inner")
    P = P[P["at_sea"] == 1].sort_values(["rfid_id", "part_dt", "hour"], kind="stable").reset_index(drop=True)
    lon, lat = P[[f"b_lon{i}" for i in range(6)]].to_numpy(float), P[[f"b_lat{i}" for i in range(6)]].to_numpy(float)
    rid = P["rfid_id"].to_numpy()
    hour_abs = pd.to_datetime(P["part_dt"].astype(str)).astype("int64").to_numpy() // 3_600_000_000_000 + P["hour"].to_numpy().astype(int)
    next_ok = np.r_[(rid[1:] == rid[:-1]) & (hour_abs[1:] == hour_abs[:-1] + 1), False]
    # 시간 안 5개 + 경계 1개 = 시간당 6칸
    lon2 = np.column_stack([lon, np.r_[lon[1:, 0], [np.nan]]])
    lat2 = np.column_stack([lat, np.r_[lat[1:, 0], [np.nan]]])
    lon2[~next_ok, 6], lat2[~next_ok, 6] = np.nan, np.nan
    d = A.hav_km(lon2[:, :-1], lat2[:, :-1], lon2[:, 1:], lat2[:, 1:])
    sp = (d * 1000 / 1852 * 6).astype(np.float32)
    sp[sp > 40] = np.nan
    dx, dy = (lon2[:, 1:] - lon2[:, :-1]) * np.cos(np.radians(lat2[:, :-1])), lat2[:, 1:] - lat2[:, :-1]
    head = np.arctan2(dy, dx)
    turn = np.abs((head[:, 1:] - head[:, :-1] + np.pi) % (2 * np.pi) - np.pi)
    turn = np.degrees(np.where((sp[:, 1:] >= 0.5) & (sp[:, :-1] >= 0.5), turn, np.nan)).astype(np.float32)
    print(f"바다 {len(P):,}시간 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    R = np.repeat(rid, 6)
    v = sp.ravel()
    ok = ~np.isnan(v)
    brk = np.r_[True, R[1:] != R[:-1]] | np.r_[True, ~ok[:-1]]
    F = pd.DataFrame(index=pd.Index(np.unique(rid), name="rfid_id"))
    total = pd.Series(ok.astype(float)).groupby(R).sum().reindex(F.index)
    band = np.digitize(v, BANDS)
    for b in range(len(BANDS) + 1):
        F[f"sb_share_{b}"] = pd.Series((band == b) & ok).groupby(R).sum().reindex(F.index) / total
    s4 = np.digitize(v, ST4)
    a, b_ = s4[:-1], s4[1:]
    pair = ok[:-1] & ok[1:] & (R[1:] == R[:-1])
    T = pd.DataFrame({"r": R[:-1][pair], "a": a[pair], "b": b_[pair]}).groupby(["r", "a", "b"]).size().unstack(["a", "b"], fill_value=0)
    T = T.reindex(columns=pd.MultiIndex.from_product([range(4), range(4)], names=["a", "b"]), fill_value=0)
    for x in range(4):
        tot = T.loc[:, x].sum(1)
        for y in range(4):
            F[f"sb_tr_{x}{y}"] = (T[(x, y)] / tot).where(tot > 0).reindex(F.index)
    F["sb_switch_per_h"] = (T.sum(1) - sum(T[(x, x)] for x in range(4))).reindex(F.index) / total * 6

    hours = pd.Series(1.0, index=R[::6]).groupby(level=0).sum().reindex(F.index)
    for name, m in [("slow", (s4 == 1) & ok), ("fast", (s4 == 3) & ok)]:
        st, n = run_lengths(m, brk)
        L = pd.DataFrame({"r": R[st], "n": n * 10.0})
        g = L.groupby("r")["n"]
        F[f"sb_{name}_run_q50"], F[f"sb_{name}_run_q90"] = g.median().reindex(F.index), g.quantile(0.9).reindex(F.index)
        F[f"sb_{name}_blocks_per_h"] = g.size().reindex(F.index) / hours
    has = lambda c: (s4.reshape(-1, 6) == c) & ok.reshape(-1, 6)
    slow, fast, mid, stop = has(1).any(1), has(3).any(1), has(2).any(1), has(0).any(1)
    F["sb_mix_slow_fast"] = pd.Series(slow & fast).groupby(rid).mean().reindex(F.index)
    F["sb_mix_slow_mid"] = pd.Series(slow & mid).groupby(rid).mean().reindex(F.index)
    F["sb_mix_stop_move"] = pd.Series(stop & (slow | mid | fast)).groupby(rid).mean().reindex(F.index)
    Tn = pd.DataFrame({"r": np.repeat(rid, 5), "t": turn.ravel()}).dropna()
    g = Tn.groupby("r")["t"]
    F["sb_turn_q50"] = g.median().reindex(F.index)
    F["sb_turn90"] = (Tn["t"] > 90).groupby(Tn["r"]).mean().reindex(F.index)
    print(f"피처 {F.shape[1]}개 ({(time.time() - t0) / 60:.1f}분)", flush=True)

    S = pd.read_parquet(C.FEAT_DIR / "ship_features.parquet")
    S = S.merge(F.reset_index(), on="rfid_id", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_parquet(OUT / "ship_features.parquet", index=False)
    return list(F.columns)


def main():
    t0 = time.time()
    names = build()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    folds = [np.array(v) for v in R["fold_val_idx"]]
    ids0, y, P0 = G.evaluate(C.FEAT_DIR / "ship_features.parquet", folds)
    ids, yv, P1 = G.evaluate(OUT / "ship_features.parquet", folds)
    assert (ids == ids0).all() and (yv == y).all()
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(folds):
        fold_id[v] = f
    a0, a1 = (P0.argmax(1) == y).astype(float), (P1.argmax(1) == y).astype(float)
    d = a1 - a0
    boot = d[np.random.default_rng(0).integers(0, len(d), (5000, len(d)))].mean(1)
    fd = " ".join(f"{d[fold_id == f].mean():+.4f}" for f in range(len(folds)))
    from sklearn.metrics import log_loss
    import cv_common as V
    E = pd.DataFrame([dict(변형="base", cv=a0.mean(), macro=V.macro_f1(y, P0.argmax(1)), logloss=log_loss(y, P0)),
                      dict(변형="base+sb", cv=a1.mean(), 차이=d.mean(), 차이SE=boot.std(), fold차=fd, macro=V.macro_f1(y, P1.argmax(1)), logloss=log_loss(y, P1))])
    E.round(4).to_csv(C.RESULT_DIR / f"step15_sub_{STAMP}.csv", index=False, encoding="utf-8-sig")
    print(E.round(4).to_string(index=False), flush=True)
    # 클래스 분리력 참고
    S = pd.read_parquet(OUT / "ship_features.parquet")
    S = S[S["fishing_code"].isin(C.CLASSES)]
    print(S.groupby("fishing_code")[["sb_share_1", "sb_slow_run_q50", "sb_mix_slow_fast", "sb_switch_per_h"]].median().round(3).to_string(), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
