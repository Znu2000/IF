"""
[ANOM-6] 사고 라벨 재매칭 ─ 시간 집계(시작·끝·중심점 3개) 대신 원본 항적의 모든 점으로 거리 계산
  v1(anomaly_labels.py) 한계: 1시간 항적이 길면 사고 위치를 지나가도 시작·끝·중심점이 멀어서 놓치거나 엉뚱한 배와 매칭
  방법
    · 사고 시각이 걸친 시간대(±1h)의 원본 CSV 행만 골라냄 (월별 파일 병렬 스캔)
    · 점별 시각이 없으므로 그 시간 안에서 균등 간격으로 가정 → 점마다 대략의 분 단위 시각
    · 같은 업종 선박마다: 사고 시각 ±60분 안의 점들 중 사고 위치와 최소 거리 + 그때 시각
  신뢰 등급 (v1 과 같은 규칙, 거리 기준만 점 단위라 더 엄격)
    A : 가장 가까운 배가 1.0km 안이고, 두 번째 배보다 2배 이상 가깝거나 두 번째가 3km 밖
    B : 가장 가까운 배가 3km 안 (애매)
  라벨 시각 = 최근접 점의 시각(정시 내림) ─ 사고 신고 시각보다 실제 항적이 사고 위치에 있던 시각이 정확
  출력 : results/anom_labels_raw.csv  (실제 rfid_id 포함 → 깃헙 업로드 금지), v1 과 일치율 출력
실행: python anomaly_labels_raw.py   (SED_DATA_ROOT 로 원본 2026 CSV 위치 지정, 6개 파일 병렬)
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import fe_aggregate as A
import fe_config as C
from fe_hourly import parse_parts

ACC = C.PROJECT_ROOT / "문제 데이터_수정_1027" / "26년 1~6월 어선 사고현황(테스트 데이터).xlsx"
TYPE = {"연승": "채낚기연승낚시"}
BEFORE_H, AFTER_H, NEAR_MIN = 1, 1, 60


def load_acc():
    acc = pd.read_excel(ACC)
    acc["gear"] = acc["어업면허"].map(lambda g: TYPE.get(g, g))
    return acc


def wanted_keys(acc):
    """사고마다 [t0-1h, t0+1h] 시간대 → 'YYYYMMDD|H' 키 집합"""
    keys = set()
    for t in acc["발생일시"]:
        t0 = t.floor("h")
        for h in range(-BEFORE_H, AFTER_H + 1):
            tt = t0 + pd.Timedelta(hours=h)
            keys.add(f"{tt:%Y%m%d}|{tt.hour}")
    return keys


def scan(path, keys):
    t0 = time.time()
    out = []
    for raw in pd.read_csv(path, chunksize=C.CHUNK_SIZE, dtype=str, encoding="utf-8-sig",
                           usecols=["part_dt", "rfid_id", "rfid_time", "geom", "fishing_code"]):
        k = raw["part_dt"].str.strip() + "|" + pd.to_numeric(raw["rfid_time"], errors="coerce").astype("Int64").astype(str)
        out.append(raw[k.isin(keys)])
    df = pd.concat(out, ignore_index=True)
    print(f"  {path.name}: {len(df):,}행 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    return df


def points(df):
    """행마다 점 배열과 점별 대략 시각 (그 시간 안 균등 간격) → {(업종, 시간대 시작): [...]}"""
    rows = {}
    for r in df.itertuples(index=False):
        parts = parse_parts(r.geom)
        if not parts:
            continue
        p = np.vstack(parts)
        base = pd.Timestamp(r.part_dt) + pd.Timedelta(hours=int(r.rfid_time))
        frac = (np.arange(len(p)) + 0.5) / len(p)
        rows.setdefault((r.fishing_code, base), []).append(
            (int(r.rfid_id), p, base + pd.to_timedelta(frac * 3600, unit="s")))
    return rows


def match(acc, R):
    out = []
    for i, a in acc.iterrows():
        t = a["발생일시"]
        base = dict(acc_idx=i, acc_time=t, acc_type=a["발생유형"], gear=a["gear"], area=a["발생해역"],
                    lat=a["위도(변환)"], lon=a["경도(변환)"])
        best = {}
        t0 = t.floor("h")
        cand = [r for h in range(-BEFORE_H, AFTER_H + 1) for r in R.get((a["gear"], t0 + pd.Timedelta(hours=h)), [])]
        for rid, p, ts in cand:
            near = np.abs((ts - t).total_seconds().to_numpy()) <= NEAR_MIN * 60
            if not near.any():
                continue
            d = A.hav_km(p[near, 0], p[near, 1], a["경도(변환)"], a["위도(변환)"])
            j = int(np.argmin(d))
            if rid not in best or d[j] < best[rid][0]:
                best[rid] = (float(d[j]), ts[near][j])
        if not best:
            out.append(base | dict(tier="none"))
            continue
        per = sorted(best.items(), key=lambda kv: kv[1][0])
        (rid, (d1, tc)), d2 = per[0], (per[1][1][0] if len(per) > 1 else np.inf)
        tier = "A" if d1 <= 1.0 and (d2 >= 2 * d1 or d2 > 3) else ("B" if d1 <= 3 else "none")
        out.append(base | dict(tier=tier, rfid_id=rid, d1_km=round(d1, 3), d2_km=round(d2, 3), n_cand=len(per),
                               t_closest=tc, label_ts=tc.floor("h")))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    acc = load_acc()
    keys = wanted_keys(acc)
    files = C.vpass_files()
    print(f"사고 {len(acc)}건, 시간대 키 {len(keys)}개, 파일 {len(files)}개")
    with ProcessPoolExecutor(min(len(files), C.N_JOBS)) as ex:
        df = pd.concat(list(ex.map(scan, files, [keys] * len(files))), ignore_index=True)
    df = df[df["fishing_code"].isin(C.CLASSES)]
    R = points(df)
    print(f"후보 항적 {sum(map(len, R.values())):,}개 ({(time.time() - t0) / 60:.1f}분)")
    L = match(acc, R)
    L["rfid_id"] = L["rfid_id"].astype("Int64")
    L.to_csv(C.RESULT_DIR / "anom_labels_raw.csv", index=False, encoding="utf-8-sig")
    print(L["tier"].value_counts().to_string())

    p1 = C.RESULT_DIR / "anom_labels.csv"
    if p1.exists():  # v1 과 비교: 같은 사고에 같은 배가 잡혔는지
        L1 = pd.read_csv(p1, parse_dates=["label_ts"]).set_index("acc_idx")
        J = L.set_index("acc_idx").join(L1[["tier", "rfid_id", "label_ts"]], rsuffix="_v1")
        both = J["rfid_id"].notna() & J["rfid_id_v1"].notna()
        print(pd.crosstab(J["tier_v1"], J["tier"], rownames=["v1"], colnames=["raw"]).to_string())
        print(f"같은 배 매칭 {(J.loc[both, 'rfid_id'] == J.loc[both, 'rfid_id_v1']).mean():.2f} "
              f"({both.sum()}건 중), 라벨 시각 차이 중앙값 "
              f"{(J.loc[both, 'label_ts'] - J.loc[both, 'label_ts_v1']).abs().median()}")
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분 → anom_labels_raw.csv")


if __name__ == "__main__":
    main()
