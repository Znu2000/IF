"""
[STEP 0] 캐시 생성 ─ 5.9GB 원본을 '딱 한 번만' 훑어서 EDA용 경량 파일을 만든다.

왜 필요한가?
  - 원본 6개 CSV ≈ 5.9GB, 약 2,200만 행. geom 문자열 파싱까지 전부 하면
    일반 노트북(RAM 16GB)에서는 메모리 부족 + 수 시간 소요.
  - 그래서 1회 순회하면서
      (a) 전체 데이터 기준 '가벼운 집계'(행 수, 결측, 클래스, 월별)  ← geom 파싱 안 함
      (b) 선박 5% 샘플에 대해서만 행 단위 피처 계산               ← geom 파싱
      (c) 선박 1% 샘플은 좌표(점) 단위로 저장                      ← 시각화/클러스터링
      (d) 사고 지점·시각 근처 항적 후보 행 저장                    ← 사고 매칭
    를 동시에 만든다. 이후 01~09 스크립트는 이 캐시만 읽으므로 빠르다.

실행:  python 00_build_cache.py                 (전체, 로컬 약 10~20분 예상)
       python 00_build_cache.py --max-chunks 1  (파일당 1청크만: 코드 동작 테스트용)
"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

import config as C
from geo_utils import (clean_chunk, hash_pct, features_from_df, explode_points,
                       first_point_vectorized, haversine_km, load_accidents)


def main(max_chunks=None):
    t_all = time.time()
    acc = load_accidents(C.accident_file())
    acc_ts = acc["acc_time"].dt.floor("h").to_numpy()
    acc_lon, acc_lat = acc["lon"].to_numpy(), acc["lat"].to_numpy()
    acc_dates = set()
    for d in acc["acc_time"].dt.normalize():
        for k in (-1, 0, 1):
            acc_dates.add(int((d + pd.Timedelta(days=k)).strftime("%Y%m%d")))

    stats_parts, vday_parts, feat_parts, pt_parts, cand_parts = [], [], [], [], []
    invalid_vals = Counter()
    q = Counter()

    for f in C.vpass_files():
        print(f"\n▶ {f.split('/')[-1].split(chr(92))[-1]}")
        reader = pd.read_csv(f, chunksize=C.CHUNK_SIZE, dtype=str, encoding="utf-8-sig",
                             usecols=["part_dt", "rfid_id", "rfid_time", "geom", "fishing_code"])
        for ci, raw in enumerate(reader):
            if max_chunks is not None and ci >= max_chunks:
                break
            t = time.time()
            # ── 품질 집계 (원본 기준)
            q["rows"] += len(raw)
            for col in raw.columns:
                q[f"na_{col}"] += int(raw[col].isna().sum())
            q["geom_not_multiline"] += int((~raw["geom"].fillna("").str.startswith("MULTILINESTRING")).sum())
            q["dup_key_in_chunk"] += int(raw.duplicated(["rfid_id", "part_dt", "rfid_time"]).sum())
            q["dup_full_in_chunk"] += int(raw.duplicated().sum())

            df = clean_chunk(raw)
            broken = df["part_dt"].isna() | df["rfid_id"].isna()
            q["broken_rows_dropped"] += int(broken.sum())
            df = df[~broken]
            q["invalid_hour"] += int((~df["hour_valid"]).sum())
            q["invalid_fishing_code"] += int(df["fishing_code"].isna().sum())
            invalid_vals.update(raw.loc[df.index[~df["hour_valid"]], "rfid_time"].fillna("<NA>").tolist())

            # ── (a) 전체 집계
            h = df["hour"].fillna(-1).astype(int)
            stats_parts.append(df.assign(hour_k=h).groupby(
                ["part_dt", "fishing_code", "hour_k"], dropna=False).size().rename("n").reset_index())
            vday_parts.append(df.groupby(["rfid_id", "part_dt", "fishing_code"], dropna=False)
                              .size().rename("n_rows").reset_index())

            # ── (b) 5% 선박 피처
            hp = hash_pct(df["rfid_id"])
            sub = df[hp < C.FEATURE_SAMPLE_PCT]
            if len(sub):
                feats = features_from_df(sub)
                keep = sub[["part_dt", "rfid_id", "rfid_time", "hour", "hour_valid", "fishing_code"]]
                feat_parts.append(pd.concat([keep, feats], axis=1))

            # ── (c) 1% 선박 좌표
            sub = df[hp < C.POINT_SAMPLE_PCT]
            if len(sub):
                pt_parts.append(explode_points(sub))

            # ── (d) 사고 후보
            dmask = df["part_dt"].isin(acc_dates).fillna(False).to_numpy()
            if dmask.any():
                s = df[dmask]
                fp = first_point_vectorized(s["geom"])
                ts = (pd.to_datetime(s["part_dt"].astype(str), format="%Y%m%d")
                      + pd.to_timedelta(s["hour"].fillna(12).astype(float), unit="h")).to_numpy()
                dt_h = np.abs((ts[:, None] - acc_ts[None, :]) / np.timedelta64(1, "h"))
                same_day = (s["part_dt"].to_numpy()[:, None] == acc["part_dt"].to_numpy()[None, :])
                tmask = (dt_h <= C.ACC_TIME_WINDOW_H) | (~s["hour_valid"].to_numpy()[:, None] & same_day)
                dist = haversine_km(fp["lon0"].to_numpy()[:, None], fp["lat0"].to_numpy()[:, None],
                                    acc_lon[None, :], acc_lat[None, :])
                hit = (tmask & (dist <= C.ACC_PREFILTER_KM)).any(axis=1)
                if hit.any():
                    cand_parts.append(s.loc[hit, ["part_dt", "rfid_id", "rfid_time", "hour",
                                                  "hour_valid", "fishing_code", "geom"]])
            print(f"  chunk {ci:02d}: {len(raw):,} rows  {time.time() - t:5.1f}s")

    print("\n▶ 저장 중...")
    stats = pd.concat(stats_parts).groupby(["part_dt", "fishing_code", "hour_k"], dropna=False)["n"].sum().reset_index()
    stats.to_parquet(C.CACHE_DIR / "stats_dt_hour.parquet", index=False)

    vday = pd.concat(vday_parts).groupby(["rfid_id", "part_dt", "fishing_code"], dropna=False)["n_rows"].sum().reset_index()
    vday.to_parquet(C.CACHE_DIR / "vessel_day.parquet", index=False)

    pd.concat(feat_parts).reset_index(drop=True).to_parquet(C.CACHE_DIR / "sample_feats.parquet", index=False)
    pd.concat(pt_parts).reset_index(drop=True).to_parquet(C.CACHE_DIR / "sample_points.parquet", index=False)
    if cand_parts:
        pd.concat(cand_parts).reset_index(drop=True).to_parquet(C.CACHE_DIR / "acc_candidates.parquet", index=False)

    pd.Series(invalid_vals).rename("n").rename_axis("value").reset_index().sort_values(
        "n", ascending=False).to_parquet(C.CACHE_DIR / "invalid_rfid_time.parquet", index=False)
    (C.CACHE_DIR / "quality.json").write_text(json.dumps(dict(q), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(dict(q), ensure_ascii=False, indent=2))
    print(f"\n✅ 완료: {(time.time() - t_all) / 60:.1f}분  →  {C.CACHE_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=None, help="파일당 처리할 최대 청크 수(테스트용)")
    main(ap.parse_args().max_chunks)
