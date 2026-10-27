"""
항적(geom) 파싱 + 피처 계산 유틸

데이터 구조 메모
- 1행 = (선박 rfid_id, 날짜 part_dt, 시간대 rfid_time) 1시간 동안의 항적
- geom = WKT MULTILINESTRING ((lon lat, lon lat, ...), (lon lat, ...))
  · 괄호 하나 = 한 구간(part). 신호가 끊겼다 다시 잡히면 part 가 나뉘는 것으로 추정
  · 점마다 타임스탬프가 없음 → "1시간 동안 균등 간격으로 찍혔다" 가정 하에 속도 근사
- rfid_time 약 2.6%에 숫자가 아닌 문자열(연안복합 등)이 들어감 → hour = NaN 으로 처리
"""
import re
import numpy as np
import pandas as pd

from config import CLASSES

EARTH_R_KM = 6371.0088
_FIRST_PT = re.compile(r"\(\(\s*([-\d.]+)\s+([-\d.]+)")


# ─────────────────────────────────────────────
# 1. 파싱
# ─────────────────────────────────────────────
def parse_multilinestring(wkt):
    """WKT MULTILINESTRING → [ndarray(n_i, 2) (lon, lat), ...]  (실패 시 [])"""
    if not isinstance(wkt, str) or "(" not in wkt:
        return []
    body = wkt[wkt.find("(") :].strip()
    body = body.strip("()")  # 바깥 괄호 제거
    parts = []
    for p in body.split("),"):
        p = p.replace("(", " ").replace(")", " ").replace(",", " ")
        arr = np.array(p.split(), dtype=np.float64)
        if arr.size >= 2 and arr.size % 2 == 0:
            parts.append(arr.reshape(-1, 2))
    return parts


def first_point_vectorized(geom: pd.Series) -> pd.DataFrame:
    """정규식으로 첫 점(lon, lat)만 빠르게 추출 (전체 파싱 없이 사전필터용)"""
    ext = geom.str.extract(_FIRST_PT)
    return pd.DataFrame(
        {"lon0": pd.to_numeric(ext[0], errors="coerce"), "lat0": pd.to_numeric(ext[1], errors="coerce")},
        index=geom.index,
    )


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """원본 청크 정리: 타입 변환 + rfid_time 오염 플래그"""
    df = df.copy()
    # 6월 CSV 등에서 geom 이 줄바꿈으로 쪼개진 '깨진 행'이 존재 → part_dt 가 8자리 숫자가 아니면 NaN
    pdt = df["part_dt"].where(df["part_dt"].str.fullmatch(r"\d{8}", na=False))
    rid = df["rfid_id"].where(df["rfid_id"].str.fullmatch(r"\d+", na=False))
    df["part_dt"] = pd.to_numeric(pdt).astype("Int64")
    df["rfid_id"] = pd.to_numeric(rid).astype("Int64")
    hour = pd.to_numeric(df["rfid_time"], errors="coerce")
    df["hour_valid"] = hour.between(0, 23)
    df["hour"] = hour.where(df["hour_valid"]).astype("Int64")
    df["fishing_code"] = df["fishing_code"].where(df["fishing_code"].isin(CLASSES))
    return df


def hash_pct(ids: pd.Series) -> np.ndarray:
    """rfid_id → 0~99 (결정적 해시). 청크가 달라도 같은 선박은 같은 값 → 선박 단위 샘플링"""
    x = ids.fillna(0).astype("int64").to_numpy().astype(np.uint64)
    x = (x * np.uint64(2654435761)) % np.uint64(2**32)
    return (x % np.uint64(100)).astype(np.int64)


# ─────────────────────────────────────────────
# 2. 거리/방위 기본 함수
# ─────────────────────────────────────────────
def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def to_local_km(lon, lat, lon0=None, lat0=None):
    """작은 영역용 등거리 근사 투영 (km)"""
    lon0 = np.mean(lon) if lon0 is None else lon0
    lat0 = np.mean(lat) if lat0 is None else lat0
    x = (lon - lon0) * 111.320 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.574
    return x, y


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ─────────────────────────────────────────────
# 3. 행(1시간 항적) 단위 피처
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # 기본
    "n_parts", "n_points",
    # 공간
    "c_lon", "c_lat", "s_lon", "s_lat", "e_lon", "e_lat",
    "bbox_w_km", "bbox_h_km", "bbox_area_km2", "rog_km",
    # 이동/속도
    "path_km", "disp_km", "straightness", "speed_kn",
    "step_mean_m", "step_std_m", "step_max_m", "stop_ratio", "gap_km",
    # 방향
    "turn_abs_mean", "turn_std", "sharp_turn_ratio", "heading_R",
    # 주파수
    "spec_entropy", "lowfreq_ratio",
]


def _spectral(sig):
    """1D 신호의 스펙트럼 엔트로피(0~1)와 저주파 에너지 비율"""
    if sig.size < 8 or np.allclose(sig, sig[0]):
        return np.nan, np.nan
    s = sig - sig.mean()
    p = np.abs(np.fft.rfft(s)) ** 2
    p = p[1:]
    tot = p.sum()
    if tot <= 0:
        return np.nan, np.nan
    q = p / tot
    ent = -(q * np.log(q + 1e-12)).sum() / np.log(len(q)) if len(q) > 1 else np.nan
    k = max(1, len(q) // 4)
    return float(ent), float(q[:k].sum())


def traj_features(parts, stop_thr_m=5.0):
    """parts(list of (n,2)) → dict. 1시간 항적 1개에 대한 피처"""
    out = dict.fromkeys(FEATURE_COLS, np.nan)
    if not parts:
        return out
    xy = np.vstack(parts)
    lon, lat = xy[:, 0], xy[:, 1]
    n = len(xy)
    out.update(n_parts=len(parts), n_points=n,
               c_lon=lon.mean(), c_lat=lat.mean(),
               s_lon=lon[0], s_lat=lat[0], e_lon=lon[-1], e_lat=lat[-1])

    x, y = to_local_km(lon, lat)
    out["bbox_w_km"] = x.max() - x.min()
    out["bbox_h_km"] = y.max() - y.min()
    out["bbox_area_km2"] = out["bbox_w_km"] * out["bbox_h_km"]
    out["rog_km"] = float(np.sqrt(np.mean(x**2 + y**2)))  # 회전반경: 얼마나 넓게 퍼져 움직였나

    # part 내부 스텝만 사용 (part 사이 점프는 신호 끊김 → gap 으로 따로)
    steps, heads, turns, offs = [], [], [], 0
    gap = 0.0
    for i, p in enumerate(parts):
        m = len(p)
        px, py = x[offs : offs + m], y[offs : offs + m]
        if i > 0:
            gap += np.hypot(px[0] - x[offs - 1], py[0] - y[offs - 1])
        if m >= 2:
            dx, dy = np.diff(px), np.diff(py)
            d = np.hypot(dx, dy)
            steps.append(d)
            mv = d > stop_thr_m / 1000  # 정지 스텝은 방향 계산 제외
            h = np.arctan2(dy[mv], dx[mv])
            heads.append(h)
            if h.size >= 2:
                turns.append(_wrap(np.diff(h)))
        offs += m
    out["gap_km"] = gap

    if steps:
        st = np.concatenate(steps)
        path = st.sum()
        out["path_km"] = path
        out["disp_km"] = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        out["straightness"] = out["disp_km"] / path if path > 1e-6 else np.nan
        out["speed_kn"] = path / 1.852  # 1행=1시간 가정 → km/h → knot
        out["step_mean_m"] = st.mean() * 1000
        out["step_std_m"] = st.std() * 1000
        out["step_max_m"] = st.max() * 1000
        out["stop_ratio"] = float((st * 1000 < stop_thr_m).mean())
        out["spec_entropy"], out["lowfreq_ratio"] = _spectral(st)

    if heads and np.concatenate(heads).size:
        h = np.concatenate(heads)
        out["heading_R"] = float(np.hypot(np.cos(h).mean(), np.sin(h).mean()))  # 1=일직선, 0=사방
    if turns:
        t = np.abs(np.concatenate(turns))
        if t.size:
            out["turn_abs_mean"] = float(np.degrees(t.mean()))
            out["turn_std"] = float(np.degrees(t.std()))
            out["sharp_turn_ratio"] = float((t > np.pi / 2).mean())
    return out


def features_from_df(df: pd.DataFrame, geom_col="geom") -> pd.DataFrame:
    feats = [traj_features(parse_multilinestring(g)) for g in df[geom_col].to_numpy()]
    return pd.DataFrame(feats, index=df.index)


def explode_points(df: pd.DataFrame, geom_col="geom", keep=("rfid_id", "part_dt", "hour", "fishing_code")):
    """행 → 점 단위 DataFrame (seq: 행 내 순서, part: 구간 번호)"""
    xy_list, part_list, counts = [], [], []
    for g in df[geom_col].to_numpy():
        parts = parse_multilinestring(g)
        if parts:
            xy_list.append(np.vstack(parts))
            part_list.append(np.repeat(np.arange(len(parts)), [len(p) for p in parts]))
            counts.append(sum(len(p) for p in parts))
        else:
            counts.append(0)
    counts = np.asarray(counts)
    if counts.sum() == 0:
        return pd.DataFrame(columns=list(keep) + ["part", "seq", "lon", "lat"])
    xy = np.vstack(xy_list)
    pts = pd.DataFrame({k: np.repeat(df[k].to_numpy(), counts) for k in keep})
    pts["part"] = np.concatenate(part_list).astype("int16")
    pts["seq"] = np.concatenate([np.arange(c) for c in counts if c > 0]).astype("int16")
    pts["lon"] = xy[:, 0].astype("float32")
    pts["lat"] = xy[:, 1].astype("float32")
    return pts


# ─────────────────────────────────────────────
# 4. 일(vessel-day) 단위 피처  ─ 출항~귀항 1일 궤적 요약
# ─────────────────────────────────────────────
def daily_features(feats: pd.DataFrame) -> pd.DataFrame:
    """행 단위 피처(feats: rfid_id, part_dt, hour, fishing_code + FEATURE_COLS) → 선박-일 단위"""
    f = feats.sort_values(["rfid_id", "part_dt", "hour"])
    g = f.groupby(["rfid_id", "part_dt"], sort=False)
    d = g.agg(
        fishing_code=("fishing_code", "first"),
        n_hours=("hour", "count"),
        n_rows=("n_points", "size"),
        first_hour=("hour", "min"),
        last_hour=("hour", "max"),
        total_path_km=("path_km", "sum"),
        mean_speed_kn=("speed_kn", "mean"),
        max_speed_kn=("speed_kn", "max"),
        mean_turn=("turn_abs_mean", "mean"),
        mean_straight=("straightness", "mean"),
        mean_stop=("stop_ratio", "mean"),
        c_lon=("c_lon", "mean"),
        c_lat=("c_lat", "mean"),
        s_lon=("s_lon", "first"),
        s_lat=("s_lat", "first"),
        lat_min=("c_lat", "min"), lat_max=("c_lat", "max"),
        lon_min=("c_lon", "min"), lon_max=("c_lon", "max"),
    ).reset_index()
    # 출발점(=항구 근사)에서 가장 멀리 간 거리
    f2 = f.merge(d[["rfid_id", "part_dt", "s_lon", "s_lat"]], on=["rfid_id", "part_dt"], suffixes=("", "_d"))
    f2["dist_from_start"] = haversine_km(f2["s_lon_d"], f2["s_lat_d"], f2["c_lon"], f2["c_lat"])
    d = d.merge(f2.groupby(["rfid_id", "part_dt"])["dist_from_start"].max().rename("max_range_km").reset_index(),
                on=["rfid_id", "part_dt"])
    d["duration_h"] = d["last_hour"] - d["first_hour"] + 1
    d["date"] = pd.to_datetime(d["part_dt"].astype(str), format="%Y%m%d")
    return d


# ─────────────────────────────────────────────
# 5. 사고 데이터
# ─────────────────────────────────────────────
ACC_COLS = ["acc_time", "coast_guard", "sea_area", "lat", "lon", "weather", "cause",
            "urgency", "acc_type", "n_ships", "n_people", "rescued", "injured", "dead",
            "missing", "ship_type", "license", "tonnage"]
ACC_COLS_KR = ["발생일시", "관할해경서", "발생해역", "위도", "경도", "기상상태", "발생원인",
               "긴급/비긴급", "발생유형", "사고선박수", "발생인원", "구조", "부상", "사망",
               "실종", "선종", "어업면허", "톤수"]


def license_to_class(s):
    """사고 엑셀 '어업면허' 문자열 → 4개 조업형태 (매칭 불가면 None)"""
    if not isinstance(s, str):
        return None
    if "복합" in s:
        return "연안복합"
    if "자망" in s:
        return "자망"
    if "통발" in s:
        return "통발"
    if any(k in s for k in ("채낚기", "연승", "낚시")):
        return "채낚기연승낚시"
    return None


def load_accidents(path):
    acc = pd.read_excel(path)
    if acc.shape[1] == len(ACC_COLS):  # 열 이름 인코딩 문제 회피: 위치 기반으로 이름 부여
        acc.columns = ACC_COLS
    acc["acc_time"] = pd.to_datetime(acc["acc_time"])
    acc["part_dt"] = acc["acc_time"].dt.strftime("%Y%m%d").astype(int)
    acc["hour"] = acc["acc_time"].dt.hour
    acc["cls"] = acc["license"].map(license_to_class)
    acc = acc.reset_index(drop=True)
    acc["acc_id"] = acc.index
    return acc


def match_accidents(acc, cand, radius_km=5.0, time_window_h=2, same_class=True):
    """
    사고 ↔ 항적 행 매칭 (rfid_id 가 없으므로 시공간 근접성으로 추정)
      cand: 00_build_cache 가 저장한 후보 행 (rfid_id, part_dt, hour, fishing_code, geom)
    return: 사고별 후보 (acc_id, rfid_id, part_dt, hour, dt_h, min_dist_km, rank)
    """
    cand = cand.dropna(subset=["hour"]).copy()
    cand["ts"] = pd.to_datetime(cand["part_dt"].astype(str), format="%Y%m%d") + pd.to_timedelta(
        cand["hour"].astype(int), unit="h")
    parsed = {}
    res = []
    for a in acc.itertuples():
        t0 = a.acc_time.floor("h")
        sub = cand[(cand["ts"] - t0).abs() <= pd.Timedelta(hours=time_window_h)]
        if same_class and a.cls is not None:
            sub = sub[sub["fishing_code"] == a.cls]
        for idx, r in zip(sub.index, sub.itertuples()):
            if idx not in parsed:
                parts = parse_multilinestring(r.geom)
                parsed[idx] = np.vstack(parts) if parts else np.empty((0, 2))
            pts = parsed[idx]
            if len(pts) == 0:
                continue
            dmin = float(haversine_km(pts[:, 0], pts[:, 1], a.lon, a.lat).min())
            if dmin <= radius_km:
                res.append((a.acc_id, r.rfid_id, r.part_dt, r.hour, (r.ts - t0) / pd.Timedelta(hours=1), dmin))
    m = pd.DataFrame(res, columns=["acc_id", "rfid_id", "part_dt", "hour", "dt_h", "min_dist_km"])
    if len(m):
        m = m.sort_values(["acc_id", "min_dist_km", "dt_h"])
        # 선박 단위 순위 (같은 선박의 여러 시간대 중 최소거리)
        best = m.groupby(["acc_id", "rfid_id"])["min_dist_km"].min().reset_index()
        best["rank"] = best.groupby("acc_id")["min_dist_km"].rank(method="first").astype(int)
        m = m.merge(best[["acc_id", "rfid_id", "rank"]], on=["acc_id", "rfid_id"])
    return m
