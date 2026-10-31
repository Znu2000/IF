"""
[FE-1] 시간(L1) 단위 피처 ─ 원본 CSV 1행(선박·날짜·시간대 1시간 항적) → 피처 1행

팀원 명세 통합 (HANDOFF §4, 진행2(피처).docx)
  · 나연 v2  : 튐 제거(A→B→A), 6구간 중앙값 평활 경로, 불가능 이동 제외, 10분 구간 속도·속도구간,
               정박 판정, 범위·모양·경로형태·회전·품질 피처
  · 지유     : raw/clean 거리, step 통계, convex hull, 중심 반경, segment sinuosity,
               path_efficiency, bearing_std, gap, spike, movement_per_point
  · 민성     : 지터 강건화 ─ 5-step 속도/방향/회전(중앙값·분위수), 체류점(반경 3σ ≥3점), 속도 구간
  · 진우     : 스펙트럼 엔트로피 / 저주파 비율
  · 가설(진행1 §6) : B 직선 투망(최장 직선 run, PCA 연신율), C 속도 구조(저속 연속 최대 길이, 속도 엔트로피),
                     D 선회(누적 회전 루프 수)
  · 일 단위 재방문(가설 A)용으로 6구간 평활 위치(b_lon*, b_lat*)를 같이 저장

좌표는 행마다 첫 점 기준 등거리 근사(m)로 투영. 점별 시각이 없으므로 "1시간 동안 균등 간격" 가정.

실행:  python fe_hourly.py                 (전체 6개월, 결과 features/hourly_MM.parquet)
       python fe_hourly.py --max-chunks 1  (동작 확인용)
       python fe_hourly.py --months 1 2    (특정 월만)
       python fe_hourly.py --src "<테스트셋 폴더>" --prefix test_hourly   (테스트셋, fishing_code 없음)
"""
import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import numpy as np
import pandas as pd

import fe_config as C

NAN = np.nan
COS_DEG = np.pi / 180


# ─────────────────────────────────────────────
# 0. 파싱 / 기본 연산
# ─────────────────────────────────────────────
def parse_parts(wkt):
    """WKT MULTILINESTRING → [ndarray(n_i, 2) (lon, lat)]"""
    if not isinstance(wkt, str) or "(" not in wkt:
        return []
    body = wkt[wkt.find("(") :].strip().strip("()")
    parts = []
    for p in body.split("),"):
        arr = np.array(p.replace("(", " ").replace(")", " ").replace(",", " ").split(), dtype=np.float64)
        if arr.size >= 2 and arr.size % 2 == 0:
            parts.append(arr.reshape(-1, 2))
    return parts


def _norm(d):
    return np.hypot(d[:, 0], d[:, 1])


def remove_spikes(q, passes=C.SPIKE_PASSES):
    """A→B→A 왕복 튐 B 제거 (A-B, B-C 모두 100m 초과 & A-C < 25%·(A-B + B-C))"""
    removed = 0
    for _ in range(passes):
        if len(q) < 3:
            break
        dab, dbc, dac = _norm(q[1:-1] - q[:-2]), _norm(q[2:] - q[1:-1]), _norm(q[2:] - q[:-2])
        sp = (dab > C.SPIKE_MIN_M) & (dbc > C.SPIKE_MIN_M) & (dac < C.SPIKE_RET_RATIO * (dab + dbc))
        if not sp.any():
            break
        keep = np.ones(len(q), bool)
        keep[1:-1] = ~sp
        removed += int(sp.sum())
        q = q[keep]
    return q, removed


def hull_area(p):
    """monotone chain convex hull 면적 (m²)"""
    if len(p) < 3:
        return 0.0
    p = np.unique(np.round(p, 1), axis=0)
    if len(p) < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for pt in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (pt[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (pt[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(pt)
        return h

    hull = np.array(half(p)[:-1] + half(p[::-1])[:-1])
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def spectral(sig):
    if sig.size < 8 or np.allclose(sig, sig[0]):
        return NAN, NAN
    p = np.abs(np.fft.rfft(sig - sig.mean()))[1:] ** 2
    tot = p.sum()
    if tot <= 0 or len(p) < 2:
        return NAN, NAN
    q = p / tot
    return float(-(q * np.log(q + 1e-12)).sum() / np.log(len(q))), float(q[: max(1, len(q) // 4)].sum())


def stay_points(p, r=C.STAY_R_M, min_pts=C.STAY_MIN_PTS):
    """체류점(Li et al. 2008 방식): 기준점에서 반경 r 안에 min_pts 이상 연속 → 체류 1회
    반환: 체류 횟수, 체류에 속한 점 수, 체류 중심 좌표 목록, 체류별 점 수"""
    n, i, n_stay, in_stay = len(p), 0, 0, 0
    centers, lens = [], []
    while i < n - 1:
        d = _norm(p[i + 1 :] - p[i])
        out = np.flatnonzero(d > r)
        j = i + 1 + (out[0] if out.size else n - i - 1)  # 처음으로 반경을 벗어난 점
        if j - i >= min_pts:
            n_stay += 1
            in_stay += j - i
            centers.append(p[i:j].mean(0))
            lens.append(j - i)
            i = j
        else:
            i += 1
    return n_stay, in_stay, centers, lens


def longest_true_run(mask):
    if not mask.any():
        return 0
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


# ─────────────────────────────────────────────
# 1. 1시간 항적 → 피처
# ─────────────────────────────────────────────
BIN_COLS = [f"b_lon{i}" for i in range(C.N_BINS)] + [f"b_lat{i}" for i in range(C.N_BINS)]
FEAT_COLS = [
    # 기록량·구조
    "n_points_raw", "n_points", "n_segments", "gap_count", "pts_per_seg", "uniq_pos_ratio",
    # 품질 (구성 C안: clean + 품질 피처)
    "n_spikes", "spike_ratio", "n_jumps", "n_bin_spikes", "n_bin_jumps", "bin_dispersion_m",
    "raw_clean_diff_m", "raw_clean_ratio", "overspeed_flag", "gap_dist_m", "max_gap_m",
    # 위치 (shortcut 위험 → 랭킹 시 LOC 플래그)
    "lon_c", "lat_c", "lon_s", "lat_s", "lon_e", "lat_e",
    # 범위·모양
    "bbox_w_m", "bbox_h_m", "bbox_diag_m", "bbox_area_km2", "bbox_aspect", "rg_m", "rg_med_m",
    "max_radius_m", "hull_area_m2", "pca_elong", "smooth_diag_m",
    # 거리·경로형태
    "dist_m", "dist_pts_m", "dist_raw_m", "net_disp_m", "max_from_start_m", "straightness",
    "path_over_diag", "path_over_rg", "seg_sinuosity_mean", "movement_per_point", "near_zero_disp",
    # 1-step 통계
    "step_mean_m", "step_med_m", "step_std_m", "step_max_m", "step_max_raw_m", "step_cv",
    # 10분 구간 속도 (나연)
    "sp_mean", "sp_std", "sp_p50", "sp_p90", "sp_max", "sp_change_mean",
    "frac_stop", "frac_slow", "frac_mid", "frac_fast", "n_stop_move_switch",
    # 방향·회전 (1-step, 이동 step만)
    "n_move_steps", "heading_R", "bearing_std", "turn_mean", "turn_std", "turn_max",
    "n_turn45", "n_turn90", "n_uturn", "large_turn_ratio", "uturn_ratio", "turn90_per_km", "loops",
    # 지터 강건 5-step (민성)
    "ks_speed_mean", "ks_speed_med", "ks_speed_p90", "ks_frac_stat", "ks_frac_drift",
    "ks_frac_fish", "ks_frac_transit", "ks_heading_R", "ks_turn_med", "ks_straight_max_m",
    "slow_run_max_min", "speed_entropy",
    # 체류점 + 체류 지점 배열 (v2: 통발 어구 기하 ─ 부표를 따라 줄지어 멈추는 패턴)
    "n_stays", "stay_frac", "stay_gap_med_m", "stay_gap_cv", "stay_line_elong", "stay_dur_med_min",
    # v2: 방향 지속성 / 표류→고속 재배치 사이클 (가설 D) / 최장 직선 구간 방위 (선박 단위 직선 방향 일관성 재료)
    "heading_autocorr", "drift_cycles", "ks_straight_bearing",
    # 스펙트럼 (진우)
    "spec_entropy", "lowfreq_ratio",
    # 정박
    "is_stationary",
]
OUT_COLS = FEAT_COLS + BIN_COLS
_IDX = {c: i for i, c in enumerate(OUT_COLS)}


def row_features(parts):
    f = {}
    n_raw = sum(len(p) for p in parts)
    f["n_points_raw"], f["n_segments"] = n_raw, len(parts)
    f["gap_count"], f["pts_per_seg"] = len(parts) - 1, n_raw / len(parts)

    lon0, lat0 = parts[0][0]
    kx, ky = 111_320.0 * np.cos(lat0 * COS_DEG), 110_574.0

    def proj(p):
        return np.column_stack(((p[:, 0] - lon0) * kx, (p[:, 1] - lat0) * ky))

    P = [proj(p) for p in parts]

    # raw 거리 / gap
    raw_steps = np.concatenate([_norm(np.diff(q, axis=0)) for q in P if len(q) >= 2] or [np.empty(0)])
    f["dist_raw_m"] = float(raw_steps.sum())
    f["step_max_raw_m"] = float(raw_steps.max()) if raw_steps.size else NAN
    gaps = np.array([np.hypot(*(P[i][0] - P[i - 1][-1])) for i in range(1, len(P))])
    f["gap_dist_m"] = float(gaps.sum()) if gaps.size else 0.0
    f["max_gap_m"] = float(gaps.max()) if gaps.size else 0.0

    # clean: 튐 제거 → step (15km 초과 점프 제외)
    Q, n_sp = [], 0
    for q in P:
        q2, r = remove_spikes(q)
        n_sp += r
        Q.append(q2)
    lens = [len(q) for q in Q]
    CP = np.vstack(Q)
    n = len(CP)
    f["n_points"], f["n_spikes"], f["spike_ratio"] = n, n_sp, n_sp / n_raw
    seg_steps = [_norm(np.diff(q, axis=0)) if len(q) >= 2 else np.empty(0) for q in Q]
    steps = np.concatenate(seg_steps)
    jump = steps > C.JUMP_PT_M
    f["n_jumps"] = int(jump.sum())
    st = steps[~jump]
    dist_pts = float(st.sum())
    f["dist_pts_m"] = dist_pts
    f["raw_clean_diff_m"] = f["dist_raw_m"] - dist_pts
    f["raw_clean_ratio"] = f["dist_raw_m"] / max(dist_pts, 1.0)
    f["movement_per_point"] = dist_pts / n
    f["uniq_pos_ratio"] = len(np.unique(np.round(CP / 10.0), axis=0)) / n

    # 위치 (위경도, 중앙값 중심)
    lon = CP[:, 0] / kx + lon0
    lat = CP[:, 1] / ky + lat0
    f["lon_c"], f["lat_c"] = float(np.median(lon)), float(np.median(lat))
    f["lon_s"], f["lat_s"], f["lon_e"], f["lat_e"] = lon[0], lat[0], lon[-1], lat[-1]

    # 범위·모양
    w, h = np.ptp(CP[:, 0]), np.ptp(CP[:, 1])
    f["bbox_w_m"], f["bbox_h_m"], f["bbox_diag_m"] = w, h, float(np.hypot(w, h))
    f["bbox_area_km2"] = w * h / 1e6
    f["bbox_aspect"] = max(w, h) / max(min(w, h), 1.0)
    dc = _norm(CP - CP.mean(0))
    f["rg_m"] = float(np.sqrt((dc ** 2).mean()))
    f["max_radius_m"] = float(dc.max())
    f["rg_med_m"] = float(np.median(_norm(CP - np.median(CP, 0))))
    f["hull_area_m2"] = hull_area(CP)
    if n >= 3 and f["bbox_diag_m"] > 1:
        ev = np.linalg.eigvalsh(np.cov(CP.T))
        f["pca_elong"] = float(np.log1p(ev[-1] / max(ev[0], 1.0)))

    # 경로 형태
    f["net_disp_m"] = float(np.hypot(*(CP[-1] - CP[0])))
    f["max_from_start_m"] = float(_norm(CP - CP[0]).max())
    f["near_zero_disp"] = float(f["net_disp_m"] < 50)
    if dist_pts > 1:
        f["straightness"] = min(f["net_disp_m"] / dist_pts, 1.0)
        f["path_over_diag"] = dist_pts / max(f["bbox_diag_m"], 1.0)
        f["path_over_rg"] = dist_pts / max(f["rg_m"], 1.0)
    sins = []
    for q, s in zip(Q, seg_steps):
        if len(q) >= 2:
            d = np.hypot(*(q[-1] - q[0]))
            if d >= 100:
                sins.append(s[s <= C.JUMP_PT_M].sum() / d)
    f["seg_sinuosity_mean"] = float(np.mean(sins)) if sins else NAN

    # 1-step 통계
    if st.size:
        f["step_mean_m"], f["step_med_m"] = float(st.mean()), float(np.median(st))
        f["step_std_m"], f["step_max_m"] = float(st.std()), float(st.max())
        f["step_cv"] = f["step_std_m"] / max(f["step_mean_m"], 1e-6)
        f["spec_entropy"], f["lowfreq_ratio"] = spectral(st)

    # 10분 구간 평활 경로 (나연)
    nb = min(C.N_BINS, n)
    bid = (np.arange(n) * nb) // n                  # 점 순서대로 nb개 구간 (np.array_split 과 동일 크기)
    mx = np.bincount(bid).max()
    pad = np.full((nb, mx, 2), np.nan)
    pos = np.arange(n) - np.searchsorted(bid, bid)  # 구간 내 순번
    pad[bid, pos] = CP
    # 구간별 중앙값: NaN 은 정렬 시 뒤로 가므로 구간 점 수(cnt)로 가운데 위치를 바로 집음 (np.nanmedian 보다 빠름)
    cnt = np.bincount(bid)
    ar, lo, hi = np.arange(nb), (cnt - 1) // 2, cnt // 2
    s = np.sort(pad, axis=1)
    B = (s[ar, lo] + s[ar, hi]) / 2
    dd = np.sort(np.hypot(pad[..., 0] - B[:, None, 0], pad[..., 1] - B[:, None, 1]), axis=1)
    f["bin_dispersion_m"] = float(((dd[ar, lo] + dd[ar, hi]) / 2).mean())
    B, nbs = remove_spikes(B, passes=1)
    f["n_bin_spikes"] = nbs
    bl, bt = B[:, 0] / kx + lon0, B[:, 1] / ky + lat0
    for i in range(min(len(B), C.N_BINS)):
        f[f"b_lon{i}"], f[f"b_lat{i}"] = bl[i], bt[i]
    f["smooth_diag_m"] = float(np.hypot(*np.ptp(B, axis=0))) if len(B) > 1 else 0.0
    f["is_stationary"] = float(f["smooth_diag_m"] < C.STATIONARY_RANGE_M)
    if len(B) >= 2:
        bd = _norm(np.diff(B, axis=0))
        bj = bd > C.JUMP_BIN_M
        f["n_bin_jumps"] = int(bj.sum())
        bd = bd[~bj]
        f["dist_m"] = float(bd.sum())
        if bd.size:
            sp = bd * nb / 1852.0  # 구간 1개 ≈ 60/nb 분 → knot
            f["sp_mean"], f["sp_std"] = float(sp.mean()), float(sp.std())
            f["sp_p50"], f["sp_p90"], f["sp_max"] = float(np.median(sp)), float(np.quantile(sp, 0.9)), float(sp.max())
            f["sp_change_mean"] = float(np.abs(np.diff(sp)).mean()) if sp.size > 1 else 0.0
            f["frac_stop"] = float((sp <= 0.5).mean())
            f["frac_slow"] = float(((sp > 0.5) & (sp <= 3)).mean())
            f["frac_mid"] = float(((sp > 3) & (sp <= 6)).mean())
            f["frac_fast"] = float((sp > 6).mean())
            mv = sp > 0.5
            f["n_stop_move_switch"] = int((mv[1:] != mv[:-1]).sum())
            f["overspeed_flag"] = float(sp.max() > 20)
    else:
        f["dist_m"], f["n_bin_jumps"] = 0.0, 0

    # 방향·회전 (1-step, 이동 step만, 세그먼트 내부)
    heads, turns = [], []
    for q in Q:
        if len(q) < 2:
            continue
        d = np.diff(q, axis=0)
        dl = _norm(d)
        m = (dl > C.MOVE_STEP_M) & (dl <= C.JUMP_PT_M)
        hh = np.arctan2(d[m, 1], d[m, 0])
        heads.append(hh)
        if hh.size >= 2:
            turns.append((np.diff(hh) + np.pi) % (2 * np.pi) - np.pi)
    hh = np.concatenate(heads) if heads else np.empty(0)
    f["n_move_steps"] = int(hh.size)
    if hh.size:
        R = float(np.hypot(np.cos(hh).mean(), np.sin(hh).mean()))
        f["heading_R"] = R
        f["bearing_std"] = float(np.degrees(np.sqrt(-2 * np.log(min(max(R, 1e-6), 1.0)))))
    tt = np.concatenate(turns) if turns else np.empty(0)
    if tt.size:
        ta = np.degrees(np.abs(tt))
        f["turn_mean"], f["turn_std"], f["turn_max"] = float(ta.mean()), float(ta.std()), float(ta.max())
        f["n_turn45"], f["n_turn90"], f["n_uturn"] = int((ta > 45).sum()), int((ta > 90).sum()), int((ta > 150).sum())
        f["large_turn_ratio"], f["uturn_ratio"] = float((ta > 60).mean()), float((ta > 150).mean())
        f["turn90_per_km"] = f["n_turn90"] / max(dist_pts / 1000, 0.05)
        f["loops"] = float(abs(np.degrees(tt.sum())) / 360.0)

    # 지터 강건 k-step (민성): k점 간격 거리, 균등간격 가정 → dt = k·60/n 분
    k = C.K_STEP
    dt_h = k / max(n, 1)
    ksp, kh, kd, cycles = [], [], [], 0
    for q in Q:
        if len(q) > k:
            dk = _norm(q[k:] - q[:-k])
            ksp.append(dk[dk <= C.JUMP_PT_M] / 1852.0 / dt_h)
            s = q[::k]
            ds = np.diff(s, axis=0)
            dl = _norm(ds)
            m = (dl > 15) & (dl <= C.JUMP_PT_M)
            kh.append(np.arctan2(ds[m, 1], ds[m, 0]))
            kd.append(dl[m])
            # 표류(0.3~2kn) 직후 고속(>4kn) 재배치 + 120° 넘게 방향 전환 = 사이클 1회 (낚시형 톱니 패턴)
            if len(dl) >= 2:
                vs = dl / 1852.0 / dt_h
                hs = np.arctan2(ds[:, 1], ds[:, 0])
                turn = np.degrees(np.abs((hs[1:] - hs[:-1] + np.pi) % (2 * np.pi) - np.pi))
                cycles += int(((vs[:-1] >= 0.3) & (vs[:-1] < 2) & (vs[1:] > 4) & (dl[:-1] > 15) & (turn > 120)).sum())
    f["drift_cycles"] = cycles
    ksp = np.concatenate(ksp) if ksp else np.empty(0)
    if ksp.size:
        f["ks_speed_mean"], f["ks_speed_med"] = float(ksp.mean()), float(np.median(ksp))
        f["ks_speed_p90"] = float(np.quantile(ksp, 0.9))
        bands = np.array([(ksp < 0.3).mean(), ((ksp >= 0.3) & (ksp < 2)).mean(),
                          ((ksp >= 2) & (ksp < 5)).mean(), (ksp >= 5).mean()])
        f["ks_frac_stat"], f["ks_frac_drift"], f["ks_frac_fish"], f["ks_frac_transit"] = bands
        pb = bands[bands > 0]
        f["speed_entropy"] = float(-(pb * np.log(pb)).sum() / np.log(4))
        f["slow_run_max_min"] = longest_true_run((ksp >= 0.3) & (ksp < 2)) * 60.0 / n
    khs = [h_ for h_ in kh if h_.size]
    if khs:
        allh = np.concatenate(khs)
        f["ks_heading_R"] = float(np.hypot(np.cos(allh).mean(), np.sin(allh).mean()))
        kt, best, bseg, ac = [], 0.0, None, []
        for h_, d_ in zip(kh, kd):
            if h_.size >= 1 and d_[0] > best:
                best, bseg = float(d_[0]), h_[:1]
            if h_.size >= 2:
                t = np.degrees(np.abs((np.diff(h_) + np.pi) % (2 * np.pi) - np.pi))
                kt.append(t)
                cur, cs = d_[0], 0
                for jj in range(1, len(d_)):  # 가설 B: 회전 15° 미만으로 이어진 최장 직선
                    if t[jj - 1] < C.STRAIGHT_TURN_DEG:
                        cur += d_[jj]
                    else:
                        cur, cs = d_[jj], jj
                    if cur > best:
                        best, bseg = cur, h_[cs:jj + 1]
            if h_.size >= 3:  # 2스텝 떨어진 진행 방향의 유사도 (1=계속 같은 방향)
                ac.append(np.cos(h_[2:] - h_[:-2]))
        f["ks_straight_max_m"] = best
        if bseg is not None and best >= 300:  # 직선 구간의 축 방위(0~180°): 방향 일관성 계산용
            f["ks_straight_bearing"] = float(np.degrees(np.arctan2(np.sin(2 * bseg).mean(),
                                                                   np.cos(2 * bseg).mean())) / 2 % 180)
        if ac:
            f["heading_autocorr"] = float(np.concatenate(ac).mean())
        if kt:
            f["ks_turn_med"] = float(np.median(np.concatenate(kt)))

    # 체류점
    ns, ins, centers, lens = stay_points(CP)
    f["n_stays"], f["stay_frac"] = ns, ins / n
    if lens:
        f["stay_dur_med_min"] = float(np.median(lens)) * 60.0 / n
    if len(centers) >= 2:  # 연속 체류 지점 사이 간격: 통발은 부표 간격으로 규칙적
        sc = np.array(centers)
        gaps = _norm(np.diff(sc, axis=0))
        f["stay_gap_med_m"] = float(np.median(gaps))
        f["stay_gap_cv"] = float(gaps.std() / max(gaps.mean(), 1.0))
        if len(sc) >= 3 and np.ptp(sc, axis=0).max() > 1:  # 체류 지점들이 일렬로 놓였나
            ev = np.linalg.eigvalsh(np.cov(sc.T))
            f["stay_line_elong"] = float(np.log1p(ev[-1] / max(ev[0], 1.0)))
    return f


def process_chunk(geoms):
    out = np.full((len(geoms), len(OUT_COLS)), np.nan, dtype=np.float32)
    for i, g in enumerate(geoms):
        parts = parse_parts(g)
        if not parts:
            continue
        try:
            f = row_features(parts)
        except Exception:
            continue
        row = out[i]
        for c, v in f.items():
            row[_IDX[c]] = v
    return out


# ─────────────────────────────────────────────
# 2. 파일 단위 실행 (청크 병렬, 동시에 떠 있는 청크 수 제한 → 메모리 일정)
# ─────────────────────────────────────────────
def clean_keys(raw):
    pdt = raw["part_dt"].where(raw["part_dt"].str.fullmatch(r"\d{8}", na=False))
    rid = raw["rfid_id"].where(raw["rfid_id"].str.fullmatch(r"\d+", na=False))
    keep = pdt.notna() & rid.notna()
    raw = raw[keep]
    hour = pd.to_numeric(raw["rfid_time"], errors="coerce")
    hour = hour.where(hour.between(0, 23))
    key = pd.DataFrame({
        "part_dt": pd.to_numeric(pdt[keep]).astype("int32"),
        "rfid_id": pd.to_numeric(rid[keep]).astype("int64"),
        "hour": hour.fillna(-1).astype("int8"),
        "fishing_code": raw["fishing_code"].where(raw["fishing_code"].isin(C.CLASSES)),
    }, index=raw.index)
    dup = key.duplicated(["rfid_id", "part_dt", "hour"]) & (key["hour"] >= 0)
    return key[~dup], raw.loc[~dup, "geom"].tolist(), int((~keep).sum()), int(dup.sum())


def run_file(path, ex, max_chunks=None):
    t0 = time.time()
    head = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    cols = ["part_dt", "rfid_id", "rfid_time", "geom"] + (["fishing_code"] if "fishing_code" in head else [])
    reader = pd.read_csv(path, chunksize=C.CHUNK_SIZE, dtype=str, encoding="utf-8-sig", usecols=cols)
    pending, results, broken, dups = {}, {}, 0, 0
    for ci, raw in enumerate(reader):
        if max_chunks is not None and ci >= max_chunks:
            break
        if "fishing_code" not in raw:  # 테스트셋: 정답 없음
            raw["fishing_code"] = None
        key, geoms, b, d = clean_keys(raw)
        broken, dups = broken + b, dups + d
        # 청크를 작업자 수만큼 쪼개서 제출
        for si, sl in enumerate(np.array_split(np.arange(len(geoms)), C.N_JOBS)):
            fut = ex.submit(process_chunk, [geoms[i] for i in sl])
            pending[fut] = (ci, si)
        results[ci] = [key, {}]
        while len(pending) > 2 * C.N_JOBS:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fu in done:
                c, s = pending.pop(fu)
                results[c][1][s] = fu.result()
        print(f"  chunk {ci:02d} 제출 ({len(geoms):,}행, 누적 {time.time() - t0:5.0f}s)", flush=True)
    for fu in list(pending):
        c, s = pending.pop(fu)
        results[c][1][s] = fu.result()

    frames = []
    for ci in sorted(results):
        key, parts = results[ci]
        arr = np.vstack([parts[s] for s in sorted(parts)])
        frames.append(pd.concat([key.reset_index(drop=True), pd.DataFrame(arr, columns=OUT_COLS)], axis=1))
    df = pd.concat(frames, ignore_index=True)
    return df, broken, dups, time.time() - t0


def main(max_chunks=None, months=None, src=None, prefix="hourly"):
    from pathlib import Path
    files = sorted(Path(src).glob(C.VPASS_GLOB)) if src else C.vpass_files()
    print(f"N_JOBS={C.N_JOBS}, CHUNK={C.CHUNK_SIZE:,}, 파일 {len(files)}개 → {prefix}_MM.parquet")
    with ProcessPoolExecutor(C.N_JOBS) as ex:
        for f in files:
            mm = int(re.search(r"(\d+)월", f.name).group(1))
            if months and mm not in months:
                continue
            out = C.FEAT_DIR / f"{prefix}_{mm:02d}.parquet"
            print(f"\n▶ {f.name} → {out.name}")
            df, broken, dups, sec = run_file(f, ex, max_chunks)
            df["fishing_code"] = df["fishing_code"].astype("category")
            df.to_parquet(out, index=False)
            print(f"  ✅ {len(df):,}행 / 깨진 행 {broken} / 중복키 {dups} / 파싱 실패 "
                  f"{int(df['n_points'].isna().sum())} / {sec / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--months", type=int, nargs="*", default=None)
    ap.add_argument("--src", default=None, help="입력 폴더 (예: 테스트셋 폴더). 기본 = 학습 데이터")
    ap.add_argument("--prefix", default="hourly", help="출력 파일 접두사 (테스트: test_hourly)")
    a = ap.parse_args()
    main(a.max_chunks, a.months, a.src, a.prefix)
