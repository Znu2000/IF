"""
피처 분류표 (대분류 12 → 세부 구분 → 기본 피처) + 기본 피처 설명/출처

· 기본 피처 = 시간(L1)·선박-일(L2)·선박(L3) 단위에서 직접 계산한 값.
  선박 피처 346개는 기본 피처에 접두사(mv_, fh_)·접미사(_mean/_median/_std/_p90, _q50/_q90)를 붙여 파생됨.
· cv_common.feat_meta, make_feature_catalog.py 가 이 표를 공통으로 사용.
"""
import re

# ─────────────────────────────────────────────
# 기본 피처 설명: 이름 → (설명, 출처)
# ─────────────────────────────────────────────
H = {  # 시간(L1) 1시간 항적
    "n_points_raw": ("원본 GPS 점 수", "지유·나연·진우"), "n_points": ("튐 제거 후 점 수", "지유·나연·진우"),
    "n_segments": ("MultiLineString 구간 수 (신호 끊김+1)", "지유·나연·진우"), "gap_count": ("관측 끊김 횟수", "지유"),
    "pts_per_seg": ("구간당 평균 점 수", "지유"), "uniq_pos_ratio": ("고유 위치(10m 격자) 비율: 낮으면 제자리", "나연"),
    "n_spikes": ("제거된 A→B→A 튐 점 수", "나연·지유"), "spike_ratio": ("튐 점 비율", "나연·지유"),
    "n_jumps": ("15km 초과 점프 수", "나연"), "n_bin_spikes": ("10분 구간 수준 튐 수", "나연"),
    "n_bin_jumps": ("10분 구간 간 7km 초과 점프 수", "나연"), "bin_dispersion_m": ("10분 구간 내 점 흩어짐(m)", "나연"),
    "raw_clean_diff_m": ("원거리 − 정제거리 (튐으로 부풀려진 거리, m)", "지유"),
    "raw_clean_ratio": ("원거리 / 정제거리", "지유"), "overspeed_flag": ("10분 속도 20kn 초과 여부", "나연"),
    "gap_dist_m": ("구간 사이 끊김 거리 합(m)", "지유·진우"), "max_gap_m": ("최대 끊김 거리(m)", "지유"),
    "lon_c": ("중심 경도(중앙값)", "공통"), "lat_c": ("중심 위도(중앙값)", "공통"),
    "lon_s": ("시작 경도", "나연·진우"), "lat_s": ("시작 위도", "나연·진우"),
    "lon_e": ("끝 경도", "나연·진우"), "lat_e": ("끝 위도", "나연·진우"),
    "bbox_w_m": ("활동 사각형 가로(m)", "지유·나연·진우"), "bbox_h_m": ("활동 사각형 세로(m)", "지유·나연·진우"),
    "bbox_diag_m": ("활동 사각형 대각선(m)", "나연"), "bbox_area_km2": ("활동 사각형 넓이(km²)", "지유·진우"),
    "bbox_aspect": ("활동 사각형 가로세로비: 길쭉할수록 큼", "나연"),
    "rg_m": ("회전반경: 중심에서 퍼진 정도(m)", "민성·나연·진우"),
    "rg_med_m": ("중앙 반경: 중앙점까지 거리의 중앙값(m, 지터 강건)", "민성"),
    "max_radius_m": ("중심에서 가장 먼 점까지 거리(m)", "지유"), "hull_area_m2": ("볼록껍질 면적(m²)", "지유·민성"),
    "pca_elong": ("PCA 연신율 log(1+λ1/λ2): 직선 모양일수록 큼", "가설 B"),
    "smooth_diag_m": ("평활 경로 범위 대각선(m)", "나연"), "dist_m": ("평활 경로 이동거리(m, 메인 거리)", "나연"),
    "dist_pts_m": ("점 단위 정제 이동거리(m)", "지유·나연·진우"), "dist_raw_m": ("점 단위 원거리(m)", "지유·나연"),
    "net_disp_m": ("시작→끝 직선 변위(m)", "공통"), "max_from_start_m": ("시작점에서 최대 이격(m)", "나연"),
    "straightness": ("직진도 = 변위/거리 (0~1, 1=일직선)", "공통"),
    "path_over_diag": ("경로길이/범위 대각선: 같은 곳을 반복할수록 큼", "나연"),
    "path_over_rg": ("경로길이/회전반경", "나연"), "seg_sinuosity_mean": ("구간별 구불거림 평균 (변위 100m 이상 구간)", "지유"),
    "movement_per_point": ("점당 이동량(m)", "지유"), "near_zero_disp": ("변위 50m 미만 여부", "지유"),
    "step_mean_m": ("점 간 step 평균(m)", "지유·민성·진우"), "step_med_m": ("점 간 step 중앙값(m)", "지유"),
    "step_std_m": ("점 간 step 표준편차(m)", "지유·진우"), "step_max_m": ("점 간 step 최대(m)", "지유·진우"),
    "step_max_raw_m": ("원본 step 최대(m)", "진우"), "step_cv": ("step 변동계수: 이동 크기가 들쭉날쭉한 정도", "민성"),
    "sp_mean": ("10분 구간 추정속도 평균(kn)", "나연·진우"), "sp_std": ("10분 속도 표준편차", "나연"),
    "sp_p50": ("10분 속도 중앙값", "나연"), "sp_p90": ("10분 속도 90% 분위", "나연"), "sp_max": ("10분 속도 최대", "나연"),
    "sp_change_mean": ("10분 속도 변화량 평균(가감속)", "나연"),
    "frac_stop": ("정지(≤0.5kn) 구간 비율", "나연·진우"), "frac_slow": ("저속(0.5~3kn) 구간 비율", "나연"),
    "frac_mid": ("중속(3~6kn) 구간 비율", "나연"), "frac_fast": ("고속(>6kn) 구간 비율", "나연"),
    "n_stop_move_switch": ("정지↔이동 전환 횟수", "나연"), "n_move_steps": ("10m 이상 이동 step 수", "나연"),
    "heading_R": ("방향 일관성 (1=한 방향, 0=사방)", "진우·나연·민성"), "bearing_std": ("진행방향 원형 표준편차(°)", "지유"),
    "turn_mean": ("회전각 평균(°)", "공통"), "turn_std": ("회전각 표준편차", "공통"), "turn_max": ("최대 회전각", "지유"),
    "n_turn45": ("45° 초과 회전 수", "나연"), "n_turn90": ("90° 초과 회전 수", "나연·진우"),
    "n_uturn": ("150° 초과 U턴 수", "나연·민성"), "large_turn_ratio": ("60° 초과 급선회 비율", "지유·민성·진우"),
    "uturn_ratio": ("U턴 비율", "나연"), "turn90_per_km": ("km당 90° 초과 회전 수", "나연"),
    "loops": ("누적 회전/360 = 제자리 맴돈 루프 수", "가설 D"),
    "ks_speed_mean": ("5-step 속도 평균(kn, 지터 강건)", "민성"), "ks_speed_med": ("5-step 속도 중앙값(kn)", "민성"),
    "ks_speed_p90": ("5-step 속도 90% 분위", "민성"), "ks_frac_stat": ("5-step 정지(<0.3kn) 비율", "민성"),
    "ks_frac_drift": ("5-step 표류(0.3~2kn) 비율", "민성"), "ks_frac_fish": ("5-step 조업속도(2~5kn) 비율", "민성"),
    "ks_frac_transit": ("5-step 항해(≥5kn) 비율", "민성"), "ks_heading_R": ("5-step 방향 일관성", "민성"),
    "ks_turn_med": ("5-step 회전각 중앙값(°)", "민성"),
    "ks_straight_max_m": ("5-step 최장 직선 이동(m, 회전 15° 미만 연속) = 투망 길이 추정", "민성·가설 B"),
    "slow_run_max_min": ("저속(0.3~2kn) 최장 연속 시간(분) = 양망 추정", "가설 C"),
    "speed_entropy": ("속도 구간 분포 엔트로피: 여러 속도를 섞어 쓸수록 큼", "가설 C"),
    "n_stays": ("체류점 횟수 (반경 30m 안에 3점 이상 머묾)", "민성"), "stay_frac": ("체류점에 속한 점 비율", "민성"),
    "spec_entropy": ("step 길이 스펙트럼 엔트로피: 이동 리듬의 불규칙성", "진우"),
    "lowfreq_ratio": ("step 저주파 에너지 비율: 느린 주기 변화의 비중", "진우"),
    "is_stationary": ("정박 여부 (평활 범위 <100m)", "나연"),
}
D = {  # 선박-일(L2)
    "n_hours": ("하루 기록 시간대 수", "나연·민성"), "first_hour": ("첫 기록 시각", "민성·지유"),
    "last_hour": ("마지막 기록 시각", "민성·지유"), "n_moving_hours": ("이동 시간대 수", "나연"),
    "stationary_ratio": ("정박 시간대 비율", "지유"), "n_points_day": ("하루 점 수", "나연"),
    "n_segments_mean": ("시간당 평균 구간 수", "나연"), "frac_stop_mean": ("평균 정지 비율", "나연"),
    "n_jumps_day": ("하루 15km 초과 점프 수", "나연"), "home_km_max": ("모항에서 최대 거리(km)", "민성·나연"),
    "home_km_mean": ("모항에서 평균 거리(km)", "나연"), "port_km_mean": ("최근접 공용항구까지 평균 거리(km)", "나연"),
    "in_port_ratio": ("항구 정박 시간 비율", "나연·민성"), "sea_stop_share": ("해상 정지 시간 비율", "민성"),
    "drift_share": ("표류(0.3~2kn) 시간 비율", "민성"), "fish_share": ("조업속도(2~5kn) 시간 비율", "민성"),
    "transit_share": ("항해(>5kn, 직선) 시간 비율", "민성"), "n_transitions": ("시간대 상태 전환 횟수", "민성"),
    "departures": ("출항 횟수 (항구→비항구)", "민성"),
    "inter_hour_jump_med_km": ("연속 시간대 끝→다음 시작 거리 중앙값(km)", "민성"),
    "active_block_count": ("이동 블록 수 (투망·양망 블록)", "가설 E"),
    "night_ratio": ("이동 중 야간(18~06시) 비율", "민성·지유"),
    "mv_share_dawn": ("이동 시간 중 새벽(0~5시) 비중", "나연"), "mv_share_morning": ("이동 시간 중 오전(6~11시) 비중", "나연"),
    "mv_share_afternoon": ("이동 시간 중 오후(12~17시) 비중", "나연"), "mv_share_night": ("이동 시간 중 밤(18~23시) 비중", "나연"),
    "mv_first_hour": ("조업(이동) 시작 시각", "나연"), "mv_last_hour": ("조업(이동) 종료 시각", "나연"),
    "lon_c": ("하루 중심 경도", "나연"), "lat_c": ("하루 중심 위도", "나연"),
    "lon_start": ("하루 시작 경도", "나연"), "lat_start": ("하루 시작 위도", "나연"),
    "lon_end": ("하루 끝 경도", "나연"), "lat_end": ("하루 끝 위도", "나연"),
    "home_km_start": ("하루 시작 위치의 모항 거리", "나연"), "home_km_end": ("하루 끝 위치의 모항 거리", "나연"),
    "month": ("월", "민성"), "moving_ratio": ("이동 시간대 비율", "나연"),
    "active_span_h": ("첫~마지막 기록 시간 폭", "나연"), "mv_span_h": ("조업 시간 폭 (첫~마지막 이동)", "나연·민성"),
    "dist_day_km": ("하루 평활 이동거리(km)", "나연·지유"), "dist_raw_day_km": ("하루 원거리(km)", "나연"),
    "jump_share_day": ("튐 비중 = 1 − 평활/원거리", "나연"), "day_net_disp_km": ("하루 시작→끝 변위(km)", "나연"),
    "day_straightness": ("하루 직진도 = 변위/거리", "가설 B"), "returned_to_port": ("귀항 여부", "나연"),
    "day_rg_km": ("시간대 중심들의 퍼짐(km)", "나연"), "day_bbox_diag_km": ("하루 활동범위 대각선(km)", "나연"),
    "day_range_q95_km": ("첫 위치에서 95% 분위 이격(km, 지터 강건 day_range)", "민성"),
    "longest_streak": ("최장 연속 이동 시간 수", "민성·가설 E"),
    "max_block_gap_h": ("이동 블록 사이 최대 휴식(시간)", "가설 E"),
    "mv_lon_c": ("이동 시간대 대표 경도", "나연"), "mv_lat_c": ("이동 시간대 대표 위도", "나연"),
    "return_after_absence": ("200m 격자를 2시간 이상 떠났다 돌아온 횟수", "가설 A"),
    "revisit_cell_ratio": ("200m 격자 중 여러 시간대에 방문한 칸 비율", "가설 A"),
    "n_cells_200m": ("방문한 200m 격자 수", "가설 A"), "top3_cell_share": ("상위 3칸 체류 비중", "가설 A"),
}
S3 = {  # 선박(L3) 전용
    "n_days": ("기록일 수", "나연"), "n_months": ("기록 개월 수", "나연"), "n_moving_days": ("출어일 수", "나연"),
    "moving_days_ratio": ("출어일 비율", "나연"), "moving_days_per_month": ("월평균 출어일", "나연"),
    "hours_per_day": ("하루 평균 기록 시간", "나연"), "port_share": ("항구 정박 시간 비율", "민성"),
    "sea_stationary_share": ("해상 정지 시간 비율", "민성"), "fishing_speed_share": ("조업속도 시간 비율", "민성"),
    "drift_share": ("표류(0.3~2kn) 시간 비율 (전체 기간)", "민성"),
    "transit_share": ("항해(>5kn, 직선) 시간 비율 (전체 기간)", "민성"),
    "lon_c": ("선박 대표 경도 (전체 시간 중앙값)", "공통"), "lat_c": ("선박 대표 위도 (전체 시간 중앙값)", "공통"),
    "sea_stationary_share_sea": ("항구 제외 시간 중 해상 정지 비율", "민성(H4)"),
    "drift_share_sea": ("항구 제외 시간 중 표류 비율", "민성(H4)"),
    "fishing_speed_share_sea": ("항구 제외 시간 중 조업속도 비율", "민성(H4)"),
    "transit_share_sea": ("항구 제외 시간 중 항해 비율", "민성(H4)"),
    "moving_hour_share": ("전체 시간 중 이동 시간 비율", "나연"), "returned_ratio": ("출어일 귀항 비율", "나연"),
    "transitions_per_day": ("하루 평균 상태 전환", "민성"), "trips_per_day": ("하루 평균 출항 수", "민성"),
    "trip_hours_q50": ("항차 시간 중앙값", "민성"), "trip_max_dist_q50": ("항차 최대거리 중앙값", "민성"),
    "departure_hour_q50": ("출항(조업 시작) 시각 중앙값", "민성"),
    "fishing_night_share": ("조업 시간대 중 야간 비율", "민성"),
    "drift_heading_R_q50": ("표류 시간대 5-step 방향 일관성 중앙값", "민성"),
    "cross_day_revisit": ("조업 1km 격자 중 여러 날 다시 찾은 칸 비율", "민성·가설 A"),
    "site_jaccard_mean": ("연속 출어일 간 어장(1km 격자) 겹침(Jaccard) 평균", "가설 A"),
    "area_range_km": ("조업 해역 넓이 (일별 중심 5~95% 범위, km)", "나연"),
    "fishing_area_radius": ("조업 중심점 퍼짐 반경 중앙값(km)", "민성"),
    "day_type_entropy": ("출어일 유형 분포 엔트로피 (날마다 다른 조업일수록 큼)", "가설 G"),
    "home_lon": ("모항 경도", "민성"), "home_lat": ("모항 위도", "민성"),
    "home_port_km": ("모항↔최근접 공용항구 거리(km)", "나연"),
    "fishing_lon": ("조업 중심 경도 (중앙값)", "민성"), "fishing_lat": ("조업 중심 위도 (중앙값)", "민성"),
}
for _m in range(1, 7):
    S3[f"season_m{_m}_share"] = (f"{_m}월 출어일 비중", "가설 H")
for _k in range(8):
    S3[f"daytype_{_k}_share"] = (f"출어일 유형 {_k}번 비중 (하루 피처 KMeans 8군집, 라벨 미사용)", "가설 G")

# ─────────────────────────────────────────────
# 분류표: (대분류, 세부 구분) → (무엇을 재나, [기본 피처])
# ─────────────────────────────────────────────
TAXONOMY = [
    ("속도 구조(C)", "10분 구간 속도", "평활 경로 10분 간격 추정 속도(kn)의 통계",
     ["sp_mean", "sp_std", "sp_p50", "sp_p90", "sp_max", "sp_change_mean"]),
    ("속도 구조(C)", "속도 구간 비율", "정지/저속/중속/고속 구간이 차지하는 비율",
     ["frac_stop", "frac_slow", "frac_mid", "frac_fast", "frac_stop_mean"]),
    ("속도 구조(C)", "5-step 강건 속도", "5개 점 간격으로 잰 속도: GPS 지터에 덜 흔들림 (민성 검증)",
     ["ks_speed_mean", "ks_speed_med", "ks_speed_p90", "ks_frac_stat", "ks_frac_drift", "ks_frac_fish", "ks_frac_transit"]),
    ("속도 구조(C)", "저속 연속·속도 다양성", "양망 같은 긴 저속 구간, 여러 속도를 섞어 쓰는 정도",
     ["slow_run_max_min", "speed_entropy"]),
    ("속도 구조(C)", "점 간 step 크기", "연속 GPS 점 사이 거리의 크기와 변동",
     ["step_mean_m", "step_med_m", "step_std_m", "step_max_m", "step_max_raw_m", "step_cv", "movement_per_point"]),
    ("속도 구조(C)", "이동 리듬(스펙트럼)", "step 길이 신호의 주파수 분석 (진우)", ["spec_entropy", "lowfreq_ratio"]),
    ("속도 구조(C)", "시간대 간 점프", "앞 시간 끝점↔다음 시간 시작점 거리", ["inter_hour_jump_med_km"]),
    ("방향·회전(D)", "방향 일관성", "한 방향으로 가는지, 사방으로 흩어지는지",
     ["heading_R", "bearing_std", "ks_heading_R", "drift_heading_R_q50"]),
    ("방향·회전(D)", "회전각", "진행 방향이 꺾이는 각도", ["turn_mean", "turn_std", "turn_max", "ks_turn_med"]),
    ("방향·회전(D)", "급선회·U턴", "크게 꺾거나 되돌아가는 횟수/비율",
     ["n_turn45", "n_turn90", "n_uturn", "large_turn_ratio", "uturn_ratio", "turn90_per_km"]),
    ("방향·회전(D)", "루프", "누적 회전으로 센 제자리 맴돌기 (가설 D)", ["loops"]),
    ("경로형태·직선투망(B)", "직진도·구불거림", "얼마나 곧게 가는가", ["straightness", "day_straightness", "seg_sinuosity_mean"]),
    ("경로형태·직선투망(B)", "경로 반복도", "좁은 범위 안을 얼마나 오가는가", ["path_over_diag", "path_over_rg"]),
    ("경로형태·직선투망(B)", "직선 투망 형상", "그물을 직선으로 펼치는 투망 패턴 (가설 B)",
     ["pca_elong", "ks_straight_max_m", "bbox_aspect"]),
    ("이동량·공간범위", "이동거리", "얼마나 많이 움직였나",
     ["dist_m", "dist_pts_m", "dist_raw_m", "dist_day_km", "dist_raw_day_km"]),
    ("이동량·공간범위", "변위·이격", "출발점에서 얼마나 멀어졌나",
     ["net_disp_m", "max_from_start_m", "near_zero_disp", "day_net_disp_km", "day_range_q95_km"]),
    ("이동량·공간범위", "활동 범위(1시간)", "1시간 동안 쓴 공간의 크기",
     ["bbox_w_m", "bbox_h_m", "bbox_diag_m", "bbox_area_km2", "rg_m", "rg_med_m", "max_radius_m", "hull_area_m2", "smooth_diag_m"]),
    ("이동량·공간범위", "활동 범위(하루)", "하루 동안 시간대 중심들이 퍼진 크기", ["day_rg_km", "day_bbox_diag_km"]),
    ("이동량·공간범위", "조업 해역 크기(선박)", "6개월간 조업 해역의 넓이·퍼짐 (위치 아님, 크기)",
     ["area_range_km", "fishing_area_radius"]),
    ("이동량·공간범위", "모항·항구 거리", "모항/항구에서 얼마나 멀리 나가나",
     ["home_km_max", "home_km_mean", "home_km_start", "home_km_end", "port_km_mean", "trip_max_dist_q50"]),
    ("시간 사용·항차", "상태별 시간 비율", "항구 정박/해상 정지/표류/조업속도/항해에 쓴 시간 비율 (민성 A)",
     ["port_share", "sea_stationary_share", "sea_stationary_share_sea", "drift_share", "drift_share_sea",
      "fishing_speed_share", "fishing_speed_share_sea", "transit_share", "transit_share_sea", "sea_stop_share",
      "fish_share", "in_port_ratio", "stationary_ratio", "moving_hour_share", "is_stationary"]),
    ("시간 사용·항차", "상태 전환", "상태가 얼마나 자주 바뀌나",
     ["n_transitions", "transitions_per_day", "n_stop_move_switch"]),
    ("시간 사용·항차", "항차(출항~귀항)", "출항 횟수, 항차 길이, 귀항 여부 (민성 B)",
     ["departures", "trips_per_day", "trip_hours_q50", "returned_ratio", "returned_to_port"]),
    ("시간 사용·항차", "출어 활동량", "얼마나 자주·오래 바다에 나가나",
     ["n_days", "n_months", "n_moving_days", "moving_days_ratio", "moving_days_per_month", "hours_per_day",
      "n_hours", "n_moving_hours", "moving_ratio"]),
    ("하루 리듬·시각(E)", "시작·종료 시각", "몇 시에 나가고 몇 시에 끝내나",
     ["first_hour", "last_hour", "mv_first_hour", "mv_last_hour", "departure_hour_q50"]),
    ("하루 리듬·시각(E)", "조업 시간 폭", "하루 중 활동한 시간의 길이", ["active_span_h", "mv_span_h"]),
    ("하루 리듬·시각(E)", "시간대 비중", "새벽/오전/오후/밤 중 언제 조업하나",
     ["mv_share_dawn", "mv_share_morning", "mv_share_afternoon", "mv_share_night", "night_ratio", "fishing_night_share"]),
    ("하루 리듬·시각(E)", "이동 블록 구조", "투망-휴식-양망처럼 하루가 몇 덩어리로 나뉘나 (가설 E)",
     ["active_block_count", "longest_streak", "max_block_gap_h"]),
    ("재방문·어장반복(A)", "하루 내 재방문", "같은 200m 칸을 떠났다가 다시 오나 (어구 회수, 가설 A)",
     ["return_after_absence", "revisit_cell_ratio", "n_cells_200m", "top3_cell_share"]),
    ("재방문·어장반복(A)", "날짜 간 어장 반복", "여러 날 같은 어장을 다시 찾나", ["cross_day_revisit", "site_jaccard_mean"]),
    ("체류점", "체류점", "반경 30m 안에 3점 이상 머문 지점 (민성, 지터 강건 정지 판정)", ["n_stays", "stay_frac"]),
    ("품질·기록량", "기록량", "GPS 점·구간이 얼마나 찍혔나",
     ["n_points_raw", "n_points", "n_segments", "gap_count", "pts_per_seg", "uniq_pos_ratio", "n_points_day",
      "n_segments_mean", "n_move_steps"]),
    ("품질·기록량", "GPS 튐·정제", "튐·점프로 제거된 양과 정제 전후 차이 (구성 C안 품질 피처)",
     ["n_spikes", "spike_ratio", "n_jumps", "n_bin_spikes", "n_bin_jumps", "bin_dispersion_m", "raw_clean_diff_m",
      "raw_clean_ratio", "overspeed_flag", "jump_share_day", "n_jumps_day"]),
    ("품질·기록량", "신호 끊김", "구간 사이 끊긴 거리", ["gap_dist_m", "max_gap_m"]),
    ("일 유형·이질성(G)", "출어일 유형", "하루 패턴 군집 비율과 다양성 (bag-of-days)",
     [f"daytype_{k}_share" for k in range(8)] + ["day_type_entropy"]),
    ("계절(H)", "월별 출어", "어느 달에 주로 나가나", [f"season_m{m}_share" for m in range(1, 7)] + ["month"]),
    ("위치(LOC)", "좌표", "어디서 활동하나 (shortcut 위험 → Pool A 제외)",
     ["lon_c", "lat_c", "lon_s", "lat_s", "lon_e", "lat_e", "lon_start", "lat_start", "lon_end", "lat_end",
      "mv_lon_c", "mv_lat_c", "home_lon", "home_lat", "fishing_lon", "fishing_lat", "home_port_km"]),
]
BASE_TO_TAX = {b: (cat, sub) for cat, sub, _, bases in TAXONOMY for b in bases}
STAT = {"mean": "평균", "median": "중앙값", "std": "표준편차(날짜 간 변동)", "p90": "90% 분위"}


def base_of(c):
    """선박 피처 이름 → (기본 피처, 파생 방식 설명, 계산 단위)"""
    if c in S3:
        return c, "선박 전체 기간에서 직접 계산", "L3 선박"
    m = re.fullmatch(r"fh_(.+)_q(50|90)", c)
    if m:
        return m.group(1), f"조업 시간대 1시간 값들의 선박별 {'중앙값' if m.group(2) == '50' else '90% 분위'}", "L1→L3"
    m = re.fullmatch(r"(.+)_(mean|median|std|p90)", c)
    if m:
        b, s = m.groups()
        if b in D:
            return b, f"출어일 값들의 {STAT[s]}", "L2→L3"
        if b.startswith("mv_") and b[3:] in H:
            return b[3:], f"출어일마다 이동 시간대 평균 → 출어일들의 {STAT[s]}", "L1→L2→L3"
    if c in D:
        return c, "하루 단위 값", "L2"
    if c in H:
        return c, "1시간 단위 값", "L1"
    raise KeyError(c)


def describe_base(b):
    for tbl in (S3, D, H):
        if b in tbl:
            return tbl[b]
    raise KeyError(b)
