# HANDOFF — KISTI 2026 DATA·AI 경진대회 (어선 항적 분석) 인수인계

> 다른 환경의 AI 에이전트가 이 문서만 읽고 **바로 다음 단계(피처 랭킹 → 분류/이상탐지 모델 실험)**를 이어가도록 작성.
> 작성 시점: 2026-10-25 / 팀명: SeD (지유, 민성, 나연, 진우)
> 대회 일정: 2026-08-11 ~ 11-25 (시상·발표 11/25). 현재 6주차 → 모델 실험 단계.

---

## 0. TL;DR (먼저 읽기)

| 항목 | 내용 |
|---|---|
| 과제 1 | 항적만으로 선박 **조업형태 4종 분류** (연안복합/자망/통발/채낚기연승낚시). Test A = **Train에 없는 신규 선박** |
| 과제 2 | 시간대별 **이상도(Anomaly Score)** 산출. Test B = 정상+사고 구간 혼합, 실제 사고시각 대비 정확·조기 탐지 평가 |
| 현재 상태 | EDA·전처리 설계·**전체 피처맵(후보 풀) 완료**. 팀원 각자 피처 뽑은 상태 |
| **다음 할 일** | 조원 전원의 피처맵을 합쳐 **피처 랭킹 → 상위권 피처로 분류 모델 + 이상탐지 모델 실험 시작** |
| 핵심 난제 | **자망 ↔ 연안복합** 분리 (거의 겹침), 위치 피처 shortcut 위험, GPS 지터 |
| 현 베이스라인 | 선박 단위 macro-F1 **0.528** (LightGBM 기본값, 튜닝 X, 위치 피처 기여가 압도적) |
| 필수 검증 | **rfid_id 기준 StratifiedGroupKFold(5)** — 시간/행 단위 랜덤 분할 금지 |

---

## 1. 데이터

### 1.1 위치 / 구조
- 워크스페이스 루트: `c:\Users\jinwo\SeD_DATA_AI` (다른 환경에서는 `SED_DATA_ROOT` 환경변수로 경로 지정)
- **현행 원본 = `문제 데이터_수정_0928/`** (2026-09-28 수정본). `문제 데이터_수정/`은 구버전(사용 금지, 해당 데이터로 만든 cache도 폐기).
  - `vpass_26년(1월_6월) 데이터/vpass_1월_6월 데이터/vpass_{1..6}월 데이터.csv` (약 5.9GB)
  - `26년 1~6월 어선 사고현황(테스트 데이터).xlsx` (사고 167건)
  - `샘플데이터 (2).csv`(메타/샘플), `col.png`(컬럼/제출 형식 예시)
- 1행 = (선박 `rfid_id`, 날짜 `part_dt`, 시간대 `rfid_time`) **1시간 항적**. `geom` = WKT `MULTILINESTRING` (괄호 하나 = 끊김 없는 구간, 좌표는 경도 lon → 위도 lat 순).
- 컬럼: `part_dt, rfid_id, rfid_time(0~23), fishing_code(라벨), geom` (+ n_points 등은 파생)
- **점마다 timestamp가 없음** → 실제 순간 속도/가속도 불가. 속도는 "구간 거리 ÷ 시간" **추정치**로만 사용 (1행=1시간, 점 균등 간격 가정).

### 1.2 규모 / 분포 (1~6월)
- 3,887,069행, **7,372척**. **선박당 조업형태 라벨 1개 고정** → 선박 단위 분할 필수.
- 클래스 비율(선박/행 기준 대략): 연안복합 43.3%, 자망 36.6%, 통발 15.5%, 채낚기 4.7% (최대 불균형 9.3배) → **Macro-F1** 필수.
- 채낚기는 일평균 이동거리 ~141km로 쉽게 분리(타 업종의 8~9배). **자망↔연안복합은 1시간 이동거리(65~79m)·변위(11~12m)가 거의 동일**.

### 1.3 알려진 데이터 품질 이슈
- 1월 `rfid_time`에 숫자 대신 '연안복합' 등 문자열 오염(전체 ~2.6%) → 0928 수정본에서 해결 확인. (나연: '연안복합'→1로 복원 처리)
- 4월 중복행 → 수정본에서 해결.
- **6월 CSV 248230행 부근에 geom이 줄바꿈으로 깨진 행** → `part_dt`가 8자리 숫자가 아니면 NaN 처리 후 `broken_rows_dropped`로 집계·제외.
- 수치 이상(예: 1시간 이동 300km 등 물리적으로 불가능한 값) → 점검 완료, 아래 §3 전처리 임계값 적용.
- **GPS 튐/지터**: A→B→A 왕복 튐, 정박 중 5~10m 지터. 1분 step 회전각이 지터 때문에 1.4°→48°로 폭발.

### 1.4 사고 데이터
- 1~6월 어선 사고 **167건** (기관·추진계 고장 84건=50% — 바다 위 정지·표류로 항적에 드러날 가능성).
- 항적 매칭: 사고 지점 반경 **3km, 사고시각 ±1시간** 기준으로 **69건 매칭**(`cache/acc_matches.parquet`). 민감도 분석은 `outputs/05_match_sensitivity.png`.
- 사고 시점에 `step_max_m` 등이 튀는 이상 징후 확인. **사고 수 시간 전 예측은 거의 불가** → 목표를 "사고 발생 즉시/직후 패턴 붕괴 감지(Detection)"로 설정.

---

## 2. 레포 / 코드 구조

```
SeD_DATA_AI/
├─ HANDOFF.md                  ← 이 문서
├─ readme.md                   대회 개요 원문
├─ eda/                        EDA 코드 + 캐시 (아래 §2.1)
│   ├─ config.py               경로/샘플링/클래스/폰트 설정 (SED_DATA_ROOT)
│   ├─ geo_utils.py            WKT 파싱, haversine, 행 단위 피처(FEATURE_COLS), clean_chunk, hash_pct
│   ├─ 00_build_cache.py ~ 09_anomaly_3d.py,  run_all.py,  requirements.txt
│   ├─ cache/*.parquet         중간 산출물
│   └─ outputs/                그림/로그
├─ report/                     진행상황 PPT 생성 코드 (pptkit.py, slides_*.py, build_deck.py)
├─ SeD_Project_Progress_Report.pptx / _updated.pptx   중간보고 PPT (35슬라이드)
├─ 진행1.docx, 진행2(피처).docx                        팀 문서 (데이터 구조·EDA / 팀원별 피처)
└─ 문제 데이터_수정_0928/ , 문제 데이터_수정/(구버전)
```

### 2.1 EDA 파이프라인 (`eda/`, `python run_all.py`)
| 파일 | 역할 |
|---|---|
| `00_build_cache.py` | 원본 5.9GB 청크(50만 행) 순회 → 전체 집계 + **선박 5% 해시 샘플 피처** + 1% 좌표 + 사고 후보 행. 로컬 약 10~15분 |
| `01_overview` | 클래스 불균형, 결측·오염, 월/일 추이 |
| `02_trajectory_by_class` | 클래스별 밀도 지도, 항적 샘플 |
| `03_spatial_features` / `04_temporal_patterns` | 피처 분포, 시간대·요일 패턴 |
| `05_accident_analysis` | 사고 범주, 시공간 매칭 민감도 |
| `06_class_separability` | MI / Kruskal-Wallis / Fisher 분리력, PCA·t-SNE, GroupKFold 베이스라인 |
| `07_spatiotemporal` / `08_trajectory_clustering` | 격자 순도, HDBSCAN 어장 143개, 궤적 KMeans |
| `09_anomaly_3d` | 사고 vs 정상 3D, IsolationForest AUC |

### 2.2 캐시 (`eda/cache/`)
- `sample_feats.parquet` — 5% 선박 샘플의 1시간 행 피처 (키: part_dt, rfid_id, hour, fishing_code; 피처 = `geo_utils.FEATURE_COLS`: n_parts, n_points, c/s/e lon·lat, bbox_w/h/area, rog_km, path_km, disp_km, straightness, speed_kn, step_mean/std/max_m, stop_ratio, gap_km, turn_abs_mean, turn_std, sharp_turn_ratio, heading_R, spec_entropy, lowfreq_ratio)
- `sample_daily.parquet` — 선박-일 요약
- `vessel_day.parquet` — 전체 선박-일 행 수
- `anomaly_eval_set.parquet` — 사고 매칭 선박의 사고 ±N시간 + 대조군 (**과제 2 평가용 소형 세트**, `rel_h`=사고 대비 상대시간, `acc_id`, `acc_type`, `group` 포함)
- `acc_matches*.parquet`, `acc_candidates.parquet` — 사고 매칭 결과/후보
- `sample_points.parquet` — 1% 선박 좌표, `shape_clusters.parquet`, `stats_dt_hour.parquet`, `invalid_rfid_time.parquet`, `quality.json`

> ⚠️ **cache는 0928 데이터로 재생성한 것**. 다른 환경에서 원본이 다르면 `eda/cache`를 지우고 `00_build_cache.py` 재실행.

### 2.3 ⚠️ 이 워크스페이스에 **없는** 것 (중요)
조원 나연의 v2 피처 파이프라인 산출물(`eda_output/scripts/build_ports.py → features_hourly.py → features_aggregate.py → validate_features.py`, `hourly_MM_pN.parquet`, `shipday_MM.parquet`, `ship_features.parquet`, `port_cells.csv`)과 민성의 `PrepConfig/FeatConfig/EvalConfig` 코드, 지유/진우의 피처 코드는 **이 폴더에 포함돼 있지 않음**(각자 로컬). 다음 환경에서는 사용자가 제공한 위치를 확인하거나, `진행2(피처).docx` 정의(§4)를 보고 재구현해야 한다. **먼저 사용자에게 피처 파일 경로를 확인할 것.**

---

## 3. 전처리 체계 (합의된 설계)

### 3.1 공통 원칙
- 라벨(`fishing_code`)로 전처리 규칙을 바꾸지 않음 (Test A에는 라벨이 없어 재현 불가).
- `rfid_id`는 **그룹 분할에만** 사용, 모델 입력에서 **제외**.
- 피처 선정·임계값 추정은 **fold 안에서만** (선정 누수 방지).
- 시각 손상 행(0~23 밖, 동일 시각인데 궤적 다름)은 시간·하루 피처에서 제외.

### 3.2 GPS 튐/지터 처리 (문헌: Zhang 2025 JMSE, Zheng 2015/2008 등)
| 항목 | 규칙 |
|---|---|
| 튐 제거 | A→B→A 왕복 B 제거 (100m 초과 & 왕복거리 25% 미만, 2회 반복) |
| 평활 경로 | 1시간을 6구간(~10분)으로 나눠 구간 중앙값 위치로 재구성 |
| 불가능 이동 | 점 간 >15km, 구간 간 >7km(10분에 ~22kn) 거리에서 제외 |
| 정박 | 평활 경로 범위 <100m → 이동·방향 피처 0/빈값 |
| 항구 | 6개월 정지 행을 500m 격자로 모아 ≥10척 정박한 **501칸** (`port_cells.csv`) |
| 속도 | 10분 구간 거리/10분 (추정치) |
| **지터 강건화(민성 시뮬레이션)** | 1분 step 대신 **5분 k-step**, 최댓값 대신 **중앙값/분위수**, 정지는 step 임계값 대신 **체류점(반경 3σ 내 ≥3분)**, sinuosity·hull·day_range(최댓값 기반)는 제거 또는 95% 분위수 |
| 정박 2단계 | 모항(정지 최다 1km 격자) 2km 내 정지=**항구 정박(제거)**, 그 외=**해상 정지(비율을 피처로)** |

### 3.3 3대 쟁점 및 팀 결정
1. **Clean vs Raw+Clean**: 무조건 정제 시 이상치 정보 손실(CleanML) → **구성 C안 = Clean 피처 + 품질 피처(`spike_ratio`, `raw_clean_distance_diff` 등) 결합**을 기본안. 전처리 구성 A/B/C/D(Raw/Clean/C안/…) 성능 비교 계획.
2. **지터**: 위 3.2.
3. **피처 계산 단위**: **시간(L1) / 선박-일(L2) / 선박(L3) 계층형 Feature Store**. Test A가 선박 단위든 시간행 단위든 Left Join으로 즉시 대응.
4. **위치 피처**: 넣으면 F1 약 +0.10 상승하나 지역 shortcut 위험 → **Model A(위치 제외, 순수 행동)** vs **Model B(위치 포함)** 이원화.

---

## 4. 피처맵 (조원 전원 피처 풀)

> 전체 정의의 원문은 `진행2(피처).docx`. 아래는 요약. 중복은 표준 명칭으로 통합됨(`dist_clean`, `displacement`, `convex_hull`, `sinuosity_v2`, `turn_angle`, `sharp_turn`, `heading_R`, `stop_ratio` 등).

### 4.1 나연 — 3단위 피처 사전 v2 (총 236컬럼)
- **시간 단위** `hourly_MM_pN.parquet` (74컬럼): 키/기록량/품질(`bad_geom, n_spikes, spike_ratio, n_jumps, overspeed_flag`…)/위치(lon_c…)/범위·모양(`bbox_*, rg_m, smooth_diag_m`)/거리(`dist_m, net_disp_m, max_from_start_m`)/경로형태(`straightness, path_over_diag, path_over_rg`)/속도(`sp_mean/std/p50/p90/max`)/속도구간(`frac_stop/slow/mid/fast`)/전환(`n_stop_move_switch`)/방향(`heading_R`)/회전(`turn_mean/std, n_turn45/90, n_uturn, turn90_per_km, uturn_ratio`)/정박·항구(`is_stationary, in_port, port_km_*`)/시간(`hour_sin/cos, month, dow`)
- **선박-일 단위** `shipday_MM.parquet` (68컬럼, 39.9만 행): 활동량/거리/범위/항해(`port_km_max, returned_to_port, in_port_ratio`)/위치/조업시간(`mv_first_hour, mv_span_h, mv_share_dawn…night`)/**이동 시간대 행동 `mv_*` 20개**
- **선박 단위** `ship_features.parquet` (94컬럼, 7,372척): 활동/위치(`area_range_km`)/**출어일 통계 26피처×(mean/median/std)=78컬럼**/상위값(p90)
- 학습 제외 컬럼: `rfid_id, part_dt, fishing_code, month, dow, bad_geom`
- 검증: 선박 단위 macro-F1 **0.528** (연안복합 .663 / 자망 .491 / 통발 .311 / 채낚기 .648), v1과 거의 동일 → 정제는 분류보다 **거리·속도·회전 정확도(과제 2)** 에 의미. 중요도 상위: 위치 좌표, 항구 거리, `area_range_km`, `day_rg_km`.

### 4.2 지유 — 항적 피처 후보 (P0/P1/P2)
- P0: `num_points, num_linestrings, total_distance_raw/clean, displacement, mean/median/max_step_distance, bbox_area, convex_hull_area, segment_sinuosity_mean, path_efficiency, turn_angle_mean/std, large_turn_ratio, stationary_flag, movement_per_point, hour_sin/cos`
- P1: `max/mean_radius_from_centroid, bearing_std, spike_count/ratio, raw_clean_distance_diff, gap_count/gap_distance/max_gap_distance, is_night, is_weekend`
- P2(집계): 일 단위 mean/median/std/max, `stationary_ratio, active_hours, night_activity_ratio`, 선박 단위 장기 통계
- 원칙: raw·clean 둘 다 유지, 위치 모델 A/B 분리.

### 4.3 민성 — 4범주×2수준 (공간/방향/속도/시간 × 시간/하루) + 평가 프레임
- 시간 단위: `displacement_m, radius_gyration_m, hull_area_m2, centroid_x/y, sinuosity, heading_consistency, mean_turn_deg, large_turn_ratio, reversal_count, avg_speed_kn, moving_ratio, step_mean_m, step_cv, hour_sin/cos, is_night, month`
- 하루 단위: `day_range_m, revisit_ratio, day_speed_mean/std/max, inter_hour_jump_m, active_hours, first/last_hour, night_ratio, longest_streak, ratio_stop/transit/fishing, n_transitions`
- **최종 추천(약 30개, H4+V1+지터강건화)**: A.시간 사용구성(`port_share, sea_stationary_share, drift_share, fishing_speed_share, transit_share, transitions_per_day`), B.항차 구조(`trip_hours_q50, trip_max_dist_q50, trips_per_day, cross_day_revisit, fishing_area_radius`), C.조업시간대 움직임 분포(5분 k-step 속도·회전·방향일관성·체류점 횟수·최장직선·`r6_dist_ratio`의 중앙값/분위수), D.위치(조건부: `fishing_centroid_xy, port_xy`), E.시각(`fishing_night_share, departure_hour_q50`)
- 평가 결과 요약: 전체 vessel F1 0.515~0.525, centroid 제외 ~0.43, 노이즈 기준선 ~0.21. 순열 중요도: `centroid_xy` 0.208 ≫ `day_range_m` 0.034 > `ratio_stop`·`revisit_ratio` 0.012. 정박 시간대 36~46% 혼입, 하루 단위가 상위 → **정박 제거 + 선박 단위 집계**가 유력한 개선 방향.
- 개선 실험 설계(확정 아님): H0~H6, V1(선박 집계+GBDT), S1/S2(GRU/Transformer, PyTorch 미검증), ENS.
- 중복: `ratio_fishing↔ratio_stop` 0.95, `active_hours↔longest_streak` 0.96, `hull_area↔radius_gyration` 0.93.

### 4.4 진우 — 1시간 피처 26종 분리력 + 시공간
- 분리력(MI/Kruskal/Fisher): 위치 제외 시 `heading_R`, `rog_km`가 1·2위. `turn_std, straightness, speed_kn, step_*, stop_ratio, step_max_m, bbox_area_km2, path_km, n_points, hour_f, lowfreq_ratio` 등.
- 0.1° 격자 순도 0.90, HDBSCAN 어장 143개(`hotspot_id`) — 강력하나 해역 shortcut 위험.
- 사고 특화: `step_max_m`, `bbox_area_km2`, `path_km`.

### 4.5 자망 vs 연안복합 분리 특화 가설 (A~F 그룹, 미구현/부분 구현)
자망=고정어구 포설 후 회수 / 연안복합=배가 직접 조작·표류.
`revisit_cell_ratio`(지점 재방문율), `longest_straight_run`(직선 투망 길이), `slow_run_max_len`(저속 양망 연속 길이), `drift_reposition_cycle`(선회 사이클), 일 조업 리듬(투망-양망 2블록), `day_type_entropy`(Bag-of-Days).

---

## 5. 합의된 검증/평가 규칙
- **분할**: `StratifiedGroupKFold(5)` by `rfid_id`. 점수 = "처음 보는 배" 성능.
- **지표**: 분류 **Macro-F1** (+ 클래스별 F1, 선박 단위/시간 단위 둘 다), 이상탐지 **PR-AUC**(+ 조기탐지 지표: 사고 전후 `rel_h`별 점수).
- **선박 단위 vessel-F1**: 시간대별 예측확률을 선박별 평균 → 업종 1개 선택 후 macro-F1.
- **노이즈 기준선**: 무작위 `_noise_` 피처. 이보다 안 좋으면 신호 없음 (+0.03 이하는 사실상 노이즈).
- 표본 상한: 선박당 최대 300행(행 많은 배가 지배하지 않도록). `class_weight="balanced"`.
- SMOTE/ADASYN은 기대 작음(표본 단위가 배) — 쓰더라도 fold 안에서만.
- 선박 특이성 판정은 일 단위 피처에서 구조적으로 과대 → centroid(0.93)만 진짜 위험.

---

## 6. 🎯 다음 단계 (이어서 할 일)

**사용자 계획**: *조원 모두가 뽑은 피처맵에서 피처 랭킹을 매기고, 상위권 피처로 분류 모델과 이상탐지 모델 실험을 시작한다.*

### Step 0. 사전 확인
1. 사용자에게 **v2 피처 파일 위치**(`hourly_*.parquet`, `shipday_*.parquet`, `ship_features.parquet`) 및 지유/진우 피처 코드 위치 확인 (§2.3).
2. `eda/requirements.txt` 설치. 메모리 부족 시 `config.py`의 `CHUNK_SIZE`, `FEATURE_SAMPLE_PCT` 조정. GPU 불필요(CPU+RAM).

### Step 1. 통합 피처 테이블 + 피처 랭킹
- 3단위(시간/일/선박) 테이블을 키(`rfid_id`, `part_dt`, `rfid_time`)로 Left Join (선박→일→시간 맥락 전파).
- **중복 제거**: §4의 표준 명칭 매핑 적용, Spearman |ρ|≥0.9 클러스터에서 대표 1개.
- **랭킹 방법 (전부 fold 안에서 계산)**: ① 단변량 lift vs `_noise_` ② permutation importance ③ MI / Kruskal / Fisher ④ SHAP(LightGBM). 결과를 **Model A(위치 제외)** / **Model B(위치 포함)** 로 각각 랭킹.
- 위치 계열(`lon_c, lat_c, centroid, port_km_*, hotspot_id, area_range_km` 등)은 별도 플래그로 분리.
- 산출: `feature_ranking.csv` (피처, 단위, 범주, 위치여부, 각 지표, 최종 순위, 중복 대표 여부).

### Step 2. 분류 모델 (과제 1)
- 상위 N(예: 20/30/50) 피처로 LightGBM / CatBoost / HistGB 베이스라인. **N 스윕**으로 최적 크기 탐색.
- 비교 축: (a) 전처리 구성 A/B/C/D, (b) 위치 포함/제외, (c) 판단 단위: 선박 / 선박-일 / 시간(+선박 집계), (d) 정박 제거(H4) 유무.
- 자망↔연안복합 이진 분류기(2단계 파이프라인)와 §4.5 특화 피처 효과 검증.
- 각자 가설대로 뽑은 서브셋 → **개별 모델 → 확률평균/스태킹 앙상블**이 팀 전략(피처 완전 통일 안 함, 다양성 확보).
- 현 기준선 **선박 macro-F1 0.528 (위치 제외 ~0.43)** 을 넘는 것이 1차 목표. 통발(0.31)이 가장 약함.

### Step 3. 이상탐지 모델 (과제 2)
- 평가 세트: `eda/cache/anomaly_eval_set.parquet`(사고 ±N시간, 대조군) — 사고 69건 매칭 기준.
- 접근: 선박/동일 업종별 **평상시 분포 대비 편차** — IsolationForest, 선박별 z-score/Mahalanobis, rolling 피처(직전 3·6·12h 대비 차이), 필요 시 LSTM-AE 등 시계열 모델.
- 사고 특화 피처(`step_max_m`, `bbox_area_km2`, `path_km`, `spike_*`, 품질 피처)는 이상탐지에서 **raw(품질 포함)** 정보를 보존하는 구성 C안 적용.
- 평가: PR-AUC, 사고시각 전후 `rel_h`별 이상점수 곡선, "사고 후 몇 시간 내 탐지" 같은 조기성 지표. 사고가 극히 적으므로(69건) 과적합·변동성 주의, 반복 분할/부트스트랩 신뢰구간 권장.
- 목표 설정: 사전 예측보다 **즉시/직후 감지**. 설명력(어떤 피처가 왜 이상인가)도 평가 배점(우수성 50)에 중요.

### Step 4. 앙상블 및 제출 포맷
- 최종 Test A/B 추론 + 제출 파일 자동 변환(선박 단위 / 시간 단위 모두 대응).
- 대회 배점: 창의성 20 / 우수성 50 / 효과성 30.

### 일정(PPT 로드맵 기준)
| 주차 | 기간 | 내용 |
|---|---|---|
| 6주(현재) | 10/19~10/25 | 피처 선별 & 개별 모델 실험 (진행 중) |
| 7주 | 10/26~11/04 | 하이퍼파라미터 튜닝, 자망/연안복합 특화, 이상항적 Anomaly Score |
| 8주 | 11/05~11/12 | 다양성 기반 최종 앙상블 |
| 9주 | 11/13~11/20 | Test A/B 최종 추론, 보고서·발표자료 |
| 최종 | 11/25 | 발표평가·시상 |

---

## 7. 주의사항 / 함정 (Gotchas)
- **선박 단위 분할 필수**. 행/시간 단위 랜덤 분할은 누수로 점수가 부풀려짐.
- `rfid_id`를 피처로 넣지 말 것. 일 단위 피처는 같은 날 모든 시간행에 복사돼 구조적으로 선박 특이성이 높게 보임(과대 판정).
- 위치 피처는 F1을 크게 올리지만 "해역 shortcut" 위험 → 항상 A/B 두 모델로 보고.
- `hour`(rfid_time) 신뢰도 의심: `is_night`/`hour_sincos`가 노이즈보다 낮게 나옴. 시각 피처는 정상일 때만.
- sinuosity 기존 정의는 오류(1 미만·폭발·NaN 15.5%) → v2 `straightness`/gap-aware sinuosity 사용.
- 속도는 추정치. point timestamp가 없으므로 순간 가속도·보간 금지(데이터 지어내기).
- GRU/Transformer는 이전 환경에서 PyTorch 부재로 **실행 검증 못함**. 업종당 선박 수가 적어 과적합 쉬움 — V1(선박 집계 GBDT)보다 나을 때만 채택.
- 정박 시간대 36~46%가 업종 차이를 묻음 → 정박 제거/선박 집계 효과가 클 것으로 기대(가짜 데이터 실험: 0.78→0.85→0.96, **실데이터 검증 필요**).
- `eda/README.md`: 한글 폰트(Linux는 `fonts-nanum`), 캐시 재생성 주의.
- Windows PowerShell 환경에서 인코딩(cp949) 깨짐 주의 → 스크립트는 파일로 작성해 실행하거나 `PYTHONUTF8=1`.

---

## 8. 참고 산출물
- 중간보고 PPT: `SeD_Project_Progress_Report_updated.pptx` (35슬라이드, 생성 코드 `report/build_deck.py`). 피처 선별 및 개별 모델 실험이 "진행 중"으로 반영됨.
- 원문 문서: `진행1.docx`(데이터 구조·EDA·sinuosity 정의 문제·문헌 근거·자망 vs 연안복합 가설), `진행2(피처).docx`(팀원별 피처, v2 사전, 지터 시뮬레이션).
- EDA 그림: `eda/outputs/*.png` (01~09), 로그 `eda/outputs/logs/*.txt`.
- 참고문헌: Zheng 2015 (Trajectory Data Mining), Liu 2024 (AIS kinematic anomaly), Zhang 2025 (JMSE GPS outlier cleaning), Li 2021 (CleanML), Palmer & Wigley 2009, Zheng 2008, de Souza 2016, Xing 2023, Natale 2015, Vespe 2016.
