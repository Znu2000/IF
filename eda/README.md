# EDA 코드 (KISTI 어선 항적 대회)

## 실행 방법
```bash
pip install -r requirements.txt
# 데이터 폴더가 다른 위치에 있으면 경로 지정 (기본값: ../문제 데이터_수정_0928)
export SED_DATA_ROOT=/path/to/문제\ 데이터_수정_0928      # Windows: set SED_DATA_ROOT=...
python run_all.py            # 00 → 09 순서대로 실행, 로그는 outputs/logs/*.txt
```
- 결과 그림/HTML/CSV → `outputs/`, 중간 캐시(parquet) → `cache/`
- 리눅스 서버에서 한글 폰트: `apt-get install fonts-nanum` 후 `rm -rf ~/.cache/matplotlib`
- **이전 데이터(`문제 데이터_수정`)로 만든 `cache/` 폴더는 지우고 시작하세요.**

## 단계 구성
| 파일 | 내용 | 입력 |
|---|---|---|
| `00_build_cache.py` | 원본 5.9GB 1회 순회 → 전체 집계 + 선박 5% 피처 + 1% 좌표 + 사고 후보 행 | 원본 CSV, 사고 xlsx |
| `01_overview.py` | 클래스 불균형(행/선박-일/선박), 결측·오염, 월별/일별 추이, 가변 길이 | 캐시 |
| `02_trajectory_by_class.py` | 조업형태별 밀도 지도, 하루 항적 샘플, 출발점 정렬 궤적 | 캐시 |
| `03_spatial_features.py` | 공간/이동/방향/주파수 피처 분포 비교, 속도 구간, 선박-일 피처 | 캐시 |
| `04_temporal_patterns.py` | 시간대·월·요일 패턴, 출항/귀항, 선박별 래스터, 자정 넘김 | 캐시 |
| `05_accident_analysis.py` | 사고 범주 분포, 유형 대분류, 시공간 매칭 민감도 | 캐시 + xlsx |
| `06_class_separability.py` | 피처 분리력(MI/KW/Fisher), PCA·t-SNE, GroupKFold 베이스라인 | 캐시 |
| `07_spatiotemporal.py` | 격자 우세 클래스·순도, 월별 어장 이동, Space-Time Cube | 캐시 |
| `08_trajectory_clustering.py` | 궤적 모양 KMeans, 행동 클러스터, 어장 HDBSCAN | 캐시 |
| `09_anomaly_3d.py` | 사고 매칭 선박 vs 정상 3D 비교, IsolationForest AUC | 캐시 + 05 결과 |

## 무거운 정도 / 조절
- **GPU는 필요 없음.** CPU + RAM 작업이라 코어 수와 RAM이 중요함.
- `00_build_cache.py`: 로컬(16GB, 8스레드)에서 청크당 약 15초 → 전체 **약 10~15분** 예상.
- 그 외 단계는 각각 수십 초~수 분 (06의 t-SNE·GroupKFold, 08의 KMeans/HDBSCAN이 제일 오래 걸림).
- 조절은 `config.py`에서:
  - `FEATURE_SAMPLE_PCT` (기본 5%) / `POINT_SAMPLE_PCT` (기본 1%) → 서버 RAM이 넉넉하면 올리기
  - `CHUNK_SIZE` → RAM 8GB면 200_000
- 동작만 먼저 확인할 때: `python 00_build_cache.py --max-chunks 1`

## 결과 공유할 때
`outputs/logs/*.txt` (콘솔 출력) + `outputs/*.png` 주요 그림 몇 개를 붙여주면 해석 가능.

## 데이터 메모
- 1행 = (선박, 날짜, 시간대) 1시간 항적, `geom` = MULTILINESTRING (괄호 하나 = 끊김 없는 구간)
- 점마다 시각이 없어서 속도는 "1행 = 1시간" 가정으로 근사 (`speed_kn = 이동거리/1h`)
- 0928 수정본에서 1월 rfid_time 오류값, 4월 중복행이 고쳐졌다고 함 → 01의 품질 리포트로 실제로 확인
- 6월 CSV에 geom이 줄바꿈으로 깨진 행이 있었음 → `broken_rows_dropped`로 개수 집계 후 제외
