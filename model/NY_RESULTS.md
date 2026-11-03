# 나연 실험 결과 (2026-11-01)

규칙(팀장 지침): **모델 학습은 2026 항적만, 2025 테스트 항적은 검증에만.**
2025 검증 = 2026 학습 선박 중 2025 테스트셋에도 있는 6,751척(테스트A 100척 제외)의 2025 항적을 2026 라벨로 채점.
fold 는 팀과 같은 선박 단위 StratifiedGroupKFold(5). 지표는 리더보드와 같은 정확도(micro-F1).
검증 선박이 6,751척이라 우연만으로 ±0.7%p 정도 흔들리므로, 1%p 미만 차이는 개선으로 보지 않음(부트스트랩 95% 신뢰구간으로 판정).

## 한 줄 결론
규칙을 지키는 범위의 천장은 2025 검증 정확도 **약 0.60**. 피처 선택·범주 구성·지역 사전확률 방식을 바꿔도 넘지 못했고,
유일하게 점수를 올린 건 새 정보인 **지역 사전확률(진우 step1, +1.1~1.8%p)**.

## 결과 요약 (LightGBM, 2026 학습 → 2025 검증 정확도)
| 실험 | 파일 | 결과 |
|---|---|---|
| 피처 랭킹 조건 10가지 | `ny_rank_conditions.py` → `results/ny_rank_conditions.csv` | 피처 전체 413개(0.586)가 최고. 팀 기본 50개 0.570, 노이즈 검정 통과만 0.508 |
| 범주별 빼기·넣기 (13개 대분류) | `ny_category_ablation.py` → `results/ny_category_ablation.csv` | 위치만 필수(−3.1%p). 속도·방향·하루 리듬은 2025에서만 각 −0.8~1.0%p → 연도 일반화에 기여. 계절·품질은 빼도 그대로 |
| 지역 사전확률 + 피처 전체 | `ny_prior_allfeats.py` → `results/ny_prior_allfeats.csv` | 0.604 = 진우 step1 3모델 top-50 앙상블과 같은 수준 (LightGBM 1개로 충분) |
| 지역 사전확률 만드는 방식 11가지 | `ny_prior_variants.py` → `results/ny_prior_variants.csv` | 모두 k=30 대비 ±0.6%p 안. 항구 단위만 쓰면 −1.2%p. k=30 그대로 사용 권장 |
| 이상탐지 순위 학습 (2026 사고 매칭 77척) | `ny_anom_rank.py` → `results/ny_anom_rank_tol1.csv` | 배마다 1위 시각이 사고 ±3h 안인 비율 ≤ 4% (0~3척). 기존 방식과 차이 없음 |

## 보조 코드
- `ny_common.py`: 위 실험 공통 데이터 로드(2025 검증 표본 구성)와 LightGBM 설정
- `ny_agg_sharded.py`: `fe_aggregate.py` 메모리 절약판(선박을 나눠 집계, RAM 8GB). 항구 격자 875칸·피처 413개 그대로 재현

## 실행 순서
```bash
export SED_DATA_ROOT="/path/to/문제 데이터_수정_1027/vpass_26년(1월_6월) 데이터"
python fe_hourly.py && python ny_agg_sharded.py                        # 학습 피처
python fe_hourly.py --src "<테스트셋 폴더>" --prefix test_hourly && python ny_agg_sharded.py --test
python ny_category_ablation.py      # 약 25분 (CPU 2코어)
python ny_rank_conditions.py        # 약 25분 (중요도 캐시 results/ny_rank_cache.pkl 생성)
python ny_prior_allfeats.py         # 약 9분
python ny_prior_variants.py         # 약 21분
python anomaly_prep.py && python anomaly_labels.py && python ny_anom_rank.py
```
`ny_common.py` 는 제출샘플 경로로 `문제 데이터_수정_1027/테스트셋 및 결과제출 샘플/제출샘플.xlsx` (레포 최상위 기준)을 씁니다.
