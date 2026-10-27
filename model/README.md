# model/ — 피처 통합 → 피처 랭킹 → 조업형태 분류 (과제 1)

팀원 4명의 피처 명세(`진행2(피처).docx`, `진행1.docx` §6)를 하나의 파이프라인으로 재구현한 것.

## 실행 순서
```bash
pip install pandas numpy pyarrow scikit-learn lightgbm catboost optuna torch tabm rtdl_num_embeddings matplotlib
export SED_DATA_ROOT="/path/to/문제 데이터_수정_0928"   # 기본값: ../문제 데이터_수정_0928
export SED_N_JOBS=6                                      # 피처 추출 프로세스 수

python fe_hourly.py          # ① L1 시간 피처  → features/hourly_MM.parquet   (로컬 4코어 ≈ 50분, CPU 작업)
python fe_aggregate.py       # ② L2 선박-일 / L3 선박 집계 → features/shipday.parquet, ship_features.parquet (≈ 수 분, RAM ~6GB)
python rank_features.py      # ③ fold 내부 피처 랭킹 + k* 선정 → results/feature_ranking_*.csv, fold_rankings.json
python train_classifiers.py  # ④ 분류 실험 7종 → results/metrics.csv, oof_*.npy, cm_*.png
python mil.py                # ⑤ MIL (선박 = 출어일 묶음) → results/oof_*_mil.npy, mil_metrics.csv
python summarize_results.py  # ⑥ 전체 모델 통합표 → results/summary_table.csv, cm_{A,B}_best.png
python make_feature_catalog.py  # 팀 공유용 피처 목록 엑셀 → results/SeD_피처목록.xlsx
```
결과 요약과 해석은 [RESULTS.md](RESULTS.md)에 있습니다.
테스트용 출력 폴더 변경: `SED_FEAT_DIR`, `SED_RESULT_DIR` 환경변수.

## 파일
| 파일 | 내용 |
|---|---|
| `fe_config.py` | 경로·병렬 수·전처리 임계값 (HANDOFF §3.2 합의안) |
| `fe_hourly.py` | 1시간 항적 → 약 100개 피처 + 10분 평활 위치 12개. 나연 v2 정제 / 지유 P0·P1 / 민성 지터 강건(5-step·체류점) / 진우 스펙트럼 / 가설 B·C·D |
| `fe_aggregate.py` | 공용 항구 → 모항 → 시간대 상태(PORT/SEA_STOP/DRIFT/FISH/TRANSIT) → 선박-일 → 선박. 가설 A(재방문)·E(하루 리듬)·G(bag-of-days)·H(계절) |
| `cv_common.py` | 피처 풀(A=위치 제외/B=위치 포함), StratifiedGroupKFold(rfid_id), 피처 범주 메타 |
| `rank_features.py` | SHAP + 순열 중요도 + MI 통합 순위, Boruta shadow 검정, Spearman 중복 제거, k 스윕(1-SE rule) |
| `train_classifiers.py` | M0 LogReg(기준) / M1 RF / M2 LightGBM+Optuna(nested) / M3 CatBoost / M4 TabM(SOTA) / M5 가설 2단계 계층 모델 / M6 앙상블 (+ ablation, 옵션 TabPFN) |
| `mil.py` | 다중 인스턴스 학습: 출어일 묶음 + gated attention, 기간 길이별 성능, attention 해석 |
| `summarize_results.py` | 저장된 OOF 로 모델별 원본/클래스 보정 점수 통합표 |
| `feature_taxonomy.py` | 피처 분류표 (대분류 12 → 세부 구분 37 → 기본 피처) + 설명·출처 |
| `make_feature_catalog.py` | 팀 공유용 피처 카탈로그 엑셀 생성 |
| `run_server_tabm.sh` | GPU 서버용: TabM Pool A 실행 후 앙상블 재계산 |

## 서버(GPU 80GB)에서 돌릴 때
- ①②는 **CPU 작업**이라 GPU를 써도 빨라지지 않음. 코어가 많으면 `SED_N_JOBS`를 코어 수 −2 정도로, `SED_CHUNK=200000`.
- ④ `--device cuda` 를 주면 TabM·CatBoost 가 GPU 사용. `--trials 60` 등으로 튜닝을 늘려도 됨.
- TabPFN(옵션, `--tabpfn`): 가중치를 받으려면 Hugging Face 라이선스 동의와 로그인이 필요. 본인 계정으로 직접 설정한 뒤 사용.
