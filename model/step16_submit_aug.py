"""
[STEP16-6] 잘린 항적 증강(aug7) LightGBM 으로 Test A 제출 파일 생성 ─ 리더보드 확인용
  학습: 2026 학습 선박 전체의 전체 항적 + 첫 기록 후 3·7·14·30·90·150일 항적 (배당 7행). 2025 항적은 학습에 쓰지 않음
  모델: LGB_BASE 클래스 가중 없음, 시드 0~3 평균. 3모델 앙상블이 아니고 후처리 가중도 없음 (cv 0.612~0.614 와 같은 조건)
  Test B: submission_1103v2_all_rp_no25.xlsx 시트를 그대로 사용 (빈칸). 기존 파일은 덮어쓰지 않음
  출력: results/submission_1105_aug7.xlsx, results/step16_test_aug7.npy
실행: python step16_submit_aug.py
"""
import time

import lightgbm as lgb
import numpy as np
import openpyxl
import pandas as pd

import fe_config as C
import step16_rt_class as RC
import train_classifiers as TC
from sklearn.metrics import accuracy_score
from step1_experiments import region_prior
from step4_final import attach

BASE = "submission_1103v2_all_rp_no25.xlsx"
OUT = "submission_1105_aug7.xlsx"
SEEDS = [0, 1, 2, 3]


def main():
    t0 = time.time()
    dst = C.RESULT_DIR / OUT
    assert not dst.exists(), f"{OUT} 가 이미 있음"
    S, y, cols, Sc = RC.load_all()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    attach(T, "_test")

    def design(X, self_in):
        X = X.reset_index(drop=True)
        return pd.concat([X.reindex(columns=cols), region_prior(S, y, X, self_in_ref=self_in).reset_index(drop=True)], axis=1)

    Xa = pd.concat([design(S, True)] + [design(Sc[n], True) for n in RC.CUTS], ignore_index=True)
    ya = np.tile(y, 1 + len(RC.CUTS))
    Xt = design(T, False)
    print(f"학습 {Xa.shape} / 테스트 {Xt.shape} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    P = np.zeros((len(T), len(C.CLASSES)))
    for sd in SEEDS:
        m = lgb.LGBMClassifier(**dict(TC.LGB_BASE, class_weight=None), random_state=sd, n_jobs=C.N_JOBS).fit(Xa, ya)
        P += m.predict_proba(Xt) / len(SEEDS)
        print(f"시드 {sd} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    np.save(C.RESULT_DIR / "step16_test_aug7.npy", P)

    # 클래스 가중이 도움이 되는지 aug OOF 로 확인만 함 (적용 안 함)
    oof = np.load(C.RESULT_DIR / "step16_oof_aug_full.npy")
    print(f"aug OOF: 가중 없음 {accuracy_score(y, oof.argmax(1)):.4f}", flush=True)

    pred = P.argmax(1)
    look = pd.DataFrame({"pred": np.array(C.CLASSES)[pred], "prob": P[np.arange(len(P)), pred]}, index=T["rfid_id"].astype(str))
    wb = openpyxl.load_workbook(C.RESULT_DIR / BASE)
    wa = wb.worksheets[0]
    old, new, miss = {}, {}, []
    for row in wa.iter_rows(min_row=2):
        rid = str(row[0].value).strip()
        old[rid] = row[1].value
        if rid in look.index:
            row[1].value, row[2].value = look.at[rid, "pred"], round(float(look.at[rid, "prob"]), 4)
        else:
            miss.append(rid)    # 기존 제출값 유지
        new[rid] = row[1].value
    wb.save(dst)
    print(f"{OUT}: {len(new)}척, 피처 없음 {len(miss)}척, {BASE} 대비 변경 {sum(old[k] != new[k] for k in new)}척", flush=True)
    print(pd.Series(new).value_counts().to_string(), flush=True)
    print("기존:", pd.Series(old).value_counts().to_dict(), flush=True)
    print(f"✅ 완료 {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
