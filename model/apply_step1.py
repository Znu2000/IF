"""
[STEP1 적용] 서버에서 돌린 step1_experiments.py 결과로 테스트A 예측을 교체
  입력 : results/step1_test_{config}.npy, results/step1_results.csv (클래스 가중 weights)
  출력 : results/test_pred_all.csv, results/submission.xlsx 의 테스트A 시트만 교체 (테스트B 이상탐지는 그대로)
         기존 제출 파일은 results/submission_before_{config}.xlsx 로 백업
실행: python apply_step1.py [--config prior_now_noshift]
"""
import argparse
import json
import shutil

import numpy as np
import openpyxl
import pandas as pd

import cv_common as V
import fe_config as C


def main(a):
    S, y = V.load_ship()
    T = pd.read_parquet(C.FEAT_DIR / "test_ship_features.parquet")
    P = np.load(C.RESULT_DIR / f"step1_test_{a.config}.npy")
    assert len(P) == len(T), "테스트 행 수 불일치"
    R = pd.read_csv(C.RESULT_DIR / "step1_results.csv").set_index("config")
    w = np.array(json.loads(R.at[a.config, "weights"]))
    print(f"[{a.config}] 클래스 가중 {dict(zip(C.CLASSES, w.round(3)))}  "
          f"(cv {R.at[a.config, 'cv_adj']:.4f}, cross_seen {R.at[a.config, 'cross_seen_adj']:.4f})")

    pred = (P * w).argmax(1)
    out = pd.DataFrame({"rfid_id": T["rfid_id"].astype(str),
                        "pred": np.array(C.CLASSES)[pred],
                        "prob": P[np.arange(len(P)), pred]})
    for i, c in enumerate(C.CLASSES):
        out[f"p_{c}"] = P[:, i]
    out.round(4).to_csv(C.RESULT_DIR / "test_pred_all.csv", index=False, encoding="utf-8-sig")

    dst = C.RESULT_DIR / "submission.xlsx"
    shutil.copy(dst, C.RESULT_DIR / f"submission_before_{a.config}.xlsx")
    wb = openpyxl.load_workbook(dst)
    wa = wb.worksheets[0]
    look = out.set_index("rfid_id")
    prior = pd.Series(y).map(dict(enumerate(C.CLASSES))).value_counts(normalize=True)
    fallback = prior.idxmax()
    old, miss = {}, []
    for row in wa.iter_rows(min_row=2):
        rid = str(row[0].value).strip()
        old[rid] = row[1].value
        if rid in look.index:
            row[1].value, row[2].value = look.at[rid, "pred"], round(float(look.at[rid, "prob"]), 4)
        else:  # 테스트 항적이 없거나 파싱 불가 → 학습 최빈 클래스
            row[1].value, row[2].value = fallback, round(float(prior[fallback]), 4)
            miss.append(rid)
    wb.save(dst)
    ids = list(old)
    sub = look.reindex(ids)
    changed = sum(old[i] != (sub.at[i, "pred"] if i not in miss else fallback) for i in ids)
    print(f"테스트A {len(ids)}척 → {dst.name}  (피처 없음 {len(miss)}척, 이전 제출 대비 변경 {changed}척)")
    print(sub["pred"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="prior_now_noshift")
    main(ap.parse_args())
