"""
[CMP] 피처 v1(기존 346) vs v2(+신규 67) 비교 ─ 같은 fold, 같은 모델
  핵심 앙상블 = lgbm_base + catboost + hier_hyp 확률 평균 (두 버전 모두 같은 구성)
  results/v1/ 의 OOF 와 results/ 의 OOF 를 비교 → results/compare_v1_v2.csv
실행: python compare_versions.py
"""
import json

import numpy as np
import pandas as pd

import cv_common as V
import fe_config as C
import feature_taxonomy as T
from train_classifiers import adjust_lofo, metrics_row

MODELS = ["base", "rf", "lgbm_base", "catboost", "hier_hyp"]
CORE = ["lgbm_base", "catboost", "hier_hyp"]


def is_v2(f):
    """분류표에서 세부 구분 이름에 (v2) 가 붙은 기본 피처에서 나온 피처"""
    return "(v2)" in T.BASE_TO_TAX[T.base_of(f)[0]][1]


def main():
    S, y = V.load_ship()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(R["fold_val_idx"]):
        fold_id[v] = f
    v1_val = json.loads((C.RESULT_DIR / "v1" / "fold_rankings.json").read_text(encoding="utf-8"))["fold_val_idx"]
    assert v1_val == R["fold_val_idx"], "v1 과 v2 의 fold 가 다름 → 비교 불가"

    rows = []
    for ver, d in [("v1", C.RESULT_DIR / "v1"), ("v2", C.RESULT_DIR)]:
        for pool in ["A", "B"]:
            oof = {m: np.load(p) for m in MODELS if (p := d / f"oof_{pool}_{m}.npy").exists()}
            if all(m in oof for m in CORE):
                oof["core_ensemble"] = np.mean([oof[m] for m in CORE], axis=0)
            for m, P in oof.items():
                for adj in (False, True):
                    Q = adjust_lofo(P, y, fold_id) if adj else P
                    rows.append(metrics_row(m, pool, y, Q, fold_id) | {"version": ver, "adjusted": adj})
    T = pd.DataFrame(rows)
    T.round(4).to_csv(C.RESULT_DIR / "compare_v1_v2.csv", index=False, encoding="utf-8-sig")
    piv = T.pivot_table(index=["pool", "model", "adjusted"], columns="version", values="macro_f1")
    piv["diff"] = piv["v2"] - piv["v1"]
    print(piv.round(4).to_string())
    tong = T.pivot_table(index=["pool", "model", "adjusted"], columns="version", values="f1_통발")
    print("\n통발 F1\n", (tong.assign(diff=tong["v2"] - tong["v1"])).round(4).to_string())

    # 신규 피처가 랭킹 어디에 들었나
    for pool in ["A", "B"]:
        r = pd.read_csv(C.RESULT_DIR / f"feature_ranking_{pool}.csv")
        new = r[r["feature"].map(is_v2)]
        k = R[pool]["k_star"]
        print(f"\nPool {pool} (k*={k}): 신규 피처 {len(new)}개 중 상위 {k} 안 {int((new['final_rank'] <= k).sum())}개")
        print(new.sort_values("final_rank").head(10)[["feature", "category", "final_rank", "topk_freq"]].to_string(index=False))


if __name__ == "__main__":
    main()
