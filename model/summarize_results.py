"""
[SUM] 저장된 OOF(results/oof_{pool}_{model}.npy) → 모델별 원본 / macro-F1 보정 점수 통합표 + 혼동행렬

보정(+adj): 클래스별 확률 가중치를 "다른 fold 의 OOF 와 정답"으로만 맞춰 해당 fold 에 적용 (fold 정답 미사용)
실행: python summarize_results.py
"""
import json

import numpy as np
import pandas as pd

import cv_common as V
import fe_config as C
from train_classifiers import ENS_MEMBERS, adjust_lofo, metrics_row, plot_cm

ORDER = ["base", "rf", "lgbm_base", "lgbm_tuned", "catboost", "tabm", "hier_hyp", "tabpfn"]
LABEL = {"base": "M0 LogReg (기준)", "rf": "M1 RandomForest", "lgbm_base": "(ablation) LightGBM 기본값",
         "lgbm_tuned": "M2 LightGBM+Optuna", "catboost": "M3 CatBoost", "tabm": "M4 TabM (SOTA)",
         "hier_hyp": "M5 가설 2단계 계층", "tabpfn": "(옵션) TabPFN", "ensemble": "M6 앙상블"}


def main():
    V.set_font()
    S, y = V.load_ship()
    R = json.loads((C.RESULT_DIR / "fold_rankings.json").read_text(encoding="utf-8"))
    fold_id = np.empty(len(y), int)
    for f, v in enumerate(R["fold_val_idx"]):
        fold_id[v] = f
    rows = []
    for pool in ["A", "B"]:
        OOF = {m: np.load(p) for m in ORDER if (p := C.RESULT_DIR / f"oof_{pool}_{m}.npy").exists()}
        mem = [m for m in ENS_MEMBERS if m in OOF]
        OOF["ensemble"] = np.mean([OOF[m] for m in mem], axis=0)
        for m, P in OOF.items():
            for adj in [False, True]:
                Q = adjust_lofo(P, y, fold_id) if adj else P
                r = metrics_row(m, pool, y, Q, fold_id)
                r.update(model_label=LABEL[m], adjusted=adj, k=R[pool]["k_star"],
                         members="+".join(mem) if m == "ensemble" else "")
                rows.append(r)
        best = max([r for r in rows if r["pool"] == pool], key=lambda r: r["macro_f1"])
        P = OOF[best["model"]]
        plot_cm(y, adjust_lofo(P, y, fold_id) if best["adjusted"] else P,
                f"Pool {pool} 최고: {best['model']}{'+adj' if best['adjusted'] else ''}  F1={best['macro_f1']:.3f}",
                C.RESULT_DIR / f"cm_{pool}_best.png")
    T = pd.DataFrame(rows)
    T.round(4).to_csv(C.RESULT_DIR / "summary_table.csv", index=False, encoding="utf-8-sig")
    show = ["pool", "model_label", "adjusted", "macro_f1", "f1_fold_std"] + [f"f1_{c}" for c in C.CLASSES] + ["gill_vs_coast_f1"]
    print(T[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
