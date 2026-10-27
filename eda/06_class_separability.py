"""
[EDA 6] 클래스 분리도 ─ "지금 피처로 4개 조업형태가 얼마나 갈리나?"

  6-1 피처별 분리력 순위: Kruskal-Wallis H, Mutual Information, Fisher ratio
  6-2 피처 상관 히트맵 (중복 피처 확인)
  6-3 PCA / t-SNE 2D 투영 (클래스 균형 샘플)
  6-4 간단 베이스라인 (HistGradientBoosting + GroupKFold(rfid_id))
       - 행(1시간) 단위 vs 선박-일 단위
       - 위치 피처 포함 vs 제외   → 위치에만 의존하면 새 해역 선박에 약함
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import config as C
from geo_utils import daily_features

C.set_korean_font()
pal = C.CLASS_COLORS

feats = pd.read_parquet(C.CACHE_DIR / "sample_feats.parquet").dropna(subset=["fishing_code"])
daily_path = C.CACHE_DIR / "sample_daily.parquet"
daily = pd.read_parquet(daily_path) if daily_path.exists() else daily_features(feats.dropna(subset=["hour"]))

LOC = ["c_lon", "c_lat", "s_lon", "s_lat", "e_lon", "e_lat"]
ROW_F = ["n_points", "n_parts", "path_km", "speed_kn", "disp_km", "straightness", "rog_km", "bbox_area_km2",
         "step_mean_m", "step_std_m", "step_max_m", "stop_ratio", "gap_km", "turn_abs_mean", "turn_std",
         "sharp_turn_ratio", "heading_R", "spec_entropy", "lowfreq_ratio"]
feats["hour_f"] = feats["hour"].astype(float)
ROW_ALL = ROW_F + ["hour_f"] + LOC

DAY_F = ["n_hours", "duration_h", "first_hour", "last_hour", "total_path_km", "max_range_km",
         "mean_speed_kn", "max_speed_kn", "mean_turn", "mean_straight", "mean_stop"]
DAY_LOC = ["c_lon", "c_lat", "s_lon", "s_lat"]


def balanced(df, n_per, seed=C.RANDOM_STATE):
    return df.groupby("fishing_code", group_keys=False).apply(
        lambda g: g.sample(min(n_per, len(g)), random_state=seed))


# ─────────────────────────────────────────────
# 6-1 분리력 순위
# ─────────────────────────────────────────────
b = balanced(feats, 20_000)
X = b[ROW_ALL].astype(float)
X = X.fillna(X.median())
y = b["fishing_code"]
H = {f: kruskal(*[X.loc[y == c, f] for c in C.CLASSES]).statistic for f in ROW_ALL}
MI = dict(zip(ROW_ALL, mutual_info_classif(X, y, random_state=C.RANDOM_STATE)))
gm = X.mean()
fisher = {f: sum(((X.loc[y == c, f].mean() - gm[f]) ** 2) for c in C.CLASSES) /
             (sum(X.loc[y == c, f].var() for c in C.CLASSES) + 1e-12) for f in ROW_ALL}
rank = pd.DataFrame({"KruskalH": H, "MutualInfo": MI, "Fisher": fisher}).sort_values("MutualInfo", ascending=False)
rank.round(4).to_csv(C.OUT_DIR / "06_feature_rank.csv", encoding="utf-8-sig")
print("피처 분리력 순위 (MI 기준)\n", rank.round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, col in zip(axes, rank.columns):
    r = rank[col].sort_values()
    ax.barh(r.index, r.values, color=["#C44E52" if i in LOC else "#4C72B0" for i in r.index])
    ax.set_title(col)
fig.suptitle("피처별 클래스 분리력 (빨강=위치 피처)")
fig.tight_layout()
C.savefig("06_feature_rank", fig)

# ─────────────────────────────────────────────
# 6-2 상관
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(X[ROW_F].corr(method="spearman"), cmap="RdBu_r", center=0, annot=True, fmt=".1f",
            annot_kws={"size": 7}, ax=ax)
ax.set_title("행 피처 Spearman 상관 (|ρ|>0.9 쌍은 하나만 써도 됨)")
C.savefig("06_feature_corr", fig)

# ─────────────────────────────────────────────
# 6-3 PCA / t-SNE
# ─────────────────────────────────────────────
def project(df, cols, title, name, n_per=2500):
    s = balanced(df, n_per)
    Z = s[cols].astype(float)
    Z = StandardScaler().fit_transform(Z.fillna(Z.median()).clip(Z.quantile(0.01), Z.quantile(0.99), axis=1))
    pca = PCA(2).fit(Z)
    P = pca.transform(Z)
    T = TSNE(2, perplexity=40, init="pca", random_state=C.RANDOM_STATE).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for c in C.CLASSES:
        m = (s.fishing_code == c).to_numpy()
        axes[0].scatter(P[m, 0], P[m, 1], s=3, alpha=0.4, color=pal[c], label=c)
        axes[1].scatter(T[m, 0], T[m, 1], s=3, alpha=0.4, color=pal[c], label=c)
    axes[0].set_title(f"PCA (설명분산 {pca.explained_variance_ratio_.sum() * 100:.1f}%)")
    axes[1].set_title("t-SNE"); axes[1].legend(markerscale=6)
    fig.suptitle(title)
    C.savefig(name, fig)


project(feats, ROW_F, "행(1시간) 피처 – 위치 제외", "06_proj_row_noloc")
project(daily, DAY_F, "선박-일 피처 – 위치 제외", "06_proj_day_noloc")
project(daily, DAY_F + DAY_LOC, "선박-일 피처 – 위치 포함", "06_proj_day_loc")

# ─────────────────────────────────────────────
# 6-4 베이스라인
# ─────────────────────────────────────────────
def cv_score(df, cols, name, n_max=300_000):
    df = df.dropna(subset=["fishing_code"])
    if len(df) > n_max:
        df = df.sample(n_max, random_state=C.RANDOM_STATE)
    Xc, yc, g = df[cols].astype(float).to_numpy(), df["fishing_code"].to_numpy(), df["rfid_id"].to_numpy()
    preds = np.empty(len(df), dtype=object)
    for tr, te in GroupKFold(5).split(Xc, yc, g):  # 같은 선박이 train/test 동시에 등장하지 않음
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                             random_state=C.RANDOM_STATE)
        clf.fit(Xc[tr], yc[tr])
        preds[te] = clf.predict(Xc[te])
    f1 = f1_score(yc, preds, average="macro")
    print(f"  {name:<28s} macro-F1 = {f1:.4f}")
    return f1, yc, preds


print("\n베이스라인 (GroupKFold by rfid_id, 5-fold)")
results = {}
results["행 | 위치X"] = cv_score(feats, ROW_F + ["hour_f"], "행 | 위치 제외")
results["행 | 위치O"] = cv_score(feats, ROW_ALL, "행 | 위치 포함")
results["일 | 위치X"] = cv_score(daily, DAY_F, "선박-일 | 위치 제외")
results["일 | 위치O"] = cv_score(daily, DAY_F + DAY_LOC, "선박-일 | 위치 포함")

fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
for ax, (k, (f1, yt, yp)) in zip(axes, results.items()):
    cm = confusion_matrix(yt, yp, labels=C.CLASSES, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=C.CLASSES).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"{k}  F1={f1:.3f}"); ax.tick_params(axis="x", rotation=20)
fig.suptitle("베이스라인 혼동행렬 (행 정규화)")
C.savefig("06_baseline_confusion", fig)
print("\n해석 가이드: '위치O'가 '위치X'보다 훨씬 높으면 위치 의존 ↑ → Test A(새 선박) 일반화 점검 필요")
print("✅ EDA 6 완료")
