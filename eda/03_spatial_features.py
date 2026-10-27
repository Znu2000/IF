"""
[EDA 3] 공간/이동 피처 분포 비교   (5% 선박 샘플, 행=1시간 / 일=선박-일)

  3-1 클래스별 피처 분포 (바이올린)   → 어떤 피처가 클래스마다 다르게 생겼나
  3-2 조업 위치(위도/경도 중심) 분포  → 해역 차이 (동해=채낚기? 남해=통발? 등)
  3-3 속도 구간(정박/조업/이동) 비율  → 어업 행동 패턴
  3-4 선박-일 단위 피처 (활동반경, 총 이동거리, 조업시간)
  3-5 클래스별 요약 통계표 CSV
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from geo_utils import daily_features

C.set_korean_font()
pal = C.CLASS_COLORS

feats = pd.read_parquet(C.CACHE_DIR / "sample_feats.parquet").dropna(subset=["fishing_code"])
print(f"행 {len(feats):,} / 선박 {feats.rfid_id.nunique():,}")

ROW_FEATS = ["n_points", "path_km", "speed_kn", "disp_km", "straightness", "rog_km", "bbox_area_km2",
             "step_mean_m", "stop_ratio", "turn_abs_mean", "sharp_turn_ratio", "heading_R",
             "spec_entropy", "lowfreq_ratio", "n_parts", "gap_km"]
LOG_FEATS = {"n_points", "path_km", "disp_km", "rog_km", "bbox_area_km2", "step_mean_m", "gap_km", "speed_kn"}

# ─────────────────────────────────────────────
# 3-1 바이올린
# ─────────────────────────────────────────────
sub = feats.sample(min(len(feats), 200_000), random_state=C.RANDOM_STATE)
ncol = 4
nrow = int(np.ceil(len(ROW_FEATS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
for ax, f in zip(axes.flat, ROW_FEATS):
    v = sub[f].clip(upper=sub[f].quantile(0.99))
    if f in LOG_FEATS:
        v = np.log1p(v)
    sns.violinplot(x=sub["fishing_code"], y=v, order=C.CLASSES, palette=pal, hue=sub["fishing_code"],
                   legend=False, ax=ax, cut=0, inner="quartile", linewidth=0.6)
    ax.set_title(("log1p " if f in LOG_FEATS else "") + f, fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(axis="x", labelsize=7, rotation=10)
for ax in axes.flat[len(ROW_FEATS):]:
    ax.axis("off")
fig.suptitle("클래스별 1시간 항적 피처 분포", y=1.0)
fig.tight_layout()
C.savefig("03_feature_violin", fig)

# ─────────────────────────────────────────────
# 3-2 조업 위치 분포
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for c in C.CLASSES:
    s = sub[sub.fishing_code == c]
    sns.kdeplot(s.c_lon, ax=axes[0], color=pal[c], label=c)
    sns.kdeplot(s.c_lat, ax=axes[1], color=pal[c], label=c)
axes[0].set_title("경도 분포"); axes[1].set_title("위도 분포"); axes[0].legend()
for c in C.CLASSES:
    s = sub[sub.fishing_code == c].sample(min(3000, (sub.fishing_code == c).sum()), random_state=0)
    axes[2].scatter(s.c_lon, s.c_lat, s=1, alpha=0.3, color=pal[c], label=c)
axes[2].set_title("1시간 항적 중심점 (클래스별 3천개)"); axes[2].set_aspect(1.2)
axes[2].legend(markerscale=10)
C.savefig("03_location_dist", fig)

# ─────────────────────────────────────────────
# 3-3 속도 구간 비율
# ─────────────────────────────────────────────
bins = [-0.01, 0.5, 2, 5, 10, np.inf]
labels = ["정박(<0.5kn)", "저속(0.5-2)", "조업추정(2-5)", "이동(5-10)", "고속(>10)"]
feats["speed_band"] = pd.cut(feats["speed_kn"], bins=bins, labels=labels)
band = pd.crosstab(feats.fishing_code, feats.speed_band, normalize="index").reindex(C.CLASSES) * 100
print("\n속도 구간 비율(%)\n", band.round(1).to_string())
fig, ax = plt.subplots(figsize=(10, 4))
band.plot.barh(stacked=True, ax=ax, colormap="RdYlBu_r")
ax.set_xlabel("%"); ax.set_title("클래스별 1시간 평균속도 구간 비율 (1행=1시간 가정)")
ax.legend(bbox_to_anchor=(1, 1))
C.savefig("03_speed_band", fig)

# ─────────────────────────────────────────────
# 3-4 선박-일 단위
# ─────────────────────────────────────────────
daily = daily_features(feats.dropna(subset=["hour"]))
daily.to_parquet(C.CACHE_DIR / "sample_daily.parquet", index=False)  # 이후 스크립트 재사용
DAY_FEATS = ["n_hours", "duration_h", "first_hour", "last_hour", "total_path_km",
             "max_range_km", "mean_speed_kn", "max_speed_kn", "mean_turn", "mean_stop"]
fig, axes = plt.subplots(2, 5, figsize=(24, 8))
for ax, f in zip(axes.flat, DAY_FEATS):
    v = daily[f].clip(upper=daily[f].quantile(0.99))
    sns.boxplot(x=daily.fishing_code, y=v, order=C.CLASSES, palette=pal, hue=daily.fishing_code,
                legend=False, ax=ax, showfliers=False)
    ax.set_title(f); ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=7, rotation=10)
fig.suptitle("클래스별 선박-일(하루 조업) 피처")
fig.tight_layout()
C.savefig("03_daily_features", fig)

# ─────────────────────────────────────────────
# 3-5 요약표
# ─────────────────────────────────────────────
summary = feats.groupby("fishing_code")[ROW_FEATS].median().T[C.CLASSES]
summary_d = daily.groupby("fishing_code")[DAY_FEATS].median().T[C.CLASSES]
out = pd.concat([summary.add_prefix(""), summary_d]).round(3)
out.to_csv(C.OUT_DIR / "03_feature_median_by_class.csv", encoding="utf-8-sig")
print("\n클래스별 중앙값\n", out.to_string())
print("\n✅ EDA 3 완료")
