"""
[EDA 1] 클래스 불균형 · 결측치/오염 · 월별 추이   (전체 데이터 집계 기반)

보는 것
  1-1 클래스 분포: 행(1시간) / 선박-일 / 선박 단위 각각
       → 행 기준과 선박 기준 비율이 다르면 "특정 클래스가 오래 조업" 신호
  1-2 한 선박이 여러 조업형태를 갖는지 (라벨 일관성 → GroupKFold 근거)
  1-3 결측/오염 요약 + 오염된 rfid_time 값 종류
  1-4 오염 rfid_time 행의 '진짜 시간' 추정 가능성 (같은 선박-일에서 빠진 시간대가 1개인가?)
  1-5 월별/일별 추이 (행 수, 활동 선박 수, 클래스 비율) → 데이터 공백·계절성
  1-6 행당 점 개수 / part 개수 분포 (가변 길이 문제 확인)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as C

C.set_korean_font()
pal = C.CLASS_COLORS

stats = pd.read_parquet(C.CACHE_DIR / "stats_dt_hour.parquet")
vday = pd.read_parquet(C.CACHE_DIR / "vessel_day.parquet")
feats = pd.read_parquet(C.CACHE_DIR / "sample_feats.parquet")
quality = json.loads((C.CACHE_DIR / "quality.json").read_text(encoding="utf-8"))
invalid = pd.read_parquet(C.CACHE_DIR / "invalid_rfid_time.parquet")

stats["date"] = pd.to_datetime(stats["part_dt"].astype(str), format="%Y%m%d")
stats["month"] = stats["date"].dt.month
vday["date"] = pd.to_datetime(vday["part_dt"].astype(str), format="%Y%m%d")
vday["month"] = vday["date"].dt.month

# ─────────────────────────────────────────────
# 1-1 클래스 분포 (3가지 단위)
# ─────────────────────────────────────────────
print("=" * 60, "\n1-1 클래스 분포")
by_row = stats.groupby("fishing_code")["n"].sum()
by_vday = vday.groupby("fishing_code").size()
vessel_cls = vday.groupby("rfid_id")["fishing_code"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
by_vessel = vessel_cls.value_counts()
dist = pd.DataFrame({"행(1시간)": by_row, "선박-일": by_vday, "선박": by_vessel}).reindex(C.CLASSES)
ratio = dist / dist.sum() * 100
print(dist.to_string(), "\n\n비율(%)\n", ratio.round(2).to_string())
print(f"\n최다/최소 클래스 비 (행 기준): {by_row.max() / by_row.min():.1f}배")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, dist.columns):
    ax.bar(dist.index, dist[col], color=[pal[c] for c in dist.index])
    for i, (v, r) in enumerate(zip(dist[col], ratio[col])):
        ax.text(i, v, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"클래스 분포 - {col} 단위")
    ax.tick_params(axis="x", rotation=15)
C.savefig("01_class_distribution", fig)

# ─────────────────────────────────────────────
# 1-2 선박별 라벨 일관성
# ─────────────────────────────────────────────
print("=" * 60, "\n1-2 선박별 라벨 일관성")
n_cls = vday.dropna(subset=["fishing_code"]).groupby("rfid_id")["fishing_code"].nunique()
print(n_cls.value_counts().rename("선박 수").to_string())
print(f"→ 2개 이상 조업형태를 가진 선박 비율: {(n_cls > 1).mean() * 100:.2f}%")
active_days = vday.groupby("rfid_id")["part_dt"].nunique()
print(f"선박당 활동일수: median={active_days.median():.0f}, mean={active_days.mean():.1f}, max={active_days.max()}")

fig, ax = plt.subplots(figsize=(8, 4))
for c in C.CLASSES:
    ids = vessel_cls[vessel_cls == c].index
    sns.kdeplot(active_days.reindex(ids), ax=ax, label=c, color=pal[c], clip=(0, 182))
ax.set_xlabel("6개월 중 활동일수"); ax.set_title("클래스별 선박 활동일수 분포"); ax.legend()
C.savefig("01_active_days_by_class", fig)

# ─────────────────────────────────────────────
# 1-3 결측/오염 요약
# ─────────────────────────────────────────────
print("=" * 60, "\n1-3 결측/오염 요약 (전체 행 기준)")
rows = quality["rows"]
qtab = pd.Series({k: v for k, v in quality.items() if k != "rows"})
print(pd.DataFrame({"count": qtab, "pct": qtab / rows * 100}).round(3).to_string())
print("\n오염 rfid_time 값 상위:\n", invalid.head(15).to_string(index=False))

inv = stats[stats["hour_k"] == -1].groupby(["month", "fishing_code"])["n"].sum().unstack()
tot = stats.groupby(["month", "fishing_code"])["n"].sum().unstack()
fig, ax = plt.subplots(figsize=(8, 4))
(inv / tot * 100).reindex(columns=C.CLASSES).plot(ax=ax, marker="o", color=[pal[c] for c in C.CLASSES])
ax.set_ylabel("오염 rfid_time 비율 (%)"); ax.set_title("월·클래스별 rfid_time 오염 비율")
C.savefig("01_invalid_hour_ratio", fig)

# ─────────────────────────────────────────────
# 1-4 오염 행의 시간 추정 가능성 (5% 샘플)
# ─────────────────────────────────────────────
print("=" * 60, "\n1-4 오염 rfid_time 행의 시간 복원 가능성 (샘플)")
bad_days = feats.loc[~feats["hour_valid"], ["rfid_id", "part_dt"]].drop_duplicates()
bad_sub = feats.merge(bad_days, on=["rfid_id", "part_dt"])
res = []
for (rid, dt), grp in bad_sub.groupby(["rfid_id", "part_dt"]):
    hs = grp["hour"].dropna().astype(int)
    n_bad = (~grp["hour_valid"]).sum()
    if len(hs) == 0:
        res.append((n_bad, -1)); continue
    missing_inside = len(set(range(hs.min(), hs.max() + 1)) - set(hs))
    res.append((n_bad, missing_inside))
r = pd.DataFrame(res, columns=["n_bad", "n_missing_hours_inside"])
r["case"] = np.select(
    [r.n_missing_hours_inside == -1, r.n_missing_hours_inside == r.n_bad, r.n_missing_hours_inside == 0],
    ["유효 시간 없음", "빈 시간 수 == 오염 행 수 (복원 후보)", "중간 빈칸 없음 (앞/뒤 끝 시간일 가능성)"],
    "애매")
print(r["case"].value_counts(normalize=True).mul(100).round(1).to_string())
# 중복 키 확인: 오염 행과 같은 (선박, 날짜)에 이미 같은 피처의 행이 있는지
dup_geom = feats.duplicated(["rfid_id", "part_dt", "n_points", "path_km"], keep=False) & ~feats["hour_valid"]
print(f"오염 행 중 같은 선박-일에 '동일 항적(점수·거리 동일)' 행이 존재하는 비율: "
      f"{dup_geom.sum() / max((~feats['hour_valid']).sum(), 1) * 100:.1f}%  (높으면 중복행 → 삭제 근거)")

# ─────────────────────────────────────────────
# 1-5 월별 / 일별 추이
# ─────────────────────────────────────────────
print("=" * 60, "\n1-5 월별 추이")
m_rows = stats.groupby(["month", "fishing_code"])["n"].sum().unstack().reindex(columns=C.CLASSES)
m_vessels = vday.groupby(["month", "fishing_code"])["rfid_id"].nunique().unstack().reindex(columns=C.CLASSES)
print("월별 행 수\n", m_rows.to_string(), "\n\n월별 활동 선박 수\n", m_vessels.to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
m_rows.plot(ax=axes[0], marker="o", color=[pal[c] for c in C.CLASSES]); axes[0].set_title("월별 행 수")
m_vessels.plot(ax=axes[1], marker="o", color=[pal[c] for c in C.CLASSES]); axes[1].set_title("월별 활동 선박 수")
(m_rows.div(m_rows.sum(1), axis=0) * 100).plot.bar(stacked=True, ax=axes[2], color=[pal[c] for c in C.CLASSES])
axes[2].set_title("월별 클래스 구성비 (%)"); axes[2].legend(fontsize=8)
C.savefig("01_monthly_trend", fig)

d_rows = stats.groupby(["date", "fishing_code"])["n"].sum().unstack().reindex(columns=C.CLASSES)
fig, ax = plt.subplots(figsize=(16, 4))
d_rows.plot(ax=ax, lw=1, color=[pal[c] for c in C.CLASSES])
ax.set_title("일별 행 수 (급감 구간 = 기상 악화/명절/수집 누락 의심)")
for d in pd.date_range(stats["date"].min(), stats["date"].max(), freq="W-SUN"):
    ax.axvline(d, color="gray", lw=0.3)
C.savefig("01_daily_trend", fig)

all_days = pd.date_range(stats["date"].min(), stats["date"].max())
missing_days = all_days.difference(stats["date"].unique())
print(f"\n데이터가 아예 없는 날짜: {len(missing_days)}일 {list(missing_days.strftime('%m-%d'))[:20]}")

# ─────────────────────────────────────────────
# 1-6 가변 길이 확인
# ─────────────────────────────────────────────
print("=" * 60, "\n1-6 행당 점 개수 / part 개수")
print(feats.groupby("fishing_code")[["n_points", "n_parts"]].describe().T.round(1).to_string())
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for c in C.CLASSES:
    s = feats.loc[feats.fishing_code == c]
    sns.kdeplot(np.log1p(s["n_points"]), ax=axes[0], label=c, color=pal[c])
    axes[1].hist(s["n_parts"].clip(upper=10), bins=np.arange(1, 12) - 0.5, alpha=0.4, density=True,
                 label=c, color=pal[c])
axes[0].set_xlabel("log(1 + 점 개수)"); axes[0].set_title("행(1시간)당 점 개수"); axes[0].legend()
axes[1].set_xlabel("part 개수 (10+는 10)"); axes[1].set_title("행당 구간(part) 수 = 신호 끊김 횟수+1")
C.savefig("01_points_per_row", fig)
print("\n✅ EDA 1 완료")
