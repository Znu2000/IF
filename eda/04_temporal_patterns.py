"""
[EDA 4] 시간 패턴 분석

  4-1 시간대(0~23시)별 활동량 (전체 데이터)        → 클래스별 조업 시간대
  4-2 월 × 시간 히트맵 (클래스별)                   → 계절에 따른 출항 시간 변화
  4-3 요일별 활동                                   → 휴어일/주말 패턴
  4-4 출항(첫 시간)·귀항(마지막 시간)·조업시간 분포 (선박-일)
  4-5 시간대별 속도/선회 변화                        → 이동→조업→귀항 패턴
  4-6 선박별 '날짜 × 시간' 활동 래스터               → 하나의 선박이 시간 축으로 어떻게 이어지나
  4-7 자정 넘김(야간 연속 조업) 비율                  → 선박-일 단위로 자르면 안 되는 클래스?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from geo_utils import daily_features

C.set_korean_font()
pal = C.CLASS_COLORS

stats = pd.read_parquet(C.CACHE_DIR / "stats_dt_hour.parquet")
stats = stats[(stats.hour_k >= 0) & stats.fishing_code.notna()]
stats["date"] = pd.to_datetime(stats["part_dt"].astype(str), format="%Y%m%d")
stats["month"] = stats.date.dt.month
stats["dow"] = stats.date.dt.dayofweek
feats = pd.read_parquet(C.CACHE_DIR / "sample_feats.parquet").dropna(subset=["fishing_code", "hour"])
daily_path = C.CACHE_DIR / "sample_daily.parquet"
daily = pd.read_parquet(daily_path) if daily_path.exists() else daily_features(feats)

# ─────────────────────────────────────────────
# 4-1 시간대별 활동량
# ─────────────────────────────────────────────
hr = stats.groupby(["hour_k", "fishing_code"])["n"].sum().unstack()[C.CLASSES]
hr_n = hr / hr.sum() * 100
fig, ax = plt.subplots(figsize=(11, 4))
hr_n.plot(ax=ax, marker="o", color=[pal[c] for c in C.CLASSES])
ax.set_xticks(range(24)); ax.set_xlabel("rfid_time (시)"); ax.set_ylabel("클래스 내 비율 (%)")
ax.set_title("시간대별 활동 분포 (전체 데이터)")
C.savefig("04_hourly_activity", fig)

# ─────────────────────────────────────────────
# 4-2 월 × 시간 히트맵
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 4.5))
for ax, c in zip(axes, C.CLASSES):
    p = stats[stats.fishing_code == c].pivot_table(index="month", columns="hour_k", values="n", aggfunc="sum")
    sns.heatmap(p.div(p.sum(1), axis=0) * 100, ax=ax, cmap="magma", cbar=False)
    ax.set_title(c); ax.set_xlabel("시"); ax.set_ylabel("월")
fig.suptitle("월 × 시간대 활동 비율 (행 정규화)")
C.savefig("04_month_hour_heatmap", fig)

# ─────────────────────────────────────────────
# 4-3 요일별
# ─────────────────────────────────────────────
dow = stats.groupby(["dow", "fishing_code"])["n"].sum().unstack()[C.CLASSES]
n_days = stats.groupby("dow")["date"].nunique()
dow = dow.div(n_days, axis=0)
fig, ax = plt.subplots(figsize=(8, 4))
(dow / dow.mean() * 100).plot(ax=ax, marker="o", color=[pal[c] for c in C.CLASSES])
ax.set_xticks(range(7)); ax.set_xticklabels(list("월화수목금토일"))
ax.axhline(100, c="gray", lw=0.5); ax.set_ylabel("평균 대비 (%)"); ax.set_title("요일별 하루 평균 활동량")
C.savefig("04_dow_activity", fig)

# ─────────────────────────────────────────────
# 4-4 출항/귀항/조업시간
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for c in C.CLASSES:
    s = daily[daily.fishing_code == c]
    axes[0].hist(s.first_hour, bins=np.arange(25) - 0.5, histtype="step", density=True, lw=1.5, color=pal[c], label=c)
    axes[1].hist(s.last_hour, bins=np.arange(25) - 0.5, histtype="step", density=True, lw=1.5, color=pal[c])
    axes[2].hist(s.n_hours, bins=np.arange(1, 26) - 0.5, histtype="step", density=True, lw=1.5, color=pal[c])
axes[0].set_title("첫 활동 시각 (출항 추정)"); axes[1].set_title("마지막 활동 시각 (귀항 추정)")
axes[2].set_title("하루 활동 시간 수"); axes[0].legend()
C.savefig("04_departure_return", fig)

# ─────────────────────────────────────────────
# 4-5 시간대별 속도/선회
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for c in C.CLASSES:
    s = feats[feats.fishing_code == c].groupby("hour")
    axes[0].plot(s.speed_kn.median(), marker=".", color=pal[c], label=c)
    axes[1].plot(s.turn_abs_mean.median(), marker=".", color=pal[c])
    axes[2].plot(s.stop_ratio.mean(), marker=".", color=pal[c])
for ax, t in zip(axes, ["속도 중앙값 (kn)", "평균 선회각 중앙값 (°)", "정지 스텝 비율"]):
    ax.set_title(f"시간대별 {t}"); ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("시")
axes[0].legend()
C.savefig("04_hourly_behavior", fig)

# ─────────────────────────────────────────────
# 4-6 선박별 날짜 × 시간 래스터
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(20, 14))
for i, c in enumerate(C.CLASSES):
    vs = feats[feats.fishing_code == c].groupby("rfid_id").size()
    vs = vs[vs > vs.quantile(0.5)].sample(3, random_state=C.RANDOM_STATE).index
    for j, v in enumerate(vs):
        s = feats[feats.rfid_id == v]
        s = s.assign(date=pd.to_datetime(s.part_dt.astype(str), format="%Y%m%d"))
        ax = axes[i, j]
        sc = ax.scatter(s.date, s.hour, c=s.speed_kn.clip(upper=12), cmap="viridis", s=4, marker="s")
        ax.set_ylim(-0.5, 23.5); ax.set_title(f"{c} | {v} | {s.part_dt.nunique()}일", fontsize=9)
        ax.tick_params(labelsize=7)
fig.colorbar(sc, ax=axes, shrink=0.5, label="속도(kn)")
fig.suptitle("선박별 활동 래스터 (x=날짜, y=시간, 색=속도)", y=0.92)
C.savefig("04_vessel_raster", fig)

# ─────────────────────────────────────────────
# 4-7 자정 넘김 비율
# ─────────────────────────────────────────────
d = daily[["rfid_id", "date", "fishing_code", "first_hour", "last_hour"]].sort_values(["rfid_id", "date"])
d["next_date"] = d.groupby("rfid_id")["date"].shift(-1)
d["next_first"] = d.groupby("rfid_id")["first_hour"].shift(-1)
d["overnight"] = (d.last_hour == 23) & (d.next_date == d.date + pd.Timedelta(days=1)) & (d.next_first == 0)
on = d.groupby("fishing_code")["overnight"].mean().reindex(C.CLASSES) * 100
print("자정을 넘겨 다음날 0시까지 이어지는 선박-일 비율(%)\n", on.round(2).to_string())
print("→ 비율이 높은 클래스는 '하루' 대신 '연속 활동(trip)' 단위로 궤적을 잘라야 함")
print("\n✅ EDA 4 완료")
