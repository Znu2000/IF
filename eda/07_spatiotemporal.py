"""
[EDA 7] 공간 시각화 + 시계열-공간 결합 EDA

  7-1 격자(0.1°)별 우세 조업형태 & 순도  → "위치만 보면 클래스가 정해지나?"
  7-2 월 × 클래스 밀도 small multiples   → 계절에 따른 어장 이동
  7-3 월간 어장 안정성 (격자 활동량의 월간 상관)
  7-4 시간대별 '출발점에서의 거리'        → 출항-조업-귀항 리듬 (공간 × 시간)
  7-5 Space-Time Cube 3D (lon, lat, 시간) – plotly HTML
  7-6 시간대별 활동 위치 애니메이션 – plotly HTML
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from geo_utils import haversine_km

C.set_korean_font()
pal = C.CLASS_COLORS

feats = pd.read_parquet(C.CACHE_DIR / "sample_feats.parquet").dropna(subset=["fishing_code", "c_lon"])
feats["date"] = pd.to_datetime(feats.part_dt.astype(str), format="%Y%m%d")
feats["month"] = feats.date.dt.month
EXT = (124, 131.5, 32.5, 38.7)
G = 0.1

# ─────────────────────────────────────────────
# 7-1 격자 우세 클래스 / 순도
# ─────────────────────────────────────────────
feats["gx"] = (feats.c_lon // G * G).round(2)
feats["gy"] = (feats.c_lat // G * G).round(2)
cell = feats.groupby(["gx", "gy", "fishing_code"]).size().unstack(fill_value=0).reindex(columns=C.CLASSES, fill_value=0)
cell = cell[cell.sum(1) >= 20]
share = cell.div(cell.sum(1), axis=0)
dom = share.idxmax(1)
purity = share.max(1)
print(f"격자 {len(cell):,}개 / 순도(최다 클래스 비율) 평균 {purity.mean():.3f}, 순도>0.9 격자 비율 {(purity > 0.9).mean() * 100:.1f}%")
w_purity = (purity * cell.sum(1)).sum() / cell.sum(1).sum()
print(f"행 가중 평균 순도: {w_purity:.3f}  → '격자만 보고 최다 클래스로 찍기' 정확도 상한(같은 선박 포함이라 낙관적)")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
idx = cell.index.to_frame(index=False)
cmap = {c: i for i, c in enumerate(C.CLASSES)}
from matplotlib.colors import ListedColormap
axes[0].scatter(idx.gx + G / 2, idx.gy + G / 2, c=dom.map(cmap), cmap=ListedColormap([pal[c] for c in C.CLASSES]),
                s=6, marker="s", alpha=purity.clip(0.3, 1).to_numpy())
for c in C.CLASSES:
    axes[0].scatter([], [], color=pal[c], label=c, marker="s")
axes[0].legend(); axes[0].set_title("0.1° 격자별 우세 조업형태 (투명도=순도)")
sc = axes[1].scatter(idx.gx + G / 2, idx.gy + G / 2, c=purity, cmap="RdYlGn", s=6, marker="s", vmin=0.25, vmax=1)
fig.colorbar(sc, ax=axes[1], label="순도"); axes[1].set_title("격자 순도 (빨강=여러 조업형태 혼재)")
for ax in axes:
    ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:]); ax.set_aspect(1.2)
C.savefig("07_grid_dominant_class", fig)

# ─────────────────────────────────────────────
# 7-2 월 × 클래스 밀도
# ─────────────────────────────────────────────
months = sorted(feats.month.unique())
fig, axes = plt.subplots(len(C.CLASSES), len(months), figsize=(4 * len(months), 4.2 * len(C.CLASSES)),
                         sharex=True, sharey=True)
for i, c in enumerate(C.CLASSES):
    for j, m in enumerate(months):
        s = feats[(feats.fishing_code == c) & (feats.month == m)]
        axes[i, j].hexbin(s.c_lon, s.c_lat, gridsize=90, bins="log", cmap="inferno", mincnt=1, extent=EXT)
        axes[i, j].set_aspect(1.2)
        if i == 0:
            axes[i, j].set_title(f"{m}월")
    axes[i, 0].set_ylabel(c, fontsize=13)
fig.suptitle("월 × 조업형태별 활동 밀도", y=0.91)
C.savefig("07_month_class_density", fig)

# ─────────────────────────────────────────────
# 7-3 월간 어장 안정성
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
for ax, c in zip(axes, C.CLASSES):
    p = feats[feats.fishing_code == c].pivot_table(index=["gx", "gy"], columns="month", values="n_points",
                                                   aggfunc="size", fill_value=0)
    sns.heatmap(np.log1p(p).corr(), annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis", ax=ax, cbar=False)
    ax.set_title(f"{c}: 월간 격자활동 상관")
fig.suptitle("어장 위치의 월간 안정성 (낮을수록 계절 이동 큼 → 학습/검증 월 분리 시 주의)")
C.savefig("07_monthly_stability", fig)

# ─────────────────────────────────────────────
# 7-4 시간대별 출발점 거리
# ─────────────────────────────────────────────
f = feats.dropna(subset=["hour"]).sort_values(["rfid_id", "part_dt", "hour"])
first = f.groupby(["rfid_id", "part_dt"])[["s_lon", "s_lat"]].transform("first")
f["dist_start_km"] = haversine_km(first.s_lon, first.s_lat, f.c_lon, f.c_lat)
f["rel_hour"] = f.hour - f.groupby(["rfid_id", "part_dt"])["hour"].transform("min")
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
for c in C.CLASSES:
    s = f[f.fishing_code == c]
    axes[0].plot(s.groupby("hour").dist_start_km.median(), marker=".", color=pal[c], label=c)
    axes[1].plot(s[s.rel_hour <= 20].groupby("rel_hour").dist_start_km.median(), marker=".", color=pal[c])
axes[0].set_title("시각별 '그날 첫 위치'로부터 거리 중앙값 (km)"); axes[0].set_xlabel("시")
axes[1].set_title("출항 후 경과시간별 거리 중앙값 (km)"); axes[1].set_xlabel("첫 활동 후 경과 시간")
axes[0].legend()
C.savefig("07_distance_from_start_by_hour", fig)

# ─────────────────────────────────────────────
# 7-5 / 7-6  plotly
# ─────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px

    pts = pd.read_parquet(C.CACHE_DIR / "sample_points.parquet").dropna(subset=["fishing_code", "hour"])
    pts["t"] = pd.to_datetime(pts.part_dt.astype(str), format="%Y%m%d") + pd.to_timedelta(pts.hour.astype(int), unit="h")
    fig = go.Figure()
    for c in C.CLASSES:
        vs = pts[pts.fishing_code == c].rfid_id.drop_duplicates()
        vs = vs.sample(min(3, len(vs)), random_state=C.RANDOM_STATE)
        for k, v in enumerate(vs):
            s = pts[pts.rfid_id == v].sort_values(["t", "seq"])
            s = s[s.t < s.t.min() + pd.Timedelta(days=7)]  # 첫 7일
            th = (s.t - s.t.min()).dt.total_seconds() / 3600 + s.seq / s.groupby(["part_dt", "hour"]).seq.transform("max").clip(lower=1)
            fig.add_trace(go.Scatter3d(x=s.lon, y=s.lat, z=th, mode="lines", line=dict(color=pal[c], width=3),
                                       name=c, legendgroup=c, showlegend=(k == 0), hovertext=str(v)))
    fig.update_layout(scene=dict(xaxis_title="경도", yaxis_title="위도", zaxis_title="경과 시간(h)"),
                      title="Space-Time Cube: 선박별 7일 궤적", height=850)
    fig.write_html(C.OUT_DIR / "07_space_time_cube.html")
    print("  [saved] 07_space_time_cube.html")

    s = feats.dropna(subset=["hour"]).sample(min(60_000, len(feats)), random_state=0).sort_values("hour")
    s["hour"] = s.hour.astype(int)
    fig = px.scatter(s, x="c_lon", y="c_lat", color="fishing_code", animation_frame="hour",
                     color_discrete_map=pal, opacity=0.4, range_x=EXT[:2], range_y=EXT[2:],
                     category_orders={"fishing_code": C.CLASSES}, height=850, width=800,
                     title="시간대별 활동 위치 (1시간 항적 중심점)")
    fig.update_traces(marker_size=3)
    fig.write_html(C.OUT_DIR / "07_hourly_animation.html")
    print("  [saved] 07_hourly_animation.html")
except ImportError:
    print("  plotly 미설치 → 7-5/7-6 생략 (pip install plotly)")

print("\n✅ EDA 7 완료")
