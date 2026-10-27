"""
[EDA 2] 조업형태별 항적 시각화   (1% 선박 좌표 샘플)

  2-1 클래스별 전체 점 밀도 지도 (어디서 조업하나)
  2-2 클래스별 무작위 선박-일 궤적 (시간 색상)  → 하루 항적 '모양' 직관
  2-3 출발점 원점 정렬 스파게티 플롯             → 위치를 지우고 '움직임 패턴'만 비교
  2-4 (선택) plotly 인터랙티브 지도 HTML
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from geo_utils import to_local_km

C.set_korean_font()
pal = C.CLASS_COLORS
rng = np.random.default_rng(C.RANDOM_STATE)

pts = pd.read_parquet(C.CACHE_DIR / "sample_points.parquet")
pts = pts.dropna(subset=["fishing_code"])
pts["hour_f"] = pts["hour"].astype("float")
print(f"점 개수: {len(pts):,}  선박 수: {pts.rfid_id.nunique():,}")

LON_LIM = (pts.lon.quantile(0.001), pts.lon.quantile(0.999))
LAT_LIM = (pts.lat.quantile(0.001), pts.lat.quantile(0.999))

# ─────────────────────────────────────────────
# 2-1 클래스별 점 밀도
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True, sharey=True)
for ax, c in zip(axes, C.CLASSES):
    s = pts[pts.fishing_code == c]
    ax.hexbin(s.lon, s.lat, gridsize=180, bins="log", cmap="viridis", mincnt=1,
              extent=(*LON_LIM, *LAT_LIM))
    ax.set_title(f"{c}  (점 {len(s):,})")
    ax.set_aspect(1.2)
fig.suptitle("조업형태별 항적 점 밀도 (log)")
C.savefig("02_density_by_class", fig)

# ─────────────────────────────────────────────
# 2-2 무작위 선박-일 궤적
# ─────────────────────────────────────────────
vd = pts.groupby(["rfid_id", "part_dt", "fishing_code"], observed=True)["hour"].nunique().reset_index(name="n_h")
N = 6
fig, axes = plt.subplots(4, N, figsize=(4 * N, 16))
for i, c in enumerate(C.CLASSES):
    cand = vd[(vd.fishing_code == c) & (vd.n_h >= 6)]
    pick = cand.sample(min(N, len(cand)), random_state=C.RANDOM_STATE)
    for j, (_, r) in enumerate(pick.iterrows()):
        ax = axes[i, j]
        t = pts[(pts.rfid_id == r.rfid_id) & (pts.part_dt == r.part_dt)].sort_values(["hour_f", "seq"])
        ax.plot(t.lon, t.lat, lw=0.5, color="lightgray")
        sc = ax.scatter(t.lon, t.lat, c=t.hour_f, cmap="turbo", s=3, vmin=0, vmax=23)
        ax.scatter(t.lon.iloc[0], t.lat.iloc[0], marker="^", c="k", s=40)
        ax.set_title(f"{c} | {r.rfid_id} | {r.part_dt} | {r.n_h}h", fontsize=8)
        ax.set_aspect(1.2); ax.tick_params(labelsize=6)
fig.colorbar(sc, ax=axes, shrink=0.4, label="rfid_time (시)")
fig.suptitle("클래스별 무작위 하루 항적 (▲=첫 점)", y=0.92)
C.savefig("02_sample_tracks", fig)

# ─────────────────────────────────────────────
# 2-3 출발점 정렬 스파게티 플롯 (km)
# ─────────────────────────────────────────────
M = 60
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True, sharey=True)
for ax, c in zip(axes, C.CLASSES):
    cand = vd[(vd.fishing_code == c) & (vd.n_h >= 4)]
    pick = cand.sample(min(M, len(cand)), random_state=1)
    for _, r in pick.iterrows():
        t = pts[(pts.rfid_id == r.rfid_id) & (pts.part_dt == r.part_dt)].sort_values(["hour_f", "seq"])
        x, y = to_local_km(t.lon.to_numpy(), t.lat.to_numpy(), t.lon.iloc[0], t.lat.iloc[0])
        ax.plot(x, y, lw=0.6, alpha=0.6, color=pal[c])
    ax.axhline(0, c="k", lw=0.3); ax.axvline(0, c="k", lw=0.3)
    ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_aspect("equal")
    ax.set_title(f"{c} (n={len(pick)})"); ax.set_xlabel("동서 km")
axes[0].set_ylabel("남북 km")
fig.suptitle("출발점을 원점으로 옮긴 하루 궤적 – 이동 반경/형태 비교")
C.savefig("02_spaghetti_origin", fig)

# ─────────────────────────────────────────────
# 2-4 plotly 인터랙티브 (pip install plotly)
# ─────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    fig = go.Figure()
    for c in C.CLASSES:
        cand = vd[(vd.fishing_code == c) & (vd.n_h >= 4)]
        pick = cand.sample(min(40, len(cand)), random_state=2)
        for k, (_, r) in enumerate(pick.iterrows()):
            t = pts[(pts.rfid_id == r.rfid_id) & (pts.part_dt == r.part_dt)].sort_values(["hour_f", "seq"])
            fig.add_trace(go.Scattermap(lon=t.lon, lat=t.lat, mode="lines", line=dict(width=2, color=pal[c]),
                                        name=c, legendgroup=c, showlegend=(k == 0),
                                        hovertext=f"{c} {r.rfid_id} {r.part_dt}"))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lon=127.5, lat=35.5), zoom=5.5),
                      height=800, title="조업형태별 하루 항적 (인터랙티브)")
    fig.write_html(C.OUT_DIR / "02_tracks_interactive.html")
    print("  [saved] 02_tracks_interactive.html")
except ImportError:
    print("  plotly 미설치 → 2-4 생략 (pip install plotly)")
except Exception as e:
    print("  plotly 지도 실패(버전 문제일 수 있음):", e)

print("\n✅ EDA 2 완료")
