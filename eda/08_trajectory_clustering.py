"""
[EDA 8] 궤적 클러스터링 ─ 라벨 없이 묶었을 때 조업형태와 얼마나 일치하나?

  8-1 하루 궤적 '모양' 클러스터링
       · 선박-일 궤적을 호(arc) 길이 기준 32점으로 리샘플 → 출발점 원점 이동 → 최원점 방향 회전
       · KMeans (k는 실루엣으로 선택) → 클러스터 대표 궤적 + 클래스 구성비
  8-2 행동 피처(선박-일) 클러스터링 → 클러스터 프로파일
  8-3 어장(hotspot) 클러스터링: 저속(조업 추정) 지점 HDBSCAN → 어장 지도
  ※ DTW 기반 클러스터링은 `pip install tslearn` 후 TimeSeriesKMeans(metric="dtw") 로 확장 가능 (느림)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

import config as C
from geo_utils import to_local_km

C.set_korean_font()
pal = C.CLASS_COLORS
N_RS = 32

# ─────────────────────────────────────────────
# 8-1 모양 클러스터링
# ─────────────────────────────────────────────
pts = pd.read_parquet(C.CACHE_DIR / "sample_points.parquet").dropna(subset=["fishing_code", "hour"])
pts = pts.sort_values(["rfid_id", "part_dt", "hour", "seq"])


def resample_track(lon, lat, n=N_RS, rotate=True):
    x, y = to_local_km(lon, lat, lon[0], lat[0])
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if d[-1] < 0.2:  # 거의 안 움직인 날은 제외
        return None
    u = np.linspace(0, d[-1], n)
    xr, yr = np.interp(u, d, x), np.interp(u, d, y)
    if rotate:
        k = np.argmax(xr**2 + yr**2)
        th = -np.arctan2(yr[k], xr[k])
        xr, yr = xr * np.cos(th) - yr * np.sin(th), xr * np.sin(th) + yr * np.cos(th)
        if yr[np.argmax(np.abs(yr))] < 0:  # 좌우 반전 통일
            yr = -yr
    return np.r_[xr, yr]


tracks, meta = [], []
for (rid, dt, cls), g in pts.groupby(["rfid_id", "part_dt", "fishing_code"], observed=True, sort=False):
    if g.hour.nunique() < 3:
        continue
    v = resample_track(g.lon.to_numpy(np.float64), g.lat.to_numpy(np.float64))
    if v is not None:
        tracks.append(v); meta.append((rid, dt, cls))
T = np.vstack(tracks)
meta = pd.DataFrame(meta, columns=["rfid_id", "part_dt", "fishing_code"])
print(f"클러스터링 대상 선박-일 궤적: {len(T):,}")

# 스케일 영향 완화: 크기(최대거리)를 따로 피처로 두고 모양은 정규화한 버전도 사용
scale = np.abs(T).max(1, keepdims=True)
T_shape = np.hstack([T / np.maximum(scale, 1e-6), np.log1p(scale)])

rng = np.random.default_rng(C.RANDOM_STATE)
sub = rng.choice(len(T_shape), min(15_000, len(T_shape)), replace=False)
sil = {}
for k in range(4, 13):
    lab = KMeans(k, n_init=5, random_state=C.RANDOM_STATE).fit_predict(T_shape[sub])
    sil[k] = silhouette_score(T_shape[sub], lab, sample_size=5000, random_state=C.RANDOM_STATE)
best_k = max(sil, key=sil.get)
print("실루엣:", {k: round(v, 3) for k, v in sil.items()}, "→ k =", best_k)
km = KMeans(best_k, n_init=10, random_state=C.RANDOM_STATE).fit(T_shape)
meta["cluster"] = km.labels_
print(f"ARI={adjusted_rand_score(meta.fishing_code, meta.cluster):.3f}, "
      f"NMI={normalized_mutual_info_score(meta.fishing_code, meta.cluster):.3f}  (1에 가까울수록 클러스터=조업형태)")

ncol = 4
nrow = int(np.ceil(best_k / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
for k, ax in enumerate(axes.flat):
    if k >= best_k:
        ax.axis("off"); continue
    idx = np.where(meta.cluster == k)[0]
    for i in rng.choice(idx, min(40, len(idx)), replace=False):
        ax.plot(T[i, :N_RS], T[i, N_RS:], lw=0.5, alpha=0.5, color=pal[meta.fishing_code.iat[i]])
    comp = meta.loc[idx, "fishing_code"].value_counts(normalize=True)
    ax.set_title(f"C{k} (n={len(idx)}) " + " ".join(f"{c[:2]}{comp.get(c, 0) * 100:.0f}%" for c in C.CLASSES), fontsize=9)
    ax.set_aspect("equal"); ax.axhline(0, c="k", lw=0.3); ax.axvline(0, c="k", lw=0.3)
fig.suptitle("궤적 모양 클러스터 (선 색=실제 조업형태, 출발점 원점·회전 정렬, km)")
C.savefig("08_shape_clusters", fig)

ct = pd.crosstab(meta.cluster, meta.fishing_code, normalize="index")[C.CLASSES]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", ax=axes[0]); axes[0].set_title("클러스터별 조업형태 구성비")
sns.heatmap(pd.crosstab(meta.fishing_code, meta.cluster, normalize="index").loc[C.CLASSES], annot=True, fmt=".2f",
            cmap="Oranges", ax=axes[1]); axes[1].set_title("조업형태별 클러스터 분포")
C.savefig("08_shape_cluster_composition", fig)
meta.to_parquet(C.CACHE_DIR / "shape_clusters.parquet", index=False)

# ─────────────────────────────────────────────
# 8-2 행동 피처 클러스터링
# ─────────────────────────────────────────────
daily = pd.read_parquet(C.CACHE_DIR / "sample_daily.parquet") if (C.CACHE_DIR / "sample_daily.parquet").exists() else None
if daily is not None:
    DAY_F = ["n_hours", "first_hour", "total_path_km", "max_range_km", "mean_speed_kn",
             "max_speed_kn", "mean_turn", "mean_straight", "mean_stop"]
    D = daily.dropna(subset=DAY_F + ["fishing_code"]).copy()
    Z = StandardScaler().fit_transform(np.log1p(D[DAY_F].clip(lower=0)))
    D["bcluster"] = KMeans(6, n_init=10, random_state=C.RANDOM_STATE).fit_predict(Z)
    prof = D.groupby("bcluster")[DAY_F].median().astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    sns.heatmap((prof - prof.mean()) / prof.std(), annot=prof.round(1), fmt="", cmap="RdBu_r", center=0, ax=axes[0])
    axes[0].set_title("행동 클러스터 프로파일 (색=표준화, 숫자=중앙값)")
    sns.heatmap(pd.crosstab(D.bcluster, D.fishing_code, normalize="index")[C.CLASSES], annot=True, fmt=".2f",
                cmap="Blues", ax=axes[1]); axes[1].set_title("행동 클러스터별 조업형태 구성비")
    C.savefig("08_behavior_clusters", fig)
    print(f"행동 클러스터 NMI={normalized_mutual_info_score(D.fishing_code, D.bcluster):.3f}")

# ─────────────────────────────────────────────
# 8-3 어장 hotspot (HDBSCAN)
# ─────────────────────────────────────────────
feats = pd.read_parquet(C.CACHE_DIR / "sample_feats.parquet",
                        columns=["fishing_code", "c_lon", "c_lat", "speed_kn", "stop_ratio"]).dropna()
slow = feats[(feats.speed_kn.between(0.3, 4)) & (feats.stop_ratio < 0.8)]  # 조업 추정 구간
slow = slow.sample(min(40_000, len(slow)), random_state=C.RANDOM_STATE)
X = np.radians(slow[["c_lat", "c_lon"]].to_numpy())
hd = HDBSCAN(min_cluster_size=80, min_samples=20, metric="haversine").fit(X)
slow["hot"] = hd.labels_
n_hot = slow.hot.max() + 1
print(f"어장 hotspot {n_hot}개 (노이즈 {(slow.hot == -1).mean() * 100:.1f}%)")
hot = slow[slow.hot >= 0].groupby("hot").agg(lon=("c_lon", "mean"), lat=("c_lat", "mean"), n=("c_lon", "size"),
                                              dom=("fishing_code", lambda s: s.mode().iat[0]),
                                              purity=("fishing_code", lambda s: s.value_counts(normalize=True).iat[0]))
hot.to_csv(C.OUT_DIR / "08_hotspots.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(slow.c_lon, slow.c_lat, s=1, c="lightgray")
for c in C.CLASSES:
    h = hot[hot.dom == c]
    ax.scatter(h.lon, h.lat, s=np.sqrt(h.n) * 4, color=pal[c], alpha=0.7, edgecolor="k", lw=0.4, label=c)
ax.set_xlim(124, 131.5); ax.set_ylim(32.5, 38.7); ax.set_aspect(1.2); ax.legend()
ax.set_title(f"어장 hotspot (HDBSCAN, 크기=점수, 색=우세 조업형태) – {n_hot}개")
C.savefig("08_hotspots", fig)
print("\n✅ EDA 8 완료")
