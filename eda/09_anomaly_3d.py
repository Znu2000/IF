"""
[EDA 9] 사고(이상) ↔ 항적 매칭 후 정상/이상 분리 3D 시각화

사고 데이터는 날짜 외에 '위도·경도·(시각)' + 어업면허 정도만 항적과 연결 가능
 → (1) 시공간으로 사고 선박 후보를 찾고 (EDA 5의 acc_matches)
 → (2) 그 선박의 사고 시점 전후 항적 피처를 계산해
 → (3) 정상 항적과 3D 공간에서 갈리는지 본다.

  9-0 (경도, 위도, 시각) 3D: 정상 활동 vs 사고 지점
  9-1 이상 라벨 구성: 사고시점(±1h) / 사고 전(-12h~-2h) / 정상(같은 조업형태 무작위)
  9-2 선택 피처 3개 3D 산점도 (FEATURES_3D 바꿔가며 실험)
  9-3 전체 피처 PCA 3D + IsolationForest 점수 분포/AUC  → "비지도 점수로 사고가 튀어 보이나?"
  9-4 사고 선박 Space-Time Cube (사고 지점 ★)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import config as C
from geo_utils import load_accidents, features_from_df, parse_multilinestring, FEATURE_COLS

C.set_korean_font()
import plotly.graph_objects as go  # pip install plotly

# ★ 여기 3개를 바꿔가며 분리가 잘 되는 조합을 찾기
FEATURES_3D = ["speed_kn", "turn_abs_mean", "stop_ratio"]
MATCH_R_KM, MATCH_T_H = 3.0, 1      # 매칭 확정 기준 (EDA 5 민감도 결과 보고 조정)
NORMAL_N = 30_000

acc = load_accidents(C.accident_file())
feats = pd.read_parquet(C.CACHE_DIR / "sample_feats.parquet").dropna(subset=["fishing_code"])
feats["hour_f"] = feats.hour.astype(float)

# ─────────────────────────────────────────────
# 9-0 (lon, lat, hour) 3D
# ─────────────────────────────────────────────
bg = feats.dropna(subset=["hour"]).sample(min(40_000, len(feats)), random_state=0)
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=bg.c_lon, y=bg.c_lat, z=bg.hour_f, mode="markers",
                           marker=dict(size=1.5, color="lightgray", opacity=0.3), name="정상 활동(샘플)"))
fig.add_trace(go.Scatter3d(x=acc.lon, y=acc.lat, z=acc.hour, mode="markers",
                           marker=dict(size=5, color="red", symbol="diamond"), name="사고",
                           text=acc.acc_type.astype(str) + " / " + acc.cause.astype(str)))
fig.update_layout(scene=dict(xaxis_title="경도", yaxis_title="위도", zaxis_title="시각(시)"),
                  title="정상 활동 vs 사고 지점 (경도·위도·시각)", height=850)
fig.write_html(C.OUT_DIR / "09_lonlat_hour_3d.html"); print("  [saved] 09_lonlat_hour_3d.html")

# ─────────────────────────────────────────────
# 9-1 이상 라벨 구성
# ─────────────────────────────────────────────
mpath, cpath = C.CACHE_DIR / "acc_matches.parquet", C.CACHE_DIR / "acc_candidates.parquet"
if not (mpath.exists() and cpath.exists()):
    raise SystemExit("acc_matches/acc_candidates 없음 → 00_build_cache.py, 05_accident_analysis.py 먼저 실행")
m = pd.read_parquet(mpath)
cand = pd.read_parquet(cpath).dropna(subset=["hour"])
best = (m[(m.min_dist_km <= MATCH_R_KM) & (m.dt_h.abs() <= MATCH_T_H) & (m["rank"] == 1)]
        .sort_values("min_dist_km").drop_duplicates("acc_id"))
print(f"매칭 확정 사고: {len(best)}/{len(acc)}건  (반경 {MATCH_R_KM}km, ±{MATCH_T_H}h, 1순위 선박)")
if best.empty:
    raise SystemExit("매칭된 사고가 없음 → MATCH_R_KM / MATCH_T_H 를 늘리거나 전체 캐시로 다시 실행")

cand["ts"] = pd.to_datetime(cand.part_dt.astype(str), format="%Y%m%d") + pd.to_timedelta(cand.hour.astype(int), unit="h")
rows = []
for b in best.itertuples():
    a = acc.loc[acc.acc_id == b.acc_id].iloc[0]
    v = cand[cand.rfid_id == b.rfid_id].copy()
    v["rel_h"] = (v.ts - a.acc_time.floor("h")) / pd.Timedelta(hours=1)
    v["acc_id"] = b.acc_id
    rows.append(v)
anom = pd.concat(rows).drop_duplicates(["rfid_id", "ts", "acc_id"]) if rows else pd.DataFrame()
anom = pd.concat([anom.reset_index(drop=True), features_from_df(anom.reset_index(drop=True))], axis=1)
anom["group"] = np.select([anom.rel_h.abs() <= 1, anom.rel_h.between(-12, -2)],
                          ["사고시점(±1h)", "사고 전(-12~-2h)"], "사고 후")
anom = anom[anom.group != "사고 후"]  # 사고 후(예인/구조)는 원인이 아니라 결과라 제외
anom = anom.merge(acc[["acc_id", "type_group"]] if "type_group" in acc else acc[["acc_id", "acc_type"]],
                  on="acc_id", how="left")
anom["hour_f"] = anom.hour.astype(float)

cls_w = anom.fishing_code.value_counts(normalize=True)
normal = pd.concat([feats[feats.fishing_code == c].sample(int(NORMAL_N * w), random_state=0)
                    for c, w in cls_w.items() if c in C.CLASSES])  # 사고선박과 같은 조업형태 비율로
normal["group"] = "정상"
print(anom.group.value_counts().to_string())

ALL = pd.concat([normal, anom], ignore_index=True)
ALL.to_parquet(C.CACHE_DIR / "anomaly_eval_set.parquet", index=False)
COLORS = {"정상": "lightgray", "사고 전(-12~-2h)": "orange", "사고시점(±1h)": "red"}

# ─────────────────────────────────────────────
# 9-2 선택 피처 3D
# ─────────────────────────────────────────────
def clip_q(s):
    return s.clip(s.quantile(0.005), s.quantile(0.995))


fig = go.Figure()
for g, col in COLORS.items():
    s = ALL[ALL.group == g]
    fig.add_trace(go.Scatter3d(
        x=clip_q(s[FEATURES_3D[0]]), y=clip_q(s[FEATURES_3D[1]]), z=clip_q(s[FEATURES_3D[2]]), mode="markers", name=f"{g} ({len(s)})",
        marker=dict(size=2 if g == "정상" else 5, color=col, opacity=0.25 if g == "정상" else 0.9)))
fig.update_layout(scene=dict(xaxis_title=FEATURES_3D[0], yaxis_title=FEATURES_3D[1], zaxis_title=FEATURES_3D[2]),
                  title=f"정상 vs 사고 – {', '.join(FEATURES_3D)}", height=850)
fig.write_html(C.OUT_DIR / "09_feature3d.html"); print("  [saved] 09_feature3d.html")

fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(projection="3d")
for g, col in COLORS.items():
    s = ALL[ALL.group == g]
    ax.scatter(*[clip_q(s[f]) for f in FEATURES_3D], s=1 if g == "정상" else 18, c=col,
               alpha=0.2 if g == "정상" else 0.9, label=g)
ax.set_xlabel(FEATURES_3D[0]); ax.set_ylabel(FEATURES_3D[1]); ax.set_zlabel(FEATURES_3D[2]); ax.legend()
C.savefig("09_feature3d_static", fig)

# 피처별 효과크기 (정상 vs 사고시점) → 3D 축 후보 고르기
NUM = [f for f in FEATURE_COLS if f not in ("c_lon", "c_lat", "s_lon", "s_lat", "e_lon", "e_lat")]
eff = {}
for f in NUM:
    a, n = ALL.loc[ALL.group == "사고시점(±1h)", f].dropna(), ALL.loc[ALL.group == "정상", f].dropna()
    if len(a) > 2:
        eff[f] = (a.median() - n.median()) / (n.quantile(0.75) - n.quantile(0.25) + 1e-9)
eff = pd.Series(eff).sort_values(key=np.abs, ascending=False)
print("\n정상 대비 사고시점 중앙값 차이 / IQR (절대값 큰 순 → 3D 축 후보)\n", eff.round(3).to_string())

# ─────────────────────────────────────────────
# 9-3 PCA 3D + IsolationForest
# ─────────────────────────────────────────────
Xcols = NUM + ["hour_f"]
X = ALL[Xcols].astype(float)
X = X.fillna(normal[Xcols].astype(float).median())
sc = StandardScaler().fit(X[ALL.group == "정상"])
Z = sc.transform(X)
P = PCA(3).fit(Z[ALL.group == "정상"]).transform(Z)
fig = go.Figure()
for g, col in COLORS.items():
    msk = (ALL.group == g).to_numpy()
    fig.add_trace(go.Scatter3d(x=P[msk, 0], y=P[msk, 1], z=P[msk, 2], mode="markers", name=g,
                               marker=dict(size=2 if g == "정상" else 5, color=col, opacity=0.25 if g == "정상" else 0.9)))
fig.update_layout(title="전체 피처 PCA 3D (정상 기준 적합)", height=850)
fig.write_html(C.OUT_DIR / "09_pca3d.html"); print("  [saved] 09_pca3d.html")

iso = IsolationForest(n_estimators=300, contamination="auto", random_state=C.RANDOM_STATE)
iso.fit(Z[ALL.group == "정상"])
ALL["anom_score"] = -iso.score_samples(Z)  # 클수록 이상
fig, ax = plt.subplots(figsize=(9, 4))
for g, col in COLORS.items():
    ax.hist(ALL.loc[ALL.group == g, "anom_score"], bins=60, density=True, histtype="step", lw=2,
            color=col if g != "정상" else "gray", label=g)
ax.set_xlabel("IsolationForest 이상 점수 (클수록 이상)"); ax.legend(); ax.set_title("정상 vs 사고 이상점수 분포")
C.savefig("09_iforest_score", fig)
for g in ["사고시점(±1h)", "사고 전(-12~-2h)"]:
    msk = ALL.group.isin(["정상", g])
    if (ALL.group == g).sum() > 0:
        auc = roc_auc_score((ALL.loc[msk, "group"] == g).astype(int), ALL.loc[msk, "anom_score"])
        print(f"IsolationForest AUC (정상 vs {g}): {auc:.3f}   (0.5=구분 못함)")
print("※ 행(1시간) 단위 '정적' 피처 기준. 실제 TSAD는 선박별 과거 대비 변화(개인 baseline)도 봐야 함")

# ─────────────────────────────────────────────
# 9-4 사고 선박 Space-Time Cube
# ─────────────────────────────────────────────
fig = go.Figure()
for i, b in enumerate(best.head(12).itertuples()):
    a = acc.loc[acc.acc_id == b.acc_id].iloc[0]
    v = cand[cand.rfid_id == b.rfid_id].sort_values("ts")
    xs, ys, zs = [], [], []
    for r in v.itertuples():
        parts = parse_multilinestring(r.geom)
        if not parts:
            continue
        p = np.vstack(parts)
        rel = (r.ts - a.acc_time.floor("h")) / pd.Timedelta(hours=1)
        xs += list(p[:, 0]); ys += list(p[:, 1]); zs += list(rel + np.linspace(0, 1, len(p), endpoint=False))
    # 경로를 사고 지점 기준 상대좌표로 → 여러 사고를 한 그림에 겹쳐 비교
    fig.add_trace(go.Scatter3d(x=np.array(xs) - a.lon, y=np.array(ys) - a.lat, z=zs, mode="lines",
                               line=dict(width=3), name=f"사고{b.acc_id} {a.acc_type}"))
fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=8, color="red", symbol="diamond"),
                           name="사고 지점/시각"))
fig.update_layout(scene=dict(xaxis_title="Δ경도", yaxis_title="Δ위도", zaxis_title="사고 기준 상대시간(h)"),
                  title="사고 선박 궤적 Space-Time Cube (사고 지점 원점)", height=850)
fig.write_html(C.OUT_DIR / "09_accident_stc.html"); print("  [saved] 09_accident_stc.html")
print("\n✅ EDA 9 완료")
