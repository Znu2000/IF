"""
[EDA 5] 사고 데이터 분석 + 항적 매칭 가능성 점검

  5-1 범주형 컬럼 분포 (원인/유형/기상/어업면허/해역/긴급도)
  5-2 사고 유형 → 대분류 묶기 (클래스가 너무 잘게 쪼개진 문제 대응)
  5-3 시간 분포 (월/시/요일)
  5-4 사고 위치 지도 + 항적 밀도 배경
  5-5 시공간 매칭 민감도: 반경(km) × 시간창(h) 별 매칭률 / 후보 선박 수(모호성)
       → 결과 cache/acc_matches.parquet 저장 (EDA 9에서 사용)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from geo_utils import load_accidents, match_accidents, ACC_COLS, ACC_COLS_KR

C.set_korean_font()
acc = load_accidents(C.accident_file())
kr = dict(zip(ACC_COLS, ACC_COLS_KR))
print(f"사고 {len(acc)}건, 기간 {acc.acc_time.min()} ~ {acc.acc_time.max()}")

# ─────────────────────────────────────────────
# 5-1 범주형 분포
# ─────────────────────────────────────────────
CAT = ["cause", "acc_type", "weather", "license", "sea_area", "urgency", "coast_guard", "ship_type"]
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
for ax, c in zip(axes.flat, CAT):
    vc = acc[c].value_counts().head(15)
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color="#4C72B0")
    ax.set_title(f"{kr[c]} (고유값 {acc[c].nunique()}개)")
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
fig.tight_layout()
C.savefig("05_categorical", fig)
for c in CAT:
    print(f"\n[{kr[c]}]\n", acc[c].value_counts().to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(acc.tonnage.dropna(), bins=30); axes[0].set_title("톤수 분포")
acc[["n_people", "rescued", "injured", "dead", "missing"]].sum().plot.bar(ax=axes[1])
axes[1].set_xticklabels(["발생인원", "구조", "부상", "사망", "실종"], rotation=0); axes[1].set_title("인원 합계")
C.savefig("05_tonnage_people", fig)

# ─────────────────────────────────────────────
# 5-2 사고 유형 대분류
# ─────────────────────────────────────────────
GROUP_RULES = [  # (키워드들, 대분류, 항적으로 보일 가능성)
    (("기관", "추진기", "타기", "감김", "냉각", "연료", "배터리", "전기"), "기관/추진계 고장", "높음: 바다 한가운데 정지·표류"),
    (("충돌", "접촉"), "충돌/접촉", "중간: 두 선박 근접 후 급정지"),
    (("좌초", "좌주", "좌礁"), "좌초", "높음: 연안 급정지"),
    (("침수", "전복", "침몰"), "침수/전복", "중간: 신호 소실 가능"),
    (("화재", "폭발"), "화재/폭발", "낮음~중간"),
    (("인명", "추락", "부상", "실종", "익수", "환자"), "인명사고", "낮음: 항적은 정상일 수 있음"),
]


def group_type(s):
    s = str(s)
    for kws, g, _ in GROUP_RULES:
        if any(k in s for k in kws):
            return g
    return "기타"


acc["type_group"] = acc["acc_type"].map(group_type)
print("\n사고 유형 → 대분류\n", pd.crosstab(acc.acc_type, acc.type_group).to_string())
print("\n대분류 건수\n", acc.type_group.value_counts().to_string())
print("\n항적에서 보일 가능성 (가설)")
for _, g, v in GROUP_RULES:
    print(f"  - {g}: {v}")
print("\n원인 × 대분류\n", pd.crosstab(acc.cause, acc.type_group, margins=True).to_string())

# ─────────────────────────────────────────────
# 5-3 시간 분포
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
acc.acc_time.dt.month.value_counts().sort_index().plot.bar(ax=axes[0], title="월별 사고")
acc.hour.value_counts().reindex(range(24), fill_value=0).plot.bar(ax=axes[1], title="시간대별 사고")
acc.acc_time.dt.dayofweek.value_counts().reindex(range(7), fill_value=0).plot.bar(ax=axes[2], title="요일별 사고")
axes[2].set_xticklabels(list("월화수목금토일"), rotation=0)
C.savefig("05_temporal", fig)

# ─────────────────────────────────────────────
# 5-4 지도
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
pts_path = C.CACHE_DIR / "sample_points.parquet"
if pts_path.exists():
    pts = pd.read_parquet(pts_path, columns=["lon", "lat"])
    ax.hexbin(pts.lon, pts.lat, gridsize=250, bins="log", cmap="Greys", mincnt=1,
              extent=(124, 131.5, 32.5, 38.7))
for g, sub in acc.groupby("type_group"):
    ax.scatter(sub.lon, sub.lat, s=40, label=f"{g} ({len(sub)})", edgecolor="k", lw=0.5)
ax.set_xlim(124, 131.5); ax.set_ylim(32.5, 38.7); ax.set_aspect(1.2); ax.legend()
ax.set_title("사고 위치 (배경: 1% 선박 항적 밀도)")
C.savefig("05_accident_map", fig)

# ─────────────────────────────────────────────
# 5-5 매칭 민감도
# ─────────────────────────────────────────────
cand_path = C.CACHE_DIR / "acc_candidates.parquet"
if not cand_path.exists():
    print("acc_candidates.parquet 없음 → 00_build_cache.py 먼저 실행")
else:
    cand = pd.read_parquet(cand_path)
    print(f"\n후보 행 {len(cand):,}개 / 사고 중 어업면허→조업형태 매핑 성공 {acc.cls.notna().sum()}/{len(acc)}")
    R_MAX, T_MAX = 10, 3
    m_same = match_accidents(acc, cand, radius_km=R_MAX, time_window_h=T_MAX, same_class=True)
    m_any = match_accidents(acc, cand, radius_km=R_MAX, time_window_h=T_MAX, same_class=False)
    m_same.to_parquet(C.CACHE_DIR / "acc_matches.parquet", index=False)
    m_any.to_parquet(C.CACHE_DIR / "acc_matches_anyclass.parquet", index=False)

    rows = []
    for name, m in [("같은 조업형태만", m_same), ("조업형태 무시", m_any)]:
        for r in [0.5, 1, 2, 3, 5, 10]:
            for t in [0, 1, 2, 3]:
                s = m[(m.min_dist_km <= r) & (m.dt_h.abs() <= t)]
                nv = s.groupby("acc_id")["rfid_id"].nunique()
                rows.append(dict(조건=name, radius_km=r, window_h=t,
                                 매칭률=len(nv) / len(acc) * 100,
                                 단일후보_비율=(nv == 1).sum() / len(acc) * 100,
                                 평균후보선박수=nv.mean() if len(nv) else 0))
    sens = pd.DataFrame(rows)
    sens.to_csv(C.OUT_DIR / "05_match_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n매칭 민감도 (같은 조업형태)\n",
          sens[sens.조건 == "같은 조업형태만"].pivot(index="radius_km", columns="window_h", values="매칭률").round(1))
    print("\n단일 후보(=선박 특정 가능) 비율\n",
          sens[sens.조건 == "같은 조업형태만"].pivot(index="radius_km", columns="window_h", values="단일후보_비율").round(1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    s = sens[sens.조건 == "같은 조업형태만"]
    for ax, col in zip(axes, ["매칭률", "단일후보_비율", "평균후보선박수"]):
        sns.heatmap(s.pivot(index="radius_km", columns="window_h", values=col), annot=True, fmt=".1f",
                    cmap="YlGnBu", ax=ax)
        ax.set_title(col + (" (%)" if "비율" in col or "률" in col else ""))
    fig.suptitle("사고 ↔ 항적 매칭 민감도 (반경 × 시간창, 같은 조업형태)")
    C.savefig("05_match_sensitivity", fig)

    best = m_same[m_same["rank"] == 1].groupby("acc_id").first()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(best.min_dist_km, bins=40)
    ax.set_xlabel("사고 지점 ~ 최근접 항적 거리 (km)"); ax.set_title("사고별 1순위 후보 선박까지 거리")
    C.savefig("05_best_match_distance", fig)
    unmatched = acc[~acc.acc_id.isin(m_same.acc_id)]
    print(f"\n10km/±3h 에도 매칭 안 된 사고 {len(unmatched)}건 → 유형:\n",
          unmatched.type_group.value_counts().to_string())

print("\n✅ EDA 5 완료")
