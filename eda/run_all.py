"""
EDA 전체 순서 실행 + 각 단계 로그 저장 (outputs/logs/*.txt)

  python run_all.py              # 00~09 전부
  python run_all.py --skip-cache # 캐시가 이미 있으면 01~09만
  python run_all.py --only 03 06 # 특정 단계만
"""
import argparse
import subprocess
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_build_cache", "01_overview", "02_trajectory_by_class", "03_spatial_features",
         "04_temporal_patterns", "05_accident_analysis", "06_class_separability",
         "07_spatiotemporal", "08_trajectory_clustering", "09_anomaly_3d"]

ap = argparse.ArgumentParser()
ap.add_argument("--skip-cache", action="store_true")
ap.add_argument("--only", nargs="*", default=None)
args = ap.parse_args()

steps = STEPS[1:] if args.skip_cache else STEPS
if args.only:
    steps = [s for s in STEPS if s[:2] in args.only]

log_dir = HERE / "outputs" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

for s in steps:
    t = time.time()
    print(f"\n{'=' * 70}\n▶ {s}")
    with open(log_dir / f"{s}.txt", "w", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, f"{s}.py"], cwd=HERE, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        p.wait()
    print(f"◀ {s}  ({(time.time() - t) / 60:.1f}분, exit={p.returncode})")
    if p.returncode != 0:
        print("  ⚠ 실패 → 로그 확인 후 계속 진행합니다")
print(f"\n로그: {log_dir}")
