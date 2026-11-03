# -*- coding: utf-8 -*-
"""진행상황 보고 PPT 빌드 — python build_deck.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pptkit import *
from slides_eda import build_intro, build_eda
from slides_feat import build_prep, build_features, build_issues, build_plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "SeD_PPT_Template(v.final)_(1) (1).pptx")
OUT = os.path.join(ROOT, "SeD_Project_Progress_Report.pptx")
FIG = os.path.join(ROOT, "eda", "outputs")

d = Deck(TEMPLATE)
build_intro(d, FIG)
build_eda(d, FIG)
build_prep(d, FIG)
build_features(d, FIG)
build_issues(d, FIG)
build_plan(d, FIG)
try:
    d.save(OUT)
    print("saved", OUT, "slides:", len(d.prs.slides))
except PermissionError:
    alt_out = os.path.join(ROOT, "SeD_Project_Progress_Report_updated.pptx")
    d.save(alt_out)
    print(f"[알림] {OUT} 파일이 현재 파워포인트에서 열려 있어, 최신본을 {alt_out} 으로 저장했습니다. (슬라이드 수: {len(d.prs.slides)})")
