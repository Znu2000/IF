# -*- coding: utf-8 -*-
"""SeD 템플릿 기반 진행상황 보고 PPT 생성 — 공통 헬퍼"""
import copy, re
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.oxml.ns import qn
from PIL import Image

FONT = "Pretendard JP"
NAVY = RGBColor(0, 14, 142)
BLUE = RGBColor(57, 87, 219)
ORANGE = RGBColor(252, 175, 22)
DARK = RGBColor(33, 37, 41)
GRAY = RGBColor(108, 117, 125)
LIGHT = RGBColor(236, 240, 252)
LIGHT2 = RGBColor(247, 248, 253)
LORANGE = RGBColor(255, 245, 222)
RED = RGBColor(207, 60, 60)
GREEN = RGBColor(34, 150, 80)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(205, 212, 235)


def set_run_font(run, size=None, bold=None, color=None, italic=None):
    f = run.font
    f.name = FONT
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT)
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = color


_MARK = re.compile(r"(\*\*.+?\*\*|\[\[.+?\]\]|\{\{.+?\}\})")


def add_runs(p, text, size, color=DARK, bold=False):
    """**굵게(네이비)**, [[강조(파랑 굵게)]], {{경고(빨강 굵게)}} 마크업 지원"""
    for part in _MARK.split(text):
        if not part:
            continue
        c, b = color, bold
        if part.startswith("**"):
            part, c, b = part[2:-2], (NAVY if color == DARK else color), True
        elif part.startswith("[["):
            part, c, b = part[2:-2], BLUE, True
        elif part.startswith("{{"):
            part, c, b = part[2:-2], RED, True
        r = p.add_run()
        r.text = part
        set_run_font(r, size, b, c)


def _bullet(p, char, color, marL, indent):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(marL))))
    pPr.set("indent", str(int(-Inches(indent))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buClr", "a:buFont", "a:buSzPct"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    if char is None:
        pPr.append(pPr.makeelement(qn("a:buNone"), {}))
        return
    buClr = pPr.makeelement(qn("a:buClr"), {})
    srgb = buClr.makeelement(qn("a:srgbClr"), {"val": "%02X%02X%02X" % tuple(color)})
    buClr.append(srgb)
    pPr.append(buClr)
    pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
    pPr.append(pPr.makeelement(qn("a:buChar"), {"char": char}))


def textbox(slide, x, y, w, h, items, size=13, color=DARK, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, bold=False, spacing=4, line=1.1, margin=0.04):
    """items: str | list. list 원소 = str 또는 (level, str). level -1 = 불릿 없음"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    if isinstance(items, str):
        items = [(-1, items)]
    first = True
    for it in items:
        lvl, txt = (it if isinstance(it, tuple) else (-1, it))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(spacing)
        if lvl == -1:
            _bullet(p, None, BLUE, 0, 0)
            add_runs(p, txt, size, color, bold)
        elif lvl == 0:
            _bullet(p, "▪", BLUE, 0.2, 0.2)
            add_runs(p, txt, size, color, bold)
        elif lvl == 1:
            _bullet(p, "•", GRAY, 0.42, 0.18)
            add_runs(p, txt, size - 1.5, color, bold)
        else:
            _bullet(p, "-", GRAY, 0.62, 0.16)
            add_runs(p, txt, size - 2.5, GRAY, bold)
    return tb


def rect(slide, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.text = ""
    return s


def shape_text(s, text, size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(0.05))
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.05
        add_runs(p, ln, size, color, bold)


def card(slide, x, y, w, h, title, items, size=11.5, accent=BLUE, fill=LIGHT2, title_size=13.5):
    rect(slide, x, y, w, h, fill=fill, line=LINE)
    rect(slide, x, y, w, 0.06, fill=accent)
    textbox(slide, x + 0.12, y + 0.12, w - 0.24, 0.4, title, size=title_size, bold=True, color=accent)
    if items:
        textbox(slide, x + 0.12, y + 0.52, w - 0.24, h - 0.6, items, size=size)


def callout(slide, x, y, w, h, text, size=12.5, label="시사점"):
    rect(slide, x, y, w, h, fill=LORANGE)
    rect(slide, x, y, 0.08, h, fill=ORANGE)
    items = text if isinstance(text, list) else [(-1, f"[[{label}]]  " + text if label else text)]
    textbox(slide, x + 0.18, y, w - 0.25, h, items, size=size, anchor=MSO_ANCHOR.MIDDLE, spacing=2)


def table(slide, x, y, widths, rows, size=9.5, header_fill=BLUE, row_h=0.28,
          center_cols=(), bold_first_col=True, zebra=True, header_size=None, col_colors=None):
    nrows, ncols = len(rows), len(widths)
    gs = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * nrows))
    tbl = gs.table
    tblPr = tbl._tbl.tblPr
    # 기본 스타일 영향 제거
    for attr in ("bandRow", "firstRow"):
        tblPr.set(attr, "0")
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    for r in range(nrows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(ncols):
            cell = tbl.cell(r, c)
            txt = rows[r][c] if c < len(rows[r]) else ""
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.025)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            elif zebra and r % 2 == 0:
                cell.fill.fore_color.rgb = LIGHT
            else:
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            lines = str(txt).split("\n")
            for li, ln in enumerate(lines):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                p.line_spacing = 1.0
                p.alignment = PP_ALIGN.CENTER if (r == 0 or c in center_cols) else PP_ALIGN.LEFT
                if r == 0:
                    add_runs(p, ln, header_size or size, WHITE, True)
                else:
                    col = DARK
                    if col_colors and c in col_colors:
                        col = col_colors[c]
                    add_runs(p, ln, size, col, bold_first_col and c == 0)
    return gs


def picture(slide, path, x, y, w, h, align="center"):
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if align == "center" else x
    py = y + (h - ph) / 2
    slide.shapes.add_picture(path, Inches(px), Inches(py), Inches(pw), Inches(ph))
    return px, py, pw, ph


def caption(slide, x, y, w, text, size=9.5):
    textbox(slide, x, y, w, 0.3, text, size=size, color=GRAY, align=PP_ALIGN.CENTER)


def arrow_right(slide, x, y, w=0.35, h=0.3, color=ORANGE):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background(); s.shadow.inherit = False
    return s


def arrow_down(slide, x, y, w=0.3, h=0.3, color=ORANGE):
    s = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background(); s.shadow.inherit = False
    return s


class Deck:
    def __init__(self, template):
        self.prs = Presentation(template)
        sl = self.prs.slides._sldIdLst
        for sid in list(sl):
            self.prs.part.drop_rel(sid.rId)
            sl.remove(sid)
        L = self.prs.slide_layouts
        self.L_title, self.L_contents, self.L_body, self.L_ref, self.L_qna = L[0], L[1], L[2], L[3], L[4]

    def _slide_number(self, slide):
        for ph in slide.slide_layout.placeholders:
            if ph.placeholder_format.type == PP_PLACEHOLDER.SLIDE_NUMBER:
                slide.shapes._spTree.append(copy.deepcopy(ph._element))

    def title_slide(self, title, subtitle):
        s = self.prs.slides.add_slide(self.L_title)
        for ph in s.placeholders:
            idx = ph.placeholder_format.idx
            tf = ph.text_frame
            tf.clear()
            if idx == 0:
                for i, ln in enumerate(title.split("\n")):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    add_runs(p, ln, 34, WHITE, True)
            else:
                for i, ln in enumerate(subtitle.split("\n")):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    add_runs(p, ln, 15 if i else 17, ORANGE if i == 0 else WHITE, i == 0)
        return s

    def contents_slide(self, entries):
        s = self.prs.slides.add_slide(self.L_contents)
        for ph in s.placeholders:
            idx = ph.placeholder_format.idx
            tf = ph.text_frame
            tf.clear()
            if idx == 11:
                add_runs(tf.paragraphs[0], "Contents", 44, WHITE, True)
            elif idx == 10:
                ph.left, ph.top, ph.width, ph.height = Inches(6.9), Inches(0.75), Inches(6.2), Inches(6.2)
                tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                for i, (num, name, sub) in enumerate(entries):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    _bullet(p, None, BLUE, 0, 0)
                    p.space_after = Pt(5)
                    r = p.add_run(); r.text = f"{num}   "; set_run_font(r, 19, True, ORANGE)
                    r = p.add_run(); r.text = name; set_run_font(r, 19, True, WHITE)
                    if sub:
                        r = p.add_run(); r.text = f"   {sub}"; set_run_font(r, 11.5, False, RGBColor(214, 222, 255))
        return s

    def body_slide(self, title, section=None):
        s = self.prs.slides.add_slide(self.L_body)
        for ph in list(s.placeholders):
            idx = ph.placeholder_format.idx
            if idx == 0:
                tf = ph.text_frame
                tf.clear()
                add_runs(tf.paragraphs[0], title, 26, WHITE, True)
            else:
                ph._element.getparent().remove(ph._element)
        self._slide_number(s)
        if section:
            textbox(s, 9.6, 0.33, 3.5, 0.4, section, size=12, color=RGBColor(214, 222, 255),
                    align=PP_ALIGN.RIGHT, bold=True)
        return s

    def ref_slide(self, title, left, right, size=9.5):
        s = self.prs.slides.add_slide(self.L_ref)
        for ph in list(s.placeholders):
            if ph.placeholder_format.idx == 0:
                ph.text_frame.clear()
                add_runs(ph.text_frame.paragraphs[0], title, 26, WHITE, True)
            else:
                ph._element.getparent().remove(ph._element)
        textbox(s, 0.6, 1.3, 6.0, 5.5, left, size=size, spacing=3)
        textbox(s, 6.8, 1.3, 6.0, 5.5, right, size=size, spacing=3)
        return s

    def qna_slide(self):
        return self.prs.slides.add_slide(self.L_qna)

    def save(self, path):
        self.prs.save(path)
