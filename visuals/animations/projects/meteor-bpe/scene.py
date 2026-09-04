"""Continuous, data-backed character-level BPE animation for ANIM-BPE-002."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from manim import (
    AnimationGroup, Create, CubicBezier, Dot, DOWN, FadeIn, FadeOut,
    Indicate, LEFT, Line, MoveAlongPath, ReplacementTransform, RIGHT,
    RoundedRectangle, Scene, SurroundingRectangle, TransformFromCopy, UP,
    VGroup, rate_functions,
)

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parents[1]))
from manim_style import (  # noqa: E402
    ACCENT, BASE, COMPOSED, FOREGROUND, GRID, INTERMEDIATE, MUTED,
    centered_positions, label, token,
)

COLORS = {"流": BASE, "星": BASE, "雨": BASE, "流星": INTERMEDIATE, "流星雨": COMPOSED}
CORPUS_LEFT = -6.15
CORPUS_YS = (1.68, 0.91, 0.14, -0.63, -1.40)
FOCUS = np.array([3.25, 0.40, 0.0])


def mixed(spans: list[tuple[str, bool]], size: int, color: str = FOREGROUND) -> VGroup:
    """Keep CJK and Latin font runs separate and align their visual centers."""

    return VGroup(*(label(s, size, color, cjk=cjk) for s, cjk in spans)).arrange(RIGHT, buff=0.06)


def cell(text: str) -> VGroup:
    """Small corpus token; spacing marks boundaries rather than tracking text."""

    color = COLORS.get(text, MUTED)
    width = 0.61 + 0.39 * (len(text) - 1)
    box = RoundedRectangle(
        width=width, height=0.57, corner_radius=0.10,
        stroke_color=color if text in COLORS else GRID, stroke_width=1.1,
        fill_color=color, fill_opacity=0.07 if text in COLORS else 0.025,
    )
    glyph = label(text, 29, color if text in COLORS else FOREGROUND, cjk=True)
    if glyph.width > width - 0.13:
        glyph.scale_to_fit_width(width - 0.13)
    glyph.move_to(box)
    glyph.set_z_index(2)
    return VGroup(box, glyph)


def row_positions(items: list[VGroup], y: float) -> list[np.ndarray]:
    cursor = CORPUS_LEFT
    positions = []
    for item in items:
        positions.append(np.array([cursor + item.width / 2, y, 0.0]))
        cursor += item.width + 0.13
    return positions


class MeteorBPE(Scene):
    """Count, merge, retain the vocabulary, then encode without retraining."""

    def mark(self, name: str) -> None:
        self.checkpoints.append({"name": name, "seconds": round(float(self.time), 3)})
        # Geometry assertions complement the contact-sheet and video review.
        for obj in self.mobjects:
            if obj.width and obj.height:
                assert obj.get_left()[0] >= -7.1, (name, "left overflow")
                assert obj.get_right()[0] <= 7.1, (name, "right overflow")
                assert obj.get_top()[1] <= 4.0, (name, "top overflow")
                assert obj.get_bottom()[1] >= -4.0, (name, "bottom overflow")

    def phase(self, number: str, text: str):
        result = mixed([(number, False), (text, True)], 23, ACCENT)
        result.to_edge(RIGHT, buff=0.56).to_edge(UP, buff=0.44)
        return result

    def construct(self) -> None:
        self.trace = json.loads((PROJECT / "trace.json").read_text(encoding="utf-8"))
        assert self.trace["learned_merges"] == [["流", "星"], ["流星", "雨"]]
        self.checkpoints = []
        self.rows: list[list[VGroup]] = []
        self.symbols: list[list[str]] = []
        title = mixed([("从语料到", True), ("Token", False)], 32)
        title.to_edge(LEFT, buff=0.56).to_edge(UP, buff=0.37)
        self.phase_badge = self.phase("01", "训练")
        divider = Line([-6.55, 2.98, 0], [6.55, 2.98, 0], color=GRID, stroke_width=1)
        scope = mixed([("字符级", True), ("BPE", False)], 22, MUTED)
        scope.move_to([-5.45, 2.55, 0])
        self.play(FadeIn(title), FadeIn(self.phase_badge), Create(divider), FadeIn(scope), run_time=0.85)

        vocabulary_text = self.trace["displayed_subset"]
        self.vocabulary = [token(s, COLORS[s], width=max(1.50, 0.65 * len(s) + 0.85)).scale(0.68)
                           for s in vocabulary_text]
        self.vocab_positions = centered_positions(self.vocabulary, y=-2.74, buff=0.48)
        hero = self.vocabulary[:3]
        for obj, pos in zip(hero, centered_positions(hero, y=0.30, buff=0.65), strict=True):
            obj.move_to(pos).scale(1.45)
        self.play(AnimationGroup(*(FadeIn(obj, shift=UP * 0.15) for obj in hero), lag_ratio=0.16), run_time=1.0)
        self.wait(0.9)
        self.mark("opening_characters")

        self.vocab_caption = label("词表节选", 22, MUTED, cjk=True).move_to([-5.92, -2.18, 0])
        self.vocab_line = Line([-6.45, -1.99, 0], [6.45, -1.99, 0], color=GRID, stroke_width=1)
        source_note = label("五条语料 · 各出现一次", 22, MUTED, cjk=True).move_to([-3.68, 2.20, 0])
        raw = [label(row["text"], 32, FOREGROUND, cjk=True) for row in self.trace["initial_corpus"]]
        for obj, y in zip(raw, CORPUS_YS, strict=True):
            obj.move_to([CORPUS_LEFT + obj.width / 2, y, 0])
        self.play(
            *(obj.animate.scale(1 / 1.45).move_to(pos)
              for obj, pos in zip(hero, self.vocab_positions[:3], strict=True)),
            FadeIn(self.vocab_caption), Create(self.vocab_line), FadeIn(source_note),
            AnimationGroup(*(FadeIn(obj, shift=RIGHT * 0.12) for obj in raw), lag_ratio=0.10),
            run_time=1.35,
        )
        self.wait(1.35)
        self.mark("original_five_strings")
        for row, y in zip(self.trace["initial_corpus"], CORPUS_YS, strict=True):
            symbols = row["tokens"]
            items = [cell(s) for s in symbols]
            for item, pos in zip(items, row_positions(items, y), strict=True):
                item.move_to(pos)
            self.symbols.append(list(symbols))
            self.rows.append(items)
        split_moves = []
        for original, row in zip(raw, self.rows, strict=True):
            assert len(original) == len(row)
            for glyph, item in zip(original, row, strict=True):
                split_moves.extend([ReplacementTransform(glyph, item[1]), FadeIn(item[0])])
        self.play(*split_moves, run_time=1.25)
        for original, row in zip(raw, self.rows, strict=True):
            self.remove(original, *(part for item in row for part in item))
            self.add(*row)
        self.wait(0.8)
        self.mark("character_segmentation")

        for round_data in self.trace["rounds"]:
            self.show_training_round(round_data)

        # Vocabulary IDs are arbitrary labels assigned after learning the merges.
        new_badge = self.phase("02", "编号")
        mapping_note = mixed([("教学编号 · 完整词表", True), (str(self.trace["vocabulary_size"]), False), ("项", True)], 24, MUTED)
        mapping_note.move_to([0, 2.2, 0])
        self.play(
            *(FadeOut(item) for row in self.rows for item in row),
            FadeOut(source_note), FadeOut(scope), FadeOut(self.focus_label),
            FadeOut(self.focus_result), FadeOut(self.count), FadeOut(self.tie_note),
            ReplacementTransform(self.phase_badge, new_badge), FadeIn(mapping_note),
            run_time=1.0,
        )
        self.phase_badge = new_badge
        self.ids = []
        self.id_lines = []
        for text, obj in zip(vocabulary_text, self.vocabulary, strict=True):
            number = self.trace["token_to_id"][text]
            num = label(str(number), 29, ACCENT).move_to([obj.get_center()[0], -3.52, 0])
            line = Line(obj.get_bottom() + DOWN * .06, num.get_top() + UP * .07, color=GRID, stroke_width=1.3)
            self.ids.append(num)
            self.id_lines.append(line)
        id_title = label("ID", 22, ACCENT).move_to([-5.92, -3.52, 0])
        self.play(AnimationGroup(*(AnimationGroup(Create(line), FadeIn(num, shift=UP * .08))
                                   for line, num in zip(self.id_lines, self.ids, strict=True)),
                                 lag_ratio=0.22), FadeIn(id_title), run_time=1.9)
        # Use copies in the center so the five retained entries remain visible.
        summary = mixed([("词表保留每个基础字符", True)], 29, FOREGROUND).move_to([0, .90, 0])
        summary2 = label("合并后，新增组合", 25, MUTED, cjk=True).move_to([0, .19, 0])
        self.play(FadeIn(summary), FadeIn(summary2), run_time=.6)
        self.wait(2.0)
        self.mark("vocabulary_ids_1_to_5")

        # Runtime encoding: the vocabulary below never changes again.
        new_badge = self.phase("03", "编码")
        rule_note = label("沿用已学到的两条合并规则", 24, MUTED, cjk=True).move_to([0, 2.20, 0])
        self.play(FadeOut(summary), FadeOut(summary2),
                  ReplacementTransform(mapping_note, rule_note),
                  ReplacementTransform(self.phase_badge, new_badge), run_time=.7)
        query = [token(s, COLORS[s], width=1.55).scale(.85) for s in ("流", "星", "雨")]
        for item, pos in zip(query, centered_positions(query, y=.62, buff=.5), strict=True):
            item.move_to(pos)
        self.play(*(TransformFromCopy(self.vocabulary[i], item) for i, item in enumerate(query)), run_time=1.0)
        self.wait(.65)
        self.mark("encode_characters")
        query = self.merge_query(query, "流星", INTERMEDIATE)
        self.wait(.6)
        self.mark("encode_first_merge")
        query = self.merge_query(query, "流星雨", COMPOSED)
        self.wait(.65)
        self.mark("encode_second_merge")

        lookup_frame = SurroundingRectangle(self.vocabulary[4], color=COMPOSED, buff=.12,
                                           corner_radius=.13, stroke_width=2.0)
        self.play(Create(lookup_frame), Indicate(self.ids[4], color=ACCENT, scale_factor=1.2), run_time=.7)
        final_id = token("5", ACCENT, width=1.55).scale(.95).move_to([1.62, .62, 0])
        final_arrow = Line([.15, .62, 0], [.76, .62, 0], color=GRID, stroke_width=3)
        final_arrow.add_tip(tip_length=.14)
        self.play(query[0].animate.move_to([-1.35, .62, 0]),
                  TransformFromCopy(self.ids[4], final_id), Create(final_arrow), FadeOut(lookup_frame), run_time=1.05)
        result_caption = mixed([("Token", False), ("→", False), ("ID", False)], 22, MUTED).move_to([0, 1.57, 0])
        final_note = label("分词与编号 ≠ 语义理解", 26, ACCENT, cjk=True).move_to([0, -.78, 0])
        self.play(FadeIn(result_caption), FadeIn(final_note, shift=UP * .08), run_time=.65)
        self.wait(3.0)
        self.mark("final_mapping")
        (PROJECT / "timeline.json").write_text(json.dumps({"duration_seconds": float(self.time),
                                                         "checkpoints": self.checkpoints},
                                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def show_training_round(self, data: dict) -> None:
        pair = data["selected_pair"]
        merged_text = "".join(pair)
        occurrences = []
        for row_idx, symbols in enumerate(self.symbols):
            for index in range(len(symbols) - 1):
                if symbols[index:index + 2] == pair:
                    occurrences.append((row_idx, index))
        assert len(occurrences) == data["selected_count"]
        if data["rank"] == 2:
            self.play(FadeOut(self.focus_result), FadeOut(self.focus_label), FadeOut(self.count), run_time=.45)
        self.focus_label = label("相邻对计数", 25, FOREGROUND, cjk=True).move_to([FOCUS[0], 1.55, 0])
        focus_items = [token(s, COLORS[s], width=max(1.55, .65 * len(s) + .85)).scale(.9) for s in pair]
        plus = label("+", 32, MUTED)
        focus_group = VGroup(focus_items[0], plus, focus_items[1]).arrange(RIGHT, buff=.20).move_to(FOCUS)
        first_row, first_idx = occurrences[0]
        source = VGroup(*self.rows[first_row][first_idx:first_idx + 2])
        outlines = [SurroundingRectangle(VGroup(*self.rows[r][i:i+2]), color=ACCENT,
                                         buff=.085, corner_radius=.11, stroke_width=2)
                    for r, i in occurrences]
        self.count = label("× 0", 36, ACCENT).move_to([FOCUS[0], -.60, 0])
        self.play(FadeIn(self.focus_label),
                  TransformFromCopy(self.rows[first_row][first_idx], focus_items[0]),
                  TransformFromCopy(self.rows[first_row][first_idx + 1], focus_items[1]),
                  FadeIn(plus),
                  *(Create(outline) for outline in outlines), FadeIn(self.count), run_time=.85)
        for index, outline in enumerate(outlines, start=1):
            start = outline.get_right() + RIGHT * .06
            end = self.count.get_left() + LEFT * .1
            bead = Dot(start, radius=.055, color=ACCENT)
            path = CubicBezier(start, start + RIGHT * 1.1, end + LEFT * 1.1 + UP * .35, end)
            next_count = label(f"× {index}", 36, ACCENT).move_to(self.count)
            self.add(bead)
            self.play(MoveAlongPath(bead, path), Indicate(outline, color=ACCENT, scale_factor=1.035), run_time=.65)
            self.play(FadeOut(bead), ReplacementTransform(self.count, next_count), run_time=.25)
            self.count = next_count
        if data["rank"] == 2:
            assert len(data["tied_maxima"]) == 15
            self.tie_note = mixed([("并列最高频 ·", True), ("15", False), ("对各一次", True)], 22, ACCENT)
            self.tie_note.move_to([FOCUS[0], -1.22, 0])
            choice = label("本例择「流星 + 雨」", 22, MUTED, cjk=True).move_to([FOCUS[0], -1.64, 0])
            self.tie_note = VGroup(self.tie_note, choice)
            self.play(FadeIn(self.tie_note), run_time=.5)
        self.wait(1.65 if data["rank"] == 2 else 1.0)
        self.mark(f"round_{data['rank']}_count")

        self.play(focus_items[0].animate.shift(RIGHT * .14),
                  focus_items[1].animate.shift(LEFT * .14), FadeOut(plus), run_time=.35)
        self.focus_result = token(merged_text, COLORS[merged_text],
                                  width=.65 * len(merged_text) + .85).scale(.9).move_to(FOCUS)
        fusions = [(*focus_items, self.focus_result)]
        animations = self.fusion_moves(*focus_items, self.focus_result)
        rebuilt_rows = []
        for row_idx, row in enumerate(self.rows):
            symbols = self.symbols[row_idx]
            next_symbols = []
            next_row = []
            index = 0
            pending = []
            while index < len(symbols):
                if symbols[index:index+2] == pair:
                    result = cell(merged_text)
                    pending.append((row[index], row[index + 1], result))
                    next_row.append(result)
                    next_symbols.append(merged_text)
                    index += 2
                else:
                    next_row.append(row[index])
                    next_symbols.append(symbols[index])
                    index += 1
            positions = row_positions(next_row, CORPUS_YS[row_idx])
            fresh = [new for _, _, new in pending]
            for obj, pos in zip(next_row, positions, strict=True):
                if obj in fresh:
                    obj.move_to(pos)
                elif not np.allclose(obj.get_center(), pos):
                    animations.append(obj.animate.move_to(pos))
            for left, right, result in pending:
                animations.extend(self.fusion_moves(left, right, result))
                fusions.append((left, right, result))
            self.symbols[row_idx] = next_symbols
            rebuilt_rows.append(next_row)
        animations.extend(FadeOut(outline) for outline in outlines)
        self.play(*animations, run_time=1.0, rate_func=rate_functions.ease_in_out_cubic)
        for left, right, result in fusions:
            self.finish_fusion(left, right, result)
        self.rows = rebuilt_rows
        assert self.symbols == [row["tokens"] for row in data["corpus_after"]]
        self.play(Indicate(self.focus_result[0], color=COLORS[merged_text], scale_factor=1.045), run_time=.4)
        dest_index = self.trace["displayed_subset"].index(merged_text)
        dest = self.vocabulary[dest_index].move_to(self.vocab_positions[dest_index])
        self.play(TransformFromCopy(self.focus_result, dest), run_time=.9)
        self.wait(.9)
        self.mark(f"round_{data['rank']}_merged")

    def merge_query(self, items: list[VGroup], text: str, color: str) -> list[VGroup]:
        pair = VGroup(*items[:2])
        focus = SurroundingRectangle(pair, color=ACCENT, buff=.1, corner_radius=.12, stroke_width=2)
        self.play(Create(focus), run_time=.3)
        self.play(items[0].animate.shift(RIGHT*.1), items[1].animate.shift(LEFT*.1), run_time=.3)
        merged = token(text, color, width=.65 * len(text)+.85).scale(.85)
        remaining = [merged, *items[2:]]
        positions = centered_positions(remaining, y=.62, buff=.5)
        merged.move_to(positions[0])
        self.play(*self.fusion_moves(items[0], items[1], merged),
                  *(obj.animate.move_to(pos) for obj, pos in zip(remaining[1:], positions[1:], strict=True)),
                  FadeOut(focus), run_time=.85, rate_func=rate_functions.ease_in_out_cubic)
        self.finish_fusion(items[0], items[1], merged)
        return remaining

    def fusion_moves(self, left: VGroup, right: VGroup, result: VGroup) -> list:
        """Merge boundaries while translating intact glyphs, never morphing them."""

        for glyph in (left[1], right[1], result[1]):
            glyph.set_z_index(2)
        # Chinese glyph advances are additive; measured visible widths may omit
        # side bearings. A small natural gap keeps the characters distinct.
        gap = max(0.0, result[1].width - left[1].width - right[1].width)
        center = result[1].get_center()
        left_target = center + LEFT * (right[1].width + gap) / 2
        right_target = center + RIGHT * (left[1].width + gap) / 2
        return [
            ReplacementTransform(left[0], result[0]),
            right[0].animate.move_to(result[0]).set_opacity(0),
            left[1].animate.move_to(left_target),
            right[1].animate.move_to(right_target),
        ]

    def finish_fusion(self, left: VGroup, right: VGroup, result: VGroup) -> None:
        """Consolidate the moved glyphs into one stable token after the motion."""

        self.remove(left, right, result[0], left[1], right[1], right[0])
        self.add(result)
