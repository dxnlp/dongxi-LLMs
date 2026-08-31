"""Continuous Manim explanation of byte-level BPE coverage and compression.

The scene deliberately shows encoding replaying a frozen rulebook.  It does not
present the merge sequence as tokenizer training happening on the current input.
"""

from __future__ import annotations

from manim import (
    AnimationGroup,
    Arrow,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    LEFT,
    Line,
    MovingCameraScene,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Text,
    Transform,
    UP,
    VGroup,
    WHITE,
    Write,
    config,
    rate_functions,
)


BACKGROUND = "#0B1020"
FOREGROUND = "#F4F7FF"
MUTED = "#94A3B8"
GRID = "#334155"
BYTE = "#3B82F6"
CHARACTER = "#8B5CF6"
PHRASE = "#10B981"
ACCENT = "#F59E0B"
RULE_ACTIVE = "#FBBF24"

LATIN_FONT = "Avenir Next"
CJK_FONT = "PingFang SC"

config.background_color = BACKGROUND


def label(text: str, size: int, color: str = FOREGROUND, *, cjk: bool = False) -> Text:
    return Text(
        text,
        font=CJK_FONT if cjk else LATIN_FONT,
        font_size=size,
        color=color,
        weight="MEDIUM",
    )


def token(text: str, color: str, *, width: float | None = None) -> VGroup:
    """Return a large, phone-legible token with a stable color identity."""

    token_width = width or max(1.45, 0.44 * len(text) + 0.78)
    box = RoundedRectangle(
        corner_radius=0.16,
        height=1.12,
        width=token_width,
        stroke_color=WHITE,
        stroke_width=2.2,
        fill_color=color,
        fill_opacity=0.94,
    )
    text_label = label(text, 38, cjk=any(ord(char) > 127 for char in text))
    text_label.move_to(box)
    return VGroup(box, text_label)


def token_row(spec: list[tuple[str, str]]) -> VGroup:
    row = VGroup(*(token(text, color) for text, color in spec))
    row.arrange(RIGHT, buff=0.24)
    row.move_to(LEFT * 1.65 + UP * 0.15)
    return row


def pill(text: str, color: str) -> VGroup:
    text_label = label(text, 22, color)
    box = RoundedRectangle(
        corner_radius=0.2,
        width=text_label.width + 0.48,
        height=0.52,
        stroke_color=color,
        stroke_width=1.5,
        fill_color=color,
        fill_opacity=0.08,
    )
    text_label.move_to(box)
    return VGroup(box, text_label)


class BPEByteMerges(MovingCameraScene):
    """Animate frozen byte-level BPE rules as one continuous computation."""

    def construct(self) -> None:
        title = label("BYTE-LEVEL BPE", 34)
        title.to_edge(UP, buff=0.38).to_edge(LEFT, buff=0.56)
        subtitle = label("coverage is guaranteed · compression is learned", 23, MUTED)
        subtitle.next_to(title, DOWN, aligned_edge=LEFT, buff=0.08)

        frozen = pill("FROZEN ENCODER", ACCENT)
        frozen.to_edge(UP, buff=0.48).to_edge(RIGHT, buff=0.58)

        divider = Line(LEFT * 6.55, RIGHT * 6.55, color=GRID, stroke_width=1.2)
        divider.next_to(subtitle, DOWN, buff=0.28)

        rule_title = label("MERGE RANKS", 20, MUTED)
        rule_title.move_to(RIGHT * 4.78 + UP * 1.95)
        rules = VGroup(
            label("01   E6 + 95  →  E6 95", 21, MUTED),
            label("02   E6 95 + B0  →  数", 21, MUTED, cjk=True),
            label("03   数 + 据  →  数据", 21, MUTED, cjk=True),
            label("04   数据 + 库  →  数据库", 21, MUTED, cjk=True),
        )
        rules.arrange(DOWN, aligned_edge=LEFT, buff=0.27)
        rules.next_to(rule_title, DOWN, aligned_edge=LEFT, buff=0.22)
        rule_panel = RoundedRectangle(
            corner_radius=0.18,
            width=4.2,
            height=3.35,
            stroke_color=GRID,
            stroke_width=1.4,
            fill_color="#111A2E",
            fill_opacity=0.82,
        )
        rule_panel.move_to(RIGHT * 4.55 + UP * 0.35)
        rule_title.move_to(rule_panel.get_top() + DOWN * 0.36)
        if rules.width > rule_panel.width - 0.42:
            rules.scale_to_fit_width(rule_panel.width - 0.42)
        rules.next_to(rule_title, DOWN, buff=0.24)
        rules.set_x(rule_panel.get_center()[0])
        offline_note = label("learned offline · replayed at runtime", 17, ACCENT)
        offline_note.next_to(rule_panel.get_bottom(), UP, buff=0.24)

        scale_line = Line(LEFT * 4.7, RIGHT * 4.7, color=GRID, stroke_width=2.0)
        scale_line.move_to(DOWN * 2.72 + LEFT * 0.55)
        scale_points = VGroup()
        scale_labels = VGroup()
        for x, text, color in (
            (-5.25, "BYTES\ncoverage", BYTE),
            (-0.55, "CHARACTERS\ncompression", CHARACTER),
            (4.15, "PHRASES\ncompression", PHRASE),
        ):
            dot = RoundedRectangle(
                corner_radius=0.09,
                width=0.18,
                height=0.18,
                stroke_width=0,
                fill_color=color,
                fill_opacity=0.95,
            ).move_to([x, -2.72, 0])
            text_label = label(text, 17, MUTED)
            text_label.next_to(dot, DOWN, buff=0.13)
            scale_points.add(dot)
            scale_labels.add(text_label)
        progress = RoundedRectangle(
            corner_radius=0.08,
            width=0.22,
            height=0.22,
            stroke_width=0,
            fill_color=BYTE,
            fill_opacity=1.0,
        ).move_to(scale_points[0])

        self.play(
            Write(title),
            FadeIn(subtitle, shift=UP * 0.08),
            FadeIn(frozen),
            Create(divider),
            run_time=1.0,
        )
        self.play(
            FadeIn(rule_panel),
            FadeIn(rule_title),
            FadeIn(rules),
            FadeIn(offline_note),
            Create(scale_line),
            FadeIn(scale_points),
            FadeIn(scale_labels),
            FadeIn(progress),
            run_time=0.9,
        )

        glyph = token("数", CHARACTER, width=1.6)
        glyph.move_to(LEFT * 1.65 + UP * 0.48)
        glyph_caption = label("one visible glyph", 24, MUTED)
        glyph_caption.next_to(glyph, DOWN, buff=0.24)
        self.play(FadeIn(glyph, scale=0.85), FadeIn(glyph_caption), run_time=0.8)
        self.wait(0.5)

        utf8 = pill("UTF-8", BYTE)
        utf8.move_to(LEFT * 1.65 + UP * 1.53)
        encode_arrow = Arrow(
            utf8.get_bottom(),
            glyph.get_top(),
            color=BYTE,
            buff=0.12,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.16,
        )
        bytes_row = token_row([("E6", BYTE), ("95", BYTE), ("B0", BYTE)])
        byte_caption = label("three reusable byte tokens", 24, MUTED)
        byte_caption.next_to(bytes_row, DOWN, buff=0.24)
        self.play(FadeIn(utf8), GrowArrow(encode_arrow), run_time=0.55)
        self.play(
            ReplacementTransform(glyph, bytes_row),
            Transform(glyph_caption, byte_caption),
            FadeOut(encode_arrow),
            run_time=1.0,
        )
        self.wait(0.5)

        tokens = bytes_row
        caption = glyph_caption
        self.play(FadeOut(utf8), run_time=0.35)
        tokens, caption = self.merge_step(
            tokens,
            [("E6 95", BYTE), ("B0", BYTE)],
            caption,
            "rule 01 compresses two bytes",
            rules,
            0,
        )
        tokens, caption = self.merge_step(
            tokens,
            [("数", CHARACTER)],
            caption,
            "rule 02 completes one character token",
            rules,
            1,
        )
        self.play(
            progress.animate.move_to(scale_points[1]).set_fill(CHARACTER),
            run_time=0.6,
        )

        phrase_start = token_row([("数", CHARACTER), ("据", CHARACTER), ("库", CHARACTER)])
        phrase_caption = label("the same frozen encoder continues", 24, MUTED)
        phrase_caption.next_to(phrase_start, DOWN, buff=0.24)
        self.play(
            Transform(tokens, phrase_start),
            Transform(caption, phrase_caption),
            run_time=1.0,
        )

        tokens, caption = self.merge_step(
            tokens,
            [("数据", PHRASE), ("库", CHARACTER)],
            caption,
            "rule 03 compresses a frequent neighbor pair",
            rules,
            2,
        )
        tokens, caption = self.merge_step(
            tokens,
            [("数据库", PHRASE)],
            caption,
            "rule 04 compresses the complete expression",
            rules,
            3,
        )
        self.play(
            progress.animate.move_to(scale_points[2]).set_fill(PHRASE),
            run_time=0.6,
        )
        self.play(*[rule.animate.set_color(MUTED) for rule in rules], run_time=0.35)

        takeaway = VGroup(
            label("BASE BYTES GUARANTEE COVERAGE", 22, FOREGROUND),
            label("CORPUS-LEARNED MERGES REDUCE TOKEN COUNT", 22, FOREGROUND),
        )
        takeaway.arrange(DOWN, buff=0.12)
        takeaway.move_to(DOWN * 2.35)
        caveat = label("encodable ≠ understood", 23, ACCENT)
        caveat.next_to(takeaway, DOWN, buff=0.18)
        self.play(
            FadeOut(caption),
            FadeOut(scale_line),
            FadeOut(scale_points),
            FadeOut(scale_labels),
            FadeOut(progress),
            FadeIn(takeaway, shift=UP * 0.12),
            FadeIn(caveat, shift=UP * 0.12),
            run_time=0.9,
        )
        self.wait(1.7)

    def merge_step(
        self,
        old_tokens: VGroup,
        new_spec: list[tuple[str, str]],
        old_caption: Text,
        caption_text: str,
        rules: VGroup,
        active_rule: int,
    ) -> tuple[VGroup, Text]:
        """Morph a row in place while highlighting one pre-existing rule."""

        new_tokens = token_row(new_spec)
        new_caption = label(caption_text, 24, MUTED)
        new_caption.next_to(new_tokens, DOWN, buff=0.24)

        rule_animations = []
        for index, rule in enumerate(rules):
            rule_animations.append(
                rule.animate.set_color(RULE_ACTIVE if index == active_rule else MUTED)
            )

        apply_rule = pill(f"APPLY RULE {active_rule + 1:02d}  ↓", ACCENT)
        apply_rule.next_to(old_tokens, UP, buff=0.34)

        self.play(
            AnimationGroup(*rule_animations, lag_ratio=0),
            FadeIn(apply_rule, shift=DOWN * 0.06),
            run_time=0.45,
        )
        self.play(
            ReplacementTransform(old_tokens, new_tokens),
            Transform(old_caption, new_caption),
            FadeOut(apply_rule),
            run_time=1.05,
            rate_func=rate_functions.ease_in_out_cubic,
        )
        self.wait(0.55)
        return new_tokens, old_caption
