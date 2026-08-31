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
    Indicate,
    LEFT,
    Line,
    MovingCameraScene,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    SurroundingRectangle,
    UP,
    VGroup,
    Write,
    rate_functions,
)

from manim_style import (
    ACCENT,
    BASE as BYTE,
    COMPOSED as PHRASE,
    GRID,
    INTERMEDIATE as CHARACTER,
    MUTED,
    centered_positions,
    label,
    pill,
    token,
)


class BPEByteMerges(MovingCameraScene):
    """Animate frozen byte-level BPE rules as one continuous computation."""

    def construct(self) -> None:
        title = label("BYTE-LEVEL BPE", 34)
        title.to_edge(UP, buff=0.38).to_edge(LEFT, buff=0.56)

        frozen = pill("FROZEN RULES", ACCENT)
        frozen.to_edge(UP, buff=0.48).to_edge(RIGHT, buff=0.58)

        divider = Line(LEFT * 6.55, RIGHT * 6.55, color=GRID, stroke_width=1.2)
        divider.next_to(title, DOWN, buff=0.28)

        scale_line = Line(LEFT * 4.25, RIGHT * 4.25, color=GRID, stroke_width=2.0)
        scale_line.move_to(DOWN * 2.55)
        scale_points = VGroup()
        scale_labels = VGroup()
        for x, text, color in (
            (-4.25, "BYTES", BYTE),
            (0.0, "CHARACTER", CHARACTER),
            (4.25, "PHRASE", PHRASE),
        ):
            dot = RoundedRectangle(
                corner_radius=0.09,
                width=0.18,
                height=0.18,
                stroke_width=0,
                fill_color=color,
                fill_opacity=0.95,
            ).move_to([x, -2.55, 0])
            text_label = label(text, 18, MUTED)
            text_label.next_to(dot, DOWN, buff=0.16)
            scale_points.add(dot)
            scale_labels.add(text_label)
        self.play(
            Write(title),
            FadeIn(frozen),
            Create(divider),
            run_time=1.0,
        )
        self.play(
            Create(scale_line),
            FadeIn(scale_points),
            FadeIn(scale_labels),
            run_time=0.9,
        )
        self.play(Indicate(scale_points[0], color=BYTE, scale_factor=1.65), run_time=0.45)

        glyph = token("数", CHARACTER, width=1.6)
        glyph.move_to(UP * 0.25)
        self.play(FadeIn(glyph, scale=0.82), run_time=0.7)
        self.wait(0.35)

        utf8 = pill("UTF-8", BYTE)
        utf8.move_to(UP * 1.65)
        encode_arrow = Arrow(
            utf8.get_bottom(),
            glyph.get_top(),
            color=BYTE,
            buff=0.12,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.16,
        )
        tokens = [token("E6", BYTE), token("95", BYTE), token("B0", BYTE)]
        for item, position in zip(tokens, centered_positions(tokens), strict=True):
            item.move_to(position)
        bytes_row = VGroup(*tokens)
        self.play(FadeIn(utf8), GrowArrow(encode_arrow), run_time=0.55)
        self.play(
            ReplacementTransform(glyph, bytes_row),
            FadeOut(encode_arrow),
            run_time=1.0,
        )
        self.wait(0.4)
        self.play(FadeOut(utf8), run_time=0.3)

        tokens = self.merge_pair(tokens, "E6 95", BYTE)
        tokens = self.merge_pair(tokens, "数", CHARACTER)
        self.play(Indicate(scale_points[1], color=CHARACTER, scale_factor=1.65), run_time=0.45)

        additions = [token("据", CHARACTER), token("库", CHARACTER)]
        phrase_tokens = [tokens[0], *additions]
        phrase_positions = centered_positions(phrase_tokens)
        for item, position in zip(additions, phrase_positions[1:], strict=True):
            item.move_to(position)
        self.play(
            tokens[0].animate.move_to(phrase_positions[0]),
            AnimationGroup(
                *(FadeIn(item, shift=RIGHT * 0.24) for item in additions),
                lag_ratio=0.18,
            ),
            run_time=0.8,
        )
        tokens = phrase_tokens

        tokens = self.merge_pair(tokens, "数据", PHRASE)
        tokens = self.merge_pair(tokens, "数据库", PHRASE)
        self.play(Indicate(scale_points[2], color=PHRASE, scale_factor=1.65), run_time=0.45)

        coverage = pill("BYTES → COVERAGE", BYTE)
        compression = pill("MERGES → COMPRESSION", PHRASE)
        takeaway = VGroup(coverage, compression).arrange(RIGHT, buff=0.34)
        takeaway.move_to(DOWN * 1.85)
        caveat = label("≠  UNDERSTANDING", 21, ACCENT)
        caveat.next_to(takeaway, DOWN, buff=0.26)
        self.play(
            FadeOut(scale_line),
            FadeOut(scale_points),
            FadeOut(scale_labels),
            FadeIn(takeaway, shift=UP * 0.12),
            FadeIn(caveat, shift=UP * 0.12),
            run_time=0.9,
        )
        self.wait(1.8)

    def merge_pair(self, tokens: list[VGroup], result: str, color: str) -> list[VGroup]:
        """Move the first adjacent pair together and fuse it into one token."""

        left, right, *survivors = tokens
        pair = VGroup(left, right)
        focus = SurroundingRectangle(
            pair,
            color=ACCENT,
            buff=0.12,
            corner_radius=0.14,
            stroke_width=2.0,
        )
        self.play(Create(focus), run_time=0.25)
        self.play(
            left.animate.shift(RIGHT * 0.12),
            right.animate.shift(LEFT * 0.12),
            run_time=0.35,
            rate_func=rate_functions.ease_in_out_cubic,
        )

        merged = token(result, color)
        new_tokens = [merged, *survivors]
        positions = centered_positions(new_tokens)
        merged.move_to(positions[0])
        survivor_moves = [
            item.animate.move_to(position)
            for item, position in zip(survivors, positions[1:], strict=True)
        ]
        self.play(
            ReplacementTransform(pair, merged),
            *survivor_moves,
            FadeOut(focus),
            run_time=0.8,
            rate_func=rate_functions.ease_in_out_cubic,
        )
        self.play(Indicate(merged, color=color, scale_factor=1.04), run_time=0.35)
        self.wait(0.22)
        return new_tokens
