"""How chatbots learn from feedback: PPO, GRPO and DPO for beginners, told with pictures.

Render with render.py. ONLY=ppo,dpo (comma list of section names) renders just those
sections for quick iteration; the timeline is written only for full renders.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from manim import (
    DOWN, LEFT, PI, RIGHT, UP, Create, CurvedArrow, DashedLine, Dot, FadeIn, FadeOut,
    FadeTransform, Indicate, LaggedStart, Line, Rectangle, Rotate, Scene, Transform,
    ValueTracker, VGroup, always_redraw, config, rate_functions, Arrow, Circumscribe,
)

from kit import (
    BAD, BAD_FILL, BG, CARD, COACH, FAINT, GOOD, GOOD_FILL, INK, JUDGE, LEASH, ROUND, SNAP,
    SNAP_FILL, SOFT, STUDENT, STUDENT_FILL, WHEEL_COLORS, T, answer_card, balance,
    check_icon, clipboard, coach, cross_icon, dashed_tile, eye_panel, family_card, judge,
    leash_curve, memory_panel, person, pointer, polaroid, prompt_card, scorecard,
    section_header, slice_mid_angles, star, stepper, student, tag, tile, tile_row, wheel,
    wheel_labels, bowl,
)

HERE = Path(__file__).resolve().parent
FX = json.loads((HERE / "fixture.json").read_text())
config.background_color = BG
CAP_Y = -3.38
LEFT_X = -5.9
FAMILIES = [("Practice with a coach", "PPO", "online RL with a critic"),
            ("Practice and compare", "GRPO", "online policy gradient, no critic"),
            ("Learn from comparisons", "DPO", "direct preference optimization")]


def signed(v: float) -> str:
    if abs(v) < 1e-9:
        return "0"
    s = f"{abs(v):g}"
    return ("+" if v > 0 else "−") + s


def lerp(a, b, f):
    return [x + (y - x) * f for x, y in zip(a, b)]


def keep_in(m, margin=0.12):
    hw = config.frame_width / 2 - margin
    if m.get_left()[0] < -hw:
        m.shift(RIGHT * (-hw - m.get_left()[0]))
    if m.get_right()[0] > hw:
        m.shift(LEFT * (m.get_right()[0] - hw))
    return m


class BeginnerFeedback(Scene):
    def setup(self):
        self.events, self.caption, self.head, self.steps = [], None, None, None

    # ------------------------------------------------------------------ plumbing
    def mark(self, name, **data):
        hw, hh = config.frame_width / 2 + 1e-3, config.frame_height / 2 + 1e-3
        for m in self.mobjects:
            if isinstance(m, ValueTracker) or not m.get_family():
                continue
            l, r = m.get_left()[0], m.get_right()[0]
            b, t = m.get_bottom()[1], m.get_top()[1]
            assert -hw <= l and r <= hw and -hh <= b and t <= hh, (name, type(m).__name__,
                                                                    l, r, b, t)
        self.events.append(dict(name=name, t=round(self.renderer.time, 3), **data))

    def cap(self, text, t2c=None):
        new = T(text, 28, INK, t2c=t2c).move_to([0, CAP_Y, 0])
        assert new.width < config.frame_width - 0.8, text
        anims = [FadeIn(new, shift=UP * 0.12)]
        if self.caption is not None:
            anims.append(FadeOut(self.caption, shift=UP * 0.12))
        self.caption = new
        return anims

    def say(self, text, *extra, t2c=None, run_time=0.6):
        self.play(*self.cap(text, t2c), *extra, run_time=run_time)

    def open_section(self, num, title, method, sub, steps=True):
        head = section_header(num, title, method, sub)
        anims = [FadeIn(head, shift=DOWN * 0.12)]
        if self.head is not None:
            anims.append(FadeOut(self.head))
        if self.steps is not None:
            anims.append(FadeOut(self.steps))
        self.steps = stepper(None) if steps else None
        if steps:
            anims.append(FadeIn(self.steps))
        self.head = head
        self.play(*anims, run_time=0.7)

    def step(self, i, *extra):
        self.play(Transform(self.steps, stepper(i)), *extra, run_time=0.4)

    def clear_stage(self, *keep, header=False, run_time=0.6):
        keep = set(keep) | {self.caption}
        if not header:
            keep |= {self.head, self.steps}
        gone = [m for m in self.mobjects if m not in keep and not isinstance(m, ValueTracker)]
        self.remove(*[m for m in self.mobjects if isinstance(m, ValueTracker)])
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)
        if header:
            self.head = self.steps = None

    def rescore(self, card, value, color, *extra, run_time=0.5):
        """Swap a score card's number with a crossfade; returns the card now in the scene."""
        new = scorecard(value, color, card.s).move_to(card)
        self.play(FadeTransform(card, new), *extra, run_time=run_time)
        return new

    def pulse(self, mob, k=1.12, run_time=0.6):
        self.play(mob.animate.scale(k), rate_func=rate_functions.there_and_back,
                  run_time=run_time)

    def blink(self, bot):
        self.play(bot.eyes.animate.stretch(0.12, 1), run_time=0.08)
        self.play(bot.eyes.animate.stretch(1 / 0.12, 1), run_time=0.1)

    def left_team(self, bot_y=0.25, pol_y=-1.8, s=0.72):
        bot = student(s).move_to([LEFT_X, bot_y, 0])
        pol = polaroid(0.42).move_to([LEFT_X, pol_y, 0])
        rope = always_redraw(lambda: leash_curve(pol.get_top() + DOWN * 0.08,
                                                 bot.body.get_bottom() + LEFT * 0.25,
                                                 bend=-0.28))
        return bot, pol, rope

    # ------------------------------------------------------------------ story
    def construct(self):
        only = os.environ.get("ONLY")
        parts = [("intro", self.intro), ("basics", self.basics), ("feedback", self.feedback),
                 ("leash", self.leash), ("families", self.families), ("ppo", self.ppo),
                 ("grpo", self.grpo), ("dpo", self.dpo), ("summary", self.summary)]
        for name, fn in parts:
            if not only or name in only.split(","):
                fn()
        if not only:
            (HERE / "media").mkdir(exist_ok=True)
            (HERE / "media" / "timeline.json").write_text(json.dumps(
                dict(duration=round(self.renderer.time, 3), events=self.events), indent=2))

    def intro(self):
        bot = student(1.3).move_to([0, 1.3, 0])
        title = T("How do chatbots learn from feedback?", 46, INK, weight="BOLD")
        title.move_to([0, -0.55, 0])
        sub = T("PPO, GRPO and DPO, explained with pictures", 25, SOFT).move_to([0, -1.4, 0])
        self.play(FadeIn(bot, shift=UP * 0.3, scale=0.85), run_time=0.8)
        self.play(FadeIn(title, shift=UP * 0.15), run_time=0.7)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.5)
        self.mark("title")
        self.wait(0.8)
        self.blink(bot)
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub), run_time=0.5)
        self.bot = bot

    def basics(self):
        bot = getattr(self, "bot", None) or student(1.3).move_to([0, 1.3, 0])
        self.add(bot)
        self.open_section(None, "The basics", None, "what a language model does", steps=False)
        q = prompt_card("What is the capital of Australia?").move_to([0.9, 2.2, 0])
        self.play(bot.animate.scale(0.72 / 1.3).move_to([LEFT_X, 0.95, 0]),
                  FadeIn(q, shift=DOWN * 0.15),
                  *self.cap("A language model writes its answer one word at a time."),
                  run_time=0.9)
        policy = keep_in(tag("also called the policy", STUDENT).next_to(bot, DOWN, buff=0.3))
        row = tile_row(FX["ppo"]["answer"][:5], -3.75, 0.95)
        blank = dashed_tile().next_to(row, RIGHT, buff=0.14)
        self.play(FadeIn(policy, shift=UP * 0.1), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.25) for t in row], lag_ratio=0.5),
                  run_time=1.8)
        self.play(FadeIn(blank), run_time=0.35)
        self.mark("words")
        self.wait(0.9)

        shares, names = FX["wheel"]["shares"], FX["wheel"]["words"]
        wc = np.array([blank.get_center()[0] - 0.2, -1.35, 0])
        R = 1.0
        wh = wheel(shares, WHEEL_COLORS, R).move_to(wc)
        ptr = pointer(R).shift(wc)
        labs = wheel_labels(shares, names, R, center=wc)
        toy = tag("toy numbers").move_to(wc + RIGHT * 3.2 + DOWN * 0.95)
        self.say("To choose the next word, it spins a wheel of chances.",
                 FadeIn(wh, scale=0.85), FadeIn(ptr), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(lb) for lb in labs], lag_ratio=0.25), FadeIn(toy),
                  run_time=1.0)
        self.mark("wheel", shares=shares)
        self.wait(1.8)
        mid = slice_mid_angles(shares)[0]
        spin = 6 * PI + (mid - PI / 2)
        self.play(FadeOut(labs), run_time=0.3)
        self.play(Rotate(wh, angle=-spin, about_point=wc), rate_func=rate_functions.ease_out_cubic,
                  run_time=3.0)
        word = tile("Sydney").move_to(wc + UP * 0.35)
        self.play(FadeIn(word, scale=0.5), Indicate(ptr, color=BAD, scale_factor=1.3),
                  run_time=0.5)
        self.play(word.animate.move_to(blank).align_to(blank, LEFT), FadeOut(blank),
                  run_time=0.7)
        self.mark("spun", landed="Sydney")

        self.say("It learned from internet text, and that text has mistakes.",
                 FadeOut(wh), FadeOut(ptr), FadeOut(toy), run_time=0.7)
        x = cross_icon().next_to(word, RIGHT, buff=0.22)
        right = tile("Canberra", GOOD, GOOD_FILL).next_to(word, DOWN, buff=0.7)
        ok = check_icon().next_to(right, RIGHT, buff=0.22)
        note = T("correct answer", 19, GOOD).next_to(right, DOWN, buff=0.16)
        self.play(word.box.animate.set_stroke(BAD), Create(x), run_time=0.5)
        self.play(FadeIn(right, shift=UP * 0.2), Create(ok), FadeIn(note), run_time=0.6)
        self.mark("mistake")
        self.wait(2.0)

    def feedback(self):
        bot = getattr(self, "bot", None) or student(0.72).move_to([LEFT_X, 0.95, 0])
        self.clear_stage(bot, header=True)
        self.open_section(None, "Learning from feedback", None, "why a judge is needed",
                          steps=False)
        good = bowl(True, 1.05).move_to([-1.9, 0.55, 0])
        bad = bowl(False, 1.05).move_to([1.9, 0.35, 0])
        self.say("Writing a great answer is hard. Judging one is easier.",
                 bot.animate.move_to([LEFT_X, 0.4, 0]),
                 FadeIn(good, shift=UP * 0.2), FadeIn(bad, shift=UP * 0.2), run_time=0.8)
        self.wait(1.8)
        stars = VGroup(*[star(LEASH, 0.17) for _ in range(5)]).arrange(RIGHT, buff=0.07)
        stars.next_to(good, DOWN, buff=0.45)
        meh = VGroup(*[star(LEASH if i < 2 else FAINT, 0.17) for i in range(5)])
        meh.arrange(RIGHT, buff=0.07).next_to(bad, DOWN, buff=0.45).match_y(stars)
        self.say("You don't need to be a chef to tell which soup is better.",
                 LaggedStart(FadeIn(stars, scale=0.6), FadeIn(meh, scale=0.6), lag_ratio=0.4),
                 run_time=1.0)
        self.mark("soup")
        self.wait(2.0)

        people = VGroup(*[person(s=0.8) for _ in range(3)]).arrange(RIGHT, buff=0.3)
        people.move_to([-2.9, 0.35, 0])
        ca = answer_card("A", "Canberra.", True, width=3.3).move_to([1.3, 0.95, 0])
        cb = answer_card("B", "Sydney.", False, width=3.3).move_to([1.3, -0.25, 0])
        self.say("People compare two answers and pick the better one.",
                 FadeOut(good), FadeOut(bad), FadeOut(stars), FadeOut(meh),
                 FadeIn(people, shift=UP * 0.2), FadeIn(ca), FadeIn(cb), run_time=0.8)
        self.wait(2.0)
        jd = judge(0.8).move_to([5.3, 0.35, 0])
        rm = tag("reward model", JUDGE).next_to(jd, DOWN, buff=0.3)
        self.say("A judge model learns their taste, then scores any answer.",
                 FadeIn(jd, shift=LEFT * 0.3), run_time=0.7)
        self.play(FadeIn(rm, shift=UP * 0.1), run_time=0.4)
        sa = scorecard("9", GOOD, 0.62).next_to(ca, RIGHT, buff=0.2)
        sb = scorecard("2", BAD, 0.62).next_to(cb, RIGHT, buff=0.2)
        self.play(FadeIn(sa, shift=UP * 0.25), run_time=0.45)
        self.play(FadeIn(sb, shift=UP * 0.25), run_time=0.45)
        self.mark("judge")
        self.wait(2.0)
        self.jd = jd

    def leash(self):
        bot = getattr(self, "bot", None) or student(0.72).move_to([LEFT_X, 0.4, 0])
        jd = getattr(self, "jd", None) or judge(0.8).move_to([5.3, 0.35, 0])
        self.add(bot, jd)
        self.clear_stage(bot, jd)
        words, steps = FX["ppo"]["ramble"], FX["ppo"]["ramble_scores"]
        self.say("But judges have quirks. Say this one loves long answers.",
                 bot.animate.move_to([LEFT_X, 0.95, 0]), jd.animate.move_to([5.75, 0.95, 0]),
                 run_time=0.8)
        top = tile_row(words[:6], -4.55, 1.25, size=22, height=0.56)
        bottom = tile_row(words[6:], -4.55, 0.5, size=22, height=0.56)
        tiles = [*top, *bottom]
        score = scorecard(str(steps[0][1]), JUDGE, 0.7).move_to([5.75, 2.3, 0])
        self.play(FadeIn(tiles[0], shift=RIGHT * 0.2), FadeIn(score, shift=UP * 0.2),
                  run_time=0.5)
        shown = 1
        for n, s in steps[1:]:
            score = self.rescore(score, str(s), JUDGE, LaggedStart(
                *[FadeIn(t, shift=RIGHT * 0.2) for t in tiles[shown:n]], lag_ratio=0.35),
                run_time=0.9)
            shown = n
        hack = tag("reward hacking", BAD).move_to([0.9, -0.35, 0])
        self.say("The student learns to ramble: higher score, worse answer.",
                 FadeIn(hack, shift=UP * 0.1), t2c={"worse": BAD})
        self.mark("hacking", scores=[s for _, s in steps])
        self.wait(1.8)

        pol = polaroid(0.5).move_to([LEFT_X, -1.55, 0])
        rope = always_redraw(lambda: leash_curve(pol.get_top() + DOWN * 0.08,
                                                 bot.body.get_bottom() + LEFT * 0.25,
                                                 bend=-0.3))
        ref = tag("reference model", SNAP).next_to(pol, RIGHT, buff=0.3).shift(DOWN * 0.2)
        kl = tag("KL penalty", LEASH).next_to(ref, UP, buff=0.18, aligned_edge=LEFT)
        self.say("So we keep a photo of the original model, tied on a leash.",
                 FadeOut(hack), FadeIn(pol, shift=UP * 0.2), run_time=0.7)
        self.play(Create(rope), run_time=0.7)
        self.play(FadeIn(ref, shift=RIGHT * 0.1), FadeIn(kl, shift=RIGHT * 0.1), run_time=0.5)
        self.mark("leash")
        self.wait(1.4)
        self.say("Drift too far from the original, and the leash pulls back.",
                 bot.animate.shift(RIGHT * 0.35 + UP * 0.75), run_time=0.9)
        clean = tile_row(["Canberra", "is", "the", "capital."], -4.55, 1.25, size=22,
                         height=0.56)
        self.play(bot.animate.shift(LEFT * 0.35 + DOWN * 0.75),
                  *[FadeOut(t, shift=LEFT * 0.4) for t in tiles],
                  LaggedStart(*[FadeIn(t) for t in clean], lag_ratio=0.2),
                  rate_func=rate_functions.ease_out_back, run_time=1.1)
        score = self.rescore(score, "8", GOOD)
        self.say("A slightly lower score, but a sensible answer.")
        self.mark("pulled_back")
        self.wait(2.0)

    def families(self):
        self.clear_stage(header=True)
        cards = VGroup(*[family_card(i, *f) for i, f in enumerate(FAMILIES)])
        cards.arrange(RIGHT, buff=0.3).move_to([0, 0.2, 0])
        title = T("Three ways to learn from feedback", 32, INK, weight="BOLD")
        title.move_to([0, 3.05, 0])
        self.say("There are three popular recipes. Let's meet them one by one.",
                 FadeIn(title, shift=DOWN * 0.1), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.25) for c in cards], lag_ratio=0.3),
                  run_time=1.5)
        self.mark("families")
        self.wait(3.2)

    # ------------------------------------------------------------------ PPO
    def ppo(self):
        self.clear_stage(header=True)
        self.open_section(1, *FAMILIES[0])
        bot, pol, rope = self.left_team()
        jd = judge(0.72).move_to([5.75, 0.25, 0])
        self.add(rope)
        self.play(FadeIn(bot), FadeIn(pol), FadeIn(jd), run_time=0.6)
        answer = FX["ppo"]["answer"]
        row = tile_row(answer, -4.45, 0.25)
        online = tag("online: it learns from its own new answers", STUDENT)
        online.move_to([-1.0, -0.62, 0])
        self.step(0)
        self.say("The student practices by writing brand-new answers.")
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.25) for t in row], lag_ratio=0.45),
                  run_time=1.8)
        self.play(FadeIn(online, shift=UP * 0.1), run_time=0.4)
        card = scorecard(f"{FX['ppo']['judge_score']:g}", JUDGE, 0.75).move_to([5.75, 1.75, 0])
        self.say("The judge scores the whole answer: 2 out of 10.",
                 FadeIn(card, shift=UP * 0.3), run_time=0.7)
        self.mark("ppo_scored", score=FX["ppo"]["judge_score"])
        self.wait(1.6)

        self.step(1)
        qs = VGroup(*[T("?", 26, SOFT, weight="BOLD").next_to(t, UP, buff=0.2) for t in row])
        self.say("One score for six words. Which word caused the problem?",
                 FadeOut(online),
                 LaggedStart(*[FadeIn(q, shift=DOWN * 0.1) for q in qs], lag_ratio=0.15),
                 run_time=0.9)
        self.wait(1.6)
        co = coach(0.6).move_to([LEFT_X, 2.0, 0])
        ctag = tag("critic (value model)", COACH).move_to([LEFT_X, 1.15, 0])
        self.say("A coach helps: before each word, it guesses the final score.",
                 FadeOut(qs), FadeIn(co, shift=DOWN * 0.2), FadeIn(ctag), run_time=0.8)
        guesses, final = FX["ppo"]["coach_guesses"], FX["ppo"]["judge_score"]
        xs = [t.get_left()[0] - 0.07 for t in row] + [row[-1].get_right()[0] + 0.07]
        vals = guesses + [final]

        def y(v):
            return 1.05 + 0.19 * v
        pts = [np.array([x, y(v), 0]) for x, v in zip(xs, vals)]
        dots = VGroup(*[Dot(p, radius=0.075, color=COACH if i < len(guesses) else JUDGE)
                        for i, p in enumerate(pts)])
        nums = VGroup(*[T(f"{v:g}", 19, COACH if i < len(guesses) else JUDGE, weight="BOLD")
                        .next_to(d, UP, buff=0.09) for i, (d, v) in enumerate(zip(dots, vals))])
        segs = VGroup(*[Line(a, b, stroke_color=COACH if i < len(guesses) - 1 else BAD,
                             stroke_width=4 if i < len(guesses) - 1 else 6)
                        for i, (a, b) in enumerate(zip(pts, pts[1:]))])
        self.play(FadeIn(dots[0], scale=0.5), FadeIn(nums[0]), run_time=0.3)
        for i in range(1, len(guesses)):
            self.play(Create(segs[i - 1]), FadeIn(dots[i], scale=0.5), FadeIn(nums[i]),
                      run_time=0.32)
        link = DashedLine(card.get_left() + LEFT * 0.05, pts[-1] + RIGHT * 0.12,
                          stroke_color=JUDGE, stroke_width=2.5, dash_length=0.08)
        self.play(Create(link), run_time=0.4)
        self.play(Create(segs[-1]), FadeIn(dots[-1], scale=0.5), FadeIn(nums[-1]),
                  run_time=0.6)
        self.mark("ppo_guesses", guesses=vals)
        self.wait(1.2)

        credits = FX["ppo"]["credits"]
        chips = VGroup(*[T(signed(c), 23, GOOD if c > 0 else BAD if c < 0 else SOFT,
                           weight="BOLD").next_to(t, DOWN, buff=0.24)
                         for t, c in zip(row, credits)])
        drop = T(f"{guesses[-1]:g} → {final:g}", 19, BAD, weight="BOLD")
        drop.next_to(segs[-1].get_center(), DOWN + LEFT, buff=0.06)
        adv = tag("advantage", SOFT).next_to(chips[-1], RIGHT, buff=0.45)
        self.step(2)
        self.say("Credit = how much each word changed the coach's guess.",
                 LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in chips], lag_ratio=0.18),
                 run_time=1.2)
        self.play(FadeIn(drop), row[-1].box.animate.set_stroke(BAD, 4),
                  Indicate(chips[-1], color=BAD, scale_factor=1.35), run_time=0.7)
        self.play(FadeIn(adv, shift=LEFT * 0.1), run_time=0.4)
        self.mark("ppo_credit", credits=credits)
        self.wait(2.2)

        # ---- one gentle, clipped step on the wheel for the last word
        self.play(FadeOut(dots), FadeOut(nums), FadeOut(segs), FadeOut(link), FadeOut(drop),
                  FadeOut(adv), *[FadeOut(c) for c in chips[:-1]], run_time=0.5)
        shares0, names, eps = FX["wheel"]["shares"], FX["wheel"]["words"], FX["wheel"]["eps"]
        wc = np.array([row[-1].get_center()[0], -1.85, 0])
        R = 0.8
        p_s = ValueTracker(shares0[0])

        def now():
            p = p_s.get_value()
            rest = (1 - p) / (1 - shares0[0])
            return [p] + [s * rest for s in shares0[1:]]
        wh = always_redraw(lambda: wheel(now(), WHEEL_COLORS, R).move_to(wc))
        labs = always_redraw(lambda: wheel_labels(now(), names, R, center=wc, size=17))
        ptr = pointer(R).shift(wc)
        self.say("Then shrink “Sydney” on the wheel, by a small step.", t2c={"Sydney": BAD})
        self.play(FadeIn(wh), FadeIn(labs), FadeIn(ptr), run_time=0.6)
        fence_to = shares0[0] * (1 - eps)
        fa = PI / 2 - 2 * PI * fence_to
        fdir = np.array([math.cos(fa), math.sin(fa), 0])
        fence = Line(wc, wc + fdir * (R + 0.32), stroke_color=LEASH, stroke_width=6)
        fl = T("fence", 18, LEASH, weight="BOLD").next_to(fence.get_end(), RIGHT, buff=0.08)
        self.play(Create(fence), FadeIn(fl), run_time=0.5)
        self.add(fence, fl)
        self.play(p_s.animate.set_value(fence_to), run_time=1.6)
        maths = T(f"{round(100 * shares0[0])}% × {1 - eps:g} = {round(100 * fence_to)}%", 24,
                  INK, weight="BOLD").move_to([-2.5, -1.55, 0])
        clip = tag("clipping", LEASH).next_to(maths, DOWN, buff=0.25)
        self.play(Indicate(fence, color=BAD, scale_factor=1.1), FadeIn(maths, shift=UP * 0.1),
                  run_time=0.6)
        self.say("The fence stops it: one round changes it by 20% at most.",
                 FadeIn(clip, shift=UP * 0.1))
        self.mark("ppo_clip", share=round(p_s.get_value(), 6))
        self.wait(2.2)

        rounds = FX["wheel"]["round_shares"]
        k = ValueTracker(1.0)

        def at():
            v = k.get_value()
            i = min(int(v), len(rounds) - 2)
            return lerp(rounds[i], rounds[i + 1], v - i)
        wh2 = always_redraw(lambda: wheel(at(), WHEEL_COLORS, R).move_to(wc))
        labs2 = always_redraw(lambda: wheel_labels(at(), names, R, center=wc, size=17))
        counter = always_redraw(lambda: T(
            f"round {math.ceil(k.get_value() - 1e-6)} of {len(rounds) - 1}",
            19, SOFT).move_to([-2.5, -1.55, 0]))
        self.remove(wh, labs)
        self.add(wh2, labs2)
        self.say("Round after round, good words grow and bad ones shrink.",
                 FadeOut(fence), FadeOut(fl), FadeOut(maths), FadeOut(clip), FadeIn(counter))
        self.play(k.animate.set_value(len(rounds) - 1), run_time=4.2,
                  rate_func=rate_functions.linear)
        fixed = tile("Canberra", GOOD, GOOD_FILL).move_to(row[-1]).align_to(row[-1], LEFT)
        card = self.rescore(card, "9", GOOD, Transform(row[-1], fixed), FadeOut(chips[-1]),
                            run_time=0.7)
        self.mark("ppo_rounds", final=[round(v, 4) for v in rounds[-1]])
        self.wait(1.2)

        learn = CurvedArrow(card.get_left() + LEFT * 0.1, co.get_right() + RIGHT * 0.1,
                            angle=0.32, color=COACH, stroke_width=4)
        self.say("Meanwhile, the coach also learns from the judge's real scores.",
                 Create(learn), run_time=0.9)
        self.pulse(co)
        self.wait(1.4)

        self.clear_stage()
        mem = memory_panel(FX["models"]["ppo"]).move_to([0, 0.25, 0])
        self.say("Powerful, but heavy: four models share the memory.",
                 FadeIn(mem, scale=0.96), run_time=0.8)
        self.mark("ppo_done", models=FX["models"]["ppo"])
        self.wait(2.4)

    # ------------------------------------------------------------------ GRPO
    def grpo(self):
        self.clear_stage(header=True)
        self.open_section(2, *FAMILIES[1])
        bot, pol, rope = self.left_team(bot_y=0.35)
        self.add(rope)
        q = prompt_card(FX["grpo"]["question"]).move_to([-1.3, 2.3, 0])
        self.play(FadeIn(bot), FadeIn(pol), FadeIn(q, shift=DOWN * 0.15), run_time=0.6)
        ys = [1.35, 0.55, -0.25, -1.05]
        rows = [tile_row(tr, -4.6, yy, size=20, height=0.56) for tr, yy in
                zip(FX["grpo"]["tries"], ys)]
        self.step(0)
        self.say("Ask the same question several times, then compare the tries.")
        for r in rows:
            self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.2) for t in r], lag_ratio=0.3),
                      run_time=0.7)
        self.mark("grpo_tries", n=len(rows))
        self.wait(1.2)

        self.step(1)
        ghost = coach(0.55).move_to([LEFT_X, 2.1, 0]).set_opacity(0.45)
        nix = cross_icon(BAD, 0.42, 9).move_to(ghost)
        why = tag("no critic needed", COACH).move_to([LEFT_X + 0.25, 1.28, 0])
        self.say("Comparing replaces the coach, a whole extra model to train.",
                 FadeIn(ghost), run_time=0.6)
        self.play(Create(nix), FadeIn(why), run_time=0.5)
        self.mark("grpo_no_coach", models=FX["models"]["grpo"])
        self.wait(1.8)
        self.play(FadeOut(ghost), FadeOut(nix), FadeOut(why), run_time=0.4)

        self.step(2)
        key = clipboard("answer key", str(FX["grpo"]["key"]), 0.62).move_to([5.9, 1.9, 0])
        self.say("Check each try with the answer key: right = 1, wrong = 0.",
                 FadeIn(key, shift=LEFT * 0.2), run_time=0.7)
        scores = FX["grpo"]["scores"]
        marks = VGroup()
        for r, s in zip(rows, scores):
            icon = (check_icon() if s else cross_icon()).move_to([2.2, r.get_center()[1], 0])
            pts = T(str(s), 26, GOOD if s else BAD, font=ROUND).move_to(
                [2.85, r.get_center()[1], 0])
            marks.add(VGroup(icon, pts))
        self.play(LaggedStart(*[FadeIn(m, shift=LEFT * 0.1) for m in marks], lag_ratio=0.3),
                  run_time=1.2)
        self.mark("grpo_scored", scores=scores)
        self.wait(1.4)

        mean = FX["grpo"]["mean"]
        avg = VGroup(T("group average", 19, SOFT),
                     T(f"({' + '.join(map(str, scores))}) ÷ {len(scores)} = {mean:g}", 24, INK,
                       weight="BOLD")).arrange(DOWN, buff=0.12).move_to([4.75, 2.3, 0])
        self.say("The group's average score is 0.5.", FadeOut(key), FadeIn(avg), run_time=0.7)
        self.wait(1.6)
        credits = FX["grpo"]["credits"]
        chips = VGroup(*[T(f"{s} − {mean:g} = {signed(c)}", 22, GOOD if c > 0 else BAD,
                           weight="BOLD").move_to([4.75, r.get_center()[1], 0])
                         for r, s, c in zip(rows, scores, credits)])
        adv = tag("advantage", SOFT).next_to(chips, DOWN, buff=0.3)
        spread = T("(GRPO also divides by the spread)", 17, SOFT)
        spread.next_to(adv, DOWN, buff=0.14)
        self.say("Credit = score − average. Above average gets encouraged.",
                 LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in chips], lag_ratio=0.25),
                 run_time=1.2)
        self.play(FadeIn(adv), FadeIn(spread), run_time=0.5)
        self.mark("grpo_credit", mean=mean, credits=credits)
        self.wait(2.0)

        tint = []
        for r, tr, c in zip(rows, FX["grpo"]["tries"], credits):
            col, fill = (GOOD, GOOD_FILL) if c > 0 else (BAD, BAD_FILL)
            tint += [Transform(t, tile(w, col, fill, size=20, height=0.56).move_to(t))
                     for t, w in zip(r, tr)]
        self.say("Every word in a try shares that try's credit.",
                 LaggedStart(*tint, lag_ratio=0.06), run_time=1.3)
        self.mark("grpo_shared")
        self.wait(2.0)

        self.clear_stage()
        mem = memory_panel(FX["models"]["grpo"]).move_to([0, 0.35, 0])
        note = tag("the judge can even be a simple answer checker", JUDGE)
        note.next_to(mem, DOWN, buff=0.3)
        self.say("Same small steps and leash, and no coach: three models.",
                 FadeIn(mem, scale=0.96), run_time=0.8)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.4)
        self.mark("grpo_done", models=FX["models"]["grpo"])
        self.wait(2.4)

    # ------------------------------------------------------------------ DPO
    def dpo(self):
        self.clear_stage(header=True)
        self.open_section(3, *FAMILIES[2])
        self.step(0)
        p1 = eye_panel("1", False).move_to([-1.7, 0.45, 0])
        p2 = eye_panel("2", True).move_to([1.7, 0.45, 0])
        self.say("Like an eye test: you only say which one looks better.",
                 FadeIn(p1, shift=UP * 0.2), FadeIn(p2, shift=UP * 0.2), run_time=0.8)
        self.wait(0.8)
        ok = check_icon(GOOD, 0.45, 9).next_to(p2, DOWN, buff=0.3)
        self.play(Create(ok), run_time=0.5)
        self.mark("dpo_eye_test")
        self.wait(1.5)

        q = prompt_card("Explain gravity to a 5-year-old.", 23).move_to([0, 1.85, 0])
        ca = answer_card("A", "Earth pulls things toward it, so they fall.", True, width=9.6,
                         size=21)
        cb = answer_card("B", "Mass curves spacetime; objects follow geodesics.", False,
                         width=9.6, size=21)
        ca.move_to([0, 0.65, 0])
        cb.move_to([0, -0.4, 0])
        self.say("DPO learns from pairs of answers that people compared.",
                 FadeOut(p1), FadeOut(p2), FadeOut(ok), FadeIn(q, shift=DOWN * 0.1),
                 run_time=0.7)
        self.play(FadeIn(ca, shift=UP * 0.15), run_time=0.5)
        self.play(FadeIn(cb, shift=UP * 0.15), run_time=0.5)
        self.mark("dpo_pair")
        self.wait(2.0)

        self.step(1)
        gone = VGroup(judge(0.5), coach(0.5)).arrange(RIGHT, buff=0.5).move_to([0, -1.75, 0])
        gone.set_opacity(0.45)
        nix = VGroup(*[cross_icon(BAD, 0.38, 9).move_to(g) for g in gone])
        off = tag("offline: learns from saved pairs", SNAP).next_to(gone, RIGHT, buff=0.5)
        self.say("No judge, no coach, no practice loop: simpler and cheaper.",
                 FadeIn(gone), run_time=0.6)
        self.play(Create(nix), FadeIn(off), run_time=0.5)
        self.mark("dpo_why", models=FX["models"]["dpo"])
        self.wait(2.0)

        # ---- how: tip the balance, measured against the original
        self.step(2, FadeOut(q), FadeOut(ca), FadeOut(cb), FadeOut(gone), FadeOut(nix),
                  FadeOut(off))
        path, ref = FX["dpo"]["path"], FX["dpo"]["ref_liking"]
        kk = ValueTracker(0.0)

        def m_now():
            v = kk.get_value()
            i = min(int(v), len(path) - 2)
            return path[i]["margin"] + (path[i + 1]["margin"] - path[i]["margin"]) * (v - i)
        pivot = np.array([0, 0.95, 0])
        arm, drop = 2.2, 1.0
        ta = tile("A", GOOD, GOOD_FILL, size=24)
        tb = tile("B", BAD, BAD_FILL, size=24)
        bal = always_redraw(lambda: balance(0.14 * m_now(), pivot, arm=arm, drop=drop,
                                            left=ta, right=tb, post=1.95))

        def push():
            m = m_now()
            w = 1 / (1 + math.exp(m))
            tilt = 0.14 * m
            x = -arm * math.cos(tilt)
            ytop = pivot[1] - arm * math.sin(tilt) - drop + 0.78
            L = 3.0 * w
            arr = Arrow([x, ytop + L, 0], [x, ytop, 0], buff=0, color=LEASH, stroke_width=9,
                        max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=30)
            lab = T(f"push {round(100 * w)}%", 19, LEASH, weight="BOLD")
            lab.next_to(arr, LEFT, buff=0.12)
            return VGroup(arr, lab)

        base_y, k_h = -1.55, 3.5

        def bars(x, key, sign):
            m = m_now()
            ratio = math.exp(sign * m / 2)
            h0 = ref[key] * k_h
            b0 = Rectangle(width=0.46, height=h0, fill_color=SNAP_FILL, fill_opacity=1,
                           stroke_color=SNAP, stroke_width=2)
            b0.move_to([x - 0.45, base_y + h0 / 2, 0])
            b1 = Rectangle(width=0.46, height=h0 * ratio, fill_color=STUDENT_FILL,
                           fill_opacity=1, stroke_color=STUDENT, stroke_width=2)
            b1.move_to([x + 0.45, base_y + h0 * ratio / 2, 0])
            txt = f"×{ratio:.1f}" if ratio >= 1 else f"÷{1 / ratio:.1f}"
            mult = T(txt, 21, STUDENT, weight="BOLD").next_to(b1, UP, buff=0.1)
            return VGroup(b0, b1, mult)
        bars_a = always_redraw(lambda: bars(-4.3, "A", 1))
        bars_b = always_redraw(lambda: bars(4.3, "B", -1))
        heads = VGroup()
        for x, t in ((-4.3, ta), (4.3, tb)):
            h = t.copy().scale(0.8).move_to([x, 1.3, 0])
            legend = VGroup(T("original", 16, SNAP), T("student", 16, STUDENT))
            legend[0].move_to([x - 0.45, base_y - 0.26, 0])
            legend[1].move_to([x + 0.45, base_y - 0.26, 0])
            heads.add(VGroup(h, legend))
        imp = tag("implicit reward: how much more likely than the original", STUDENT)
        imp.move_to([2.0, 2.35, 0])
        arrow = always_redraw(push)
        self.say("Tip the balance: make A more likely than before, B less.",
                 FadeIn(bal), FadeIn(heads), FadeIn(bars_a), FadeIn(bars_b), run_time=0.8)
        self.play(FadeIn(imp, shift=DOWN * 0.1), run_time=0.4)
        self.mark("dpo_start", margin=m_now(), push=1 / (1 + math.exp(m_now())))
        self.wait(1.0)
        self.play(FadeIn(arrow), run_time=0.4)
        for i in range(1, len(path)):
            extra = self.cap("Once A clearly wins, the pushes get gentle.") if i == 4 else []
            self.play(kk.animate.set_value(i), *extra, run_time=0.6)
        last = path[-1]
        up, lead = last["ratio_a"], last["ratio_a"] / last["ratio_b"]
        sums = VGroup(
            T(f"A ×{up:g}, B ÷{1 / last['ratio_b']:g}:  A leads {up:g} × {1 / last['ratio_b']:g}"
              f" = {lead:g} to 1", 23, INK),
            T(f"push = 1 ÷ (1 + {lead:g}) = {round(100 / (1 + lead))}%", 23, LEASH,
              weight="BOLD")).arrange(DOWN, buff=0.16).move_to([0, -2.45, 0])
        self.play(FadeIn(sums, shift=UP * 0.1), run_time=0.6)
        self.mark("dpo_trained", margin=round(m_now(), 6), ratio_a=round(math.exp(m_now() / 2), 6),
                  ratio_b=round(math.exp(-m_now() / 2), 6),
                  push=round(1 / (1 + math.exp(m_now())), 6), lead=round(lead, 6))
        assert math.isclose(m_now(), last["margin"])
        self.wait(2.8)
        self.play(FadeOut(sums), run_time=0.4)

        bot, pol, rope = self.left_team(bot_y=1.0, pol_y=-1.25, s=0.6)
        self.say("Measuring against the original works as a built-in leash.",
                 FadeIn(bot), FadeIn(pol), run_time=0.7)
        self.play(Create(rope), Circumscribe(heads[0][1][0], color=SNAP),
                  Circumscribe(heads[1][1][0], color=SNAP), run_time=1.0)
        self.add(rope)
        self.mark("dpo_leash")
        self.wait(1.6)

        self.clear_stage()
        mem = memory_panel(FX["models"]["dpo"]).move_to([0, 0.3, 0])
        self.say("Just two models, but it only learns from the pairs it's given.",
                 FadeIn(mem, scale=0.96), run_time=0.8)
        self.mark("dpo_done", models=FX["models"]["dpo"])
        self.wait(2.6)

    # ------------------------------------------------------------------ summary
    def summary(self):
        self.clear_stage(header=True)
        title = T("The big picture", 32, INK, weight="BOLD").move_to([0, 3.15, 0])
        qs = ["Better than the\ncoach expected?", "Better than the\ngroup average?",
              "Better than the\nother answer?"]
        feet = [f"online · {len(FX['models']['ppo'])} models",
                f"online · {len(FX['models']['grpo'])} models",
                f"offline · {len(FX['models']['dpo'])} models"]
        cards = VGroup()
        for i, (f, qq, ft) in enumerate(zip(FAMILIES, qs, feet)):
            extra = VGroup(T(qq, 23, INK), T(ft, 19, SOFT)).arrange(DOWN, buff=0.22)
            cards.add(family_card(i, *f, extra=extra, height=5.55))
        cards.arrange(RIGHT, buff=0.3).move_to([0, -0.05, 0])
        self.say("Each asks: is this answer better than something?",
                 FadeIn(title, shift=DOWN * 0.1), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.25) for c in cards], lag_ratio=0.3),
                  run_time=1.5)
        self.wait(2.6)
        self.say("Then it makes better answers more likely, on a leash.")
        self.mark("summary")
        self.wait(3.2)
