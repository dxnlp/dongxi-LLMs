"""PPO, GRPO and DPO: three ways to turn feedback into a better policy.

Render (from this directory):
    python render.py --preview      # 960x540, 15 fps
    python render.py                # 1920x1080, 30 fps
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from manim import *

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kit import (  # noqa: E402
    BG, PANEL, TEXT, SUB, DIM, LINE, POLICY, REF, REWARD, CRITIC, POS, NEG, CLIP,
    LOOP_ANGLE, T, M, Mc, tok, block, make_chip, row_positions, tracker, loop_ends,
    header, spring,
)

FX = json.loads((HERE / "fixture.json").read_text())
RY = -3.5            # roster row
TY = 0.3             # main token row


def softplus_neg(m):
    return float(np.log1p(np.exp(-m)))


class FeedbackFamilies(Scene):
    # ================================================================ plumbing
    def mark(self, name, **data):
        hw, hh = config.frame_width / 2, config.frame_height / 2
        for m in self.mobjects:
            if isinstance(m, ValueTracker) or not m.family_members_with_points():
                continue
            l, r = m.get_left()[0], m.get_right()[0]
            b, t = m.get_bottom()[1], m.get_top()[1]
            assert -hw + 0.03 < l and r < hw - 0.03 and -hh + 0.03 < b and t < hh - 0.03, \
                (name, type(m).__name__, round(l, 2), round(r, 2), round(b, 2), round(t, 2))
        self.events.append(dict(name=name, t=round(float(self.time), 3), **data))

    def persistent(self):
        keep = [self.head, self.trk_strip, self.trk_bar, self.roster_label, self.leash,
                self.leash_lab, self.trk_arrow]
        keep += [c for _, c in self.roster]
        return [m for m in keep if m is not None]

    def clear_stage(self, keep=(), run_time=0.8):
        keep_ids = {id(m) for m in list(keep) + self.persistent()}
        gone = [m for m in self.mobjects if id(m) not in keep_ids]
        for m in gone:
            m.clear_updaters()
        vis = [m for m in gone if m.family_members_with_points()]
        if vis:
            self.play(*[FadeOut(m) for m in vis], run_time=run_time)
        self.remove(*gone)

    # ---------------------------------------------------------------- tracker
    def install_tracker(self, labels, loop):
        self.trk_strip, self.trk_words, self.trk_arrow = tracker(labels, loop)
        return [FadeIn(self.trk_strip)] + ([Create(self.trk_arrow)] if loop else [])

    def step(self, i):
        anims = [w.animate.set_color(TEXT if j == i else DIM) for j, w in enumerate(self.trk_words)]
        w = self.trk_words[i]
        bar = Line(w.get_corner(DL) + DOWN * 0.08, w.get_corner(DR) + DOWN * 0.08,
                   color=CLIP, stroke_width=3)
        if self.trk_bar is None:
            self.trk_bar = bar
            anims.append(Create(bar))
        else:
            anims.append(Transform(self.trk_bar, bar))
        return anims

    def reset_tracker(self):
        anims = [w.animate.set_color(DIM) for w in self.trk_words]
        if self.trk_bar is not None:
            anims.append(FadeOut(self.trk_bar))
            self.trk_bar = None
        return anims

    def swap_header(self, name, subtitle, *extra):
        new_head = header(name, subtitle)
        self.play(FadeOut(self.head, shift=UP * 0.2), *extra, run_time=0.6)
        self.play(FadeIn(new_head, shift=UP * 0.2), run_time=0.6)
        self.head = new_head

    def loop_flash(self):
        flash = self.trk_arrow.copy().set_stroke(CLIP, 4)
        return ShowPassingFlash(flash, time_width=0.7)

    # ----------------------------------------------------------------- roster
    def roster_to(self, keys):
        current = dict(self.roster)
        chips = [current.get(k) or make_chip(k) for k in keys]
        pos = row_positions(chips, RY, buff=0.36, x_center=0.55)
        anims = []
        for k, c, p in zip(keys, chips, pos):
            if k in current:
                anims.append(c.animate.move_to(p))
            else:
                c.move_to(p)
                anims.append(FadeIn(c, shift=UP * 0.15))
        for k, c in current.items():
            if k not in keys:
                anims.append(FadeOut(c, shift=DOWN * 0.15))
        target = pos[0] + LEFT * (chips[0].width / 2 + 0.3 + self.roster_label.width / 2)
        if self.roster:
            anims.append(self.roster_label.animate.move_to(target))
        else:
            self.roster_label.move_to(target)
            anims.append(FadeIn(self.roster_label))
        self.roster = list(zip(keys, chips))
        return anims

    def chip_of(self, key):
        return dict(self.roster)[key]

    def pulse(self, key, s=1.12):
        return self.chip_of(key).animate(rate_func=there_and_back).scale(s)

    def cross_out(self, key):
        c = self.chip_of(key)
        return VGroup(Line(c.get_corner(UL), c.get_corner(DR)),
                      Line(c.get_corner(DL), c.get_corner(UR))).set_stroke(NEG, 4)

    def leash_shape(self):
        a = self.chip_of("policy").get_top() + UP * 0.05
        b = self.chip_of("reference").get_top() + UP * 0.05
        return ArcBetweenPoints(a, b, angle=-0.22, color=CLIP, stroke_width=2.4)

    def add_leash(self):
        self.leash = self.leash_shape()
        label = M(r"\beta\,\mathrm{KL}", 0.4, CLIP)
        backing = BackgroundRectangle(label, color=BG, fill_opacity=1, buff=0.08)
        self.leash_lab = VGroup(backing, label)
        self.leash_lab.move_to(self.leash.point_from_proportion(0.5))
        anims = [Create(self.leash), FadeIn(self.leash_lab)]
        self.leash.add_updater(lambda m: m.become(self.leash_shape()))
        self.leash_lab.add_updater(lambda m: m.move_to(self.leash.point_from_proportion(0.5)))
        return anims

    # ------------------------------------------------------------------ props
    def prompt_card(self):
        box = RoundedRectangle(corner_radius=0.12, width=0.72, height=0.72, stroke_color=TEXT,
                               stroke_width=2, fill_color=TEXT, fill_opacity=0.06)
        x = M("x", 1.2, TEXT).move_to(box)
        name = T("prompt", 19, SUB).next_to(box, DOWN, buff=0.12)
        return VGroup(box, x, name)

    def mini_pair_card(self):
        box = RoundedRectangle(corner_radius=0.14, width=1.9, height=1.25, stroke_color=LINE,
                               stroke_width=2, fill_color=PANEL, fill_opacity=1)
        line = RoundedRectangle(corner_radius=0.05, width=1.1, height=0.12, stroke_width=0,
                                fill_color=SUB, fill_opacity=0.7)
        good = VGroup(*[tok(POS, 0.2, 0.5, 1.4) for _ in range(5)]).arrange(RIGHT, buff=0.07)
        bad = VGroup(*[tok(NEG, 0.2, 0.5, 1.4) for _ in range(5)]).arrange(RIGHT, buff=0.07)
        VGroup(line, good, bad).arrange(DOWN, buff=0.14, aligned_edge=LEFT).move_to(box)
        return VGroup(box, line, good, bad)

    # ============================================================== the film
    def construct(self):
        self.camera.background_color = BG
        self.events = []
        self.roster = []
        self.roster_label = T("models", 19, SUB)
        self.head = self.trk_strip = self.trk_bar = self.trk_arrow = None
        self.leash = self.leash_lab = None
        self.intro()
        self.ppo()
        self.grpo()
        self.dpo()
        self.summary()
        (HERE / "timeline.json").write_text(json.dumps(
            dict(duration=round(float(self.time), 3), events=self.events), indent=2) + "\n")

    # ------------------------------------------------------------------ intro
    def intro(self):
        title = T("Learning from feedback", 62, TEXT, MEDIUM)
        names = VGroup(T("PPO", 34, SUB), T("·", 34, DIM), T("GRPO", 34, SUB), T("·", 34, DIM),
                       T("DPO", 34, SUB)).arrange(RIGHT, buff=0.32)
        VGroup(title, names).arrange(DOWN, buff=0.45)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(n, shift=UP * 0.1) for n in names], lag_ratio=0.12),
                  run_time=0.9)
        self.wait(1.0)
        self.mark("title")
        self.play(FadeOut(title, shift=UP * 0.2), FadeOut(names, shift=UP * 0.2), run_time=0.7)

        # One shared goal: more reward, but stay close to a frozen reference.
        prompt = self.prompt_card().move_to([-5.1, 0.2, 0])
        pol = block(r"\pi_\theta", POLICY).move_to([-3.0, 0.45, 0])
        arrow = always_redraw(lambda: Arrow(prompt[0].get_right(), pol.get_left(), buff=0.1,
                                            color=DIM, stroke_width=3,
                                            max_tip_length_to_length_ratio=0.3))
        toks = VGroup(*[tok(POLICY).move_to([-1.35 + 0.72 * i, 0.45, 0]) for i in range(5)])
        y_lab = M(r"y\sim\pi_\theta(\cdot\,|\,x)", 0.46, SUB).next_to(toks, DOWN, buff=0.22)
        self.play(FadeIn(prompt), FadeIn(pol), FadeIn(arrow), run_time=0.8)
        self.play(LaggedStart(*[GrowFromPoint(t, pol.get_right()) for t in toks], lag_ratio=0.3),
                  run_time=1.5)
        self.play(FadeIn(y_lab), run_time=0.4)

        fb = M(r"r(x,y)", 0.6, REWARD)
        fb_box = SurroundingRectangle(fb, buff=0.14, corner_radius=0.12, color=REWARD,
                                      stroke_width=2)
        feedback = VGroup(fb_box, fb).move_to([3.55, 0.45, 0])
        fb_arrow = Arrow(toks.get_right() + RIGHT * 0.05, fb_box.get_left(), buff=0.12,
                         color=DIM, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        fb_name = T("feedback", 20, REWARD).next_to(feedback, DOWN, buff=0.15)
        self.play(GrowArrow(fb_arrow), FadeIn(feedback, shift=LEFT * 0.15), FadeIn(fb_name),
                  run_time=0.8)

        ref = block(r"\pi_{\mathrm{ref}}", REF, frozen=True).move_to([-3.0, -2.15, 0])
        ref_name = T("frozen reference copy", 20, SUB).next_to(ref, RIGHT, buff=0.25)
        leash = spring(pol.get_bottom(), ref.get_top())
        beta = M(r"\beta", 0.55, CLIP).next_to(leash, LEFT, buff=0.18)
        self.play(FadeIn(ref), FadeIn(ref_name), Create(leash), FadeIn(beta), run_time=1.0)
        leash.add_updater(lambda m: m.become(spring(pol.get_bottom(), ref.get_top())))
        beta.add_updater(lambda m: m.next_to(leash, LEFT, buff=0.18))

        goal = Mc([(r"\max_{\theta}\ \mathbb{E}\,[\,r(x,y)\,]", REWARD),
                   (r"\ -\ \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})", CLIP)], 0.72)
        goal.move_to([1.2, 2.75, 0])
        y_under = goal.get_bottom()[1] - 0.25
        more = T("more reward", 20, REWARD)
        more.move_to([VGroup(*goal.parts[0]).get_center()[0], y_under, 0])
        near = T("stay close to the reference", 20, CLIP)
        near.move_to([VGroup(*goal.parts[1]).get_center()[0], y_under, 0])
        self.play(FadeIn(goal, shift=DOWN * 0.1), run_time=0.8)
        self.play(FadeIn(more), pol.animate.shift(UP * 0.55), run_time=0.9)
        self.play(FadeIn(near), pol.animate.shift(DOWN * 0.3), run_time=0.8)
        self.wait(1.2)
        self.mark("shared_goal")
        leash.clear_updaters()
        beta.clear_updaters()
        arrow.clear_updaters()
        setup = VGroup(prompt, pol, arrow, toks, y_lab, fb_arrow, feedback, fb_name, ref,
                       ref_name, leash, beta, goal, more, near)
        self.play(FadeOut(setup), run_time=0.7)

        cards = self.family_cards()
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.22),
                  run_time=1.4)
        self.wait(3.0)
        self.mark("families")
        self.play(FadeOut(cards), run_time=0.6)

    def family_cards(self):
        specs = [("Online RL", "with a critic", "PPO", self.icon_critic()),
                 ("Online policy gradient", "without a critic", "GRPO", self.icon_group()),
                 ("Direct preference", "optimization", "DPO", self.icon_pair())]
        cards = VGroup()
        for x, (l1, l2, name, icon) in zip((-4.35, 0, 4.35), specs):
            box = RoundedRectangle(corner_radius=0.25, width=4.05, height=3.5, stroke_color=LINE,
                                   stroke_width=2, fill_color=PANEL, fill_opacity=1)
            box.move_to([x, 0, 0])
            icon.scale_to_fit_height(min(icon.height, 0.8)).move_to([x, 0.95, 0])
            c1 = T(l1, 25, TEXT, MEDIUM).move_to([x, 0, 0]).align_to([0, 0.2, 0], UP)
            c2 = T(l2, 25, TEXT, MEDIUM).move_to([x, 0, 0]).align_to([0, -0.2, 0], UP)
            badge_txt = T(name, 24, TEXT, MEDIUM)
            badge = RoundedRectangle(corner_radius=0.2, width=badge_txt.width + 0.5, height=0.5,
                                     stroke_color=SUB, stroke_width=1.6, fill_opacity=0)
            eg = T("e.g.", 19, SUB)
            row = VGroup(eg, badge).arrange(RIGHT, buff=0.16).move_to([x, -1.12, 0])
            badge_txt.move_to(badge)
            eg.shift(DOWN * 0.07)
            cards.add(VGroup(box, icon, c1, c2, row, badge_txt))
        return cards

    def icon_critic(self):
        V, R, D = FX["ppo"]["values"], FX["ppo"]["reward"], FX["ppo"]["deltas"]
        lv = V + [R]
        g = VGroup()
        for t in range(6):
            x0, x1 = -1.2 + 0.4 * t, -0.8 + 0.4 * t
            g.add(Line([x0, lv[t], 0], [x1, lv[t], 0], color=CRITIC, stroke_width=3.5))
            g.add(Line([x1, lv[t], 0], [x1, lv[t + 1], 0], color=POS if D[t] > 0 else NEG,
                       stroke_width=3.5))
        return g.stretch_to_fit_height(0.75)

    def icon_group(self):
        rw = FX["grpo"]["rewards"]
        bars = VGroup(*[Rectangle(width=1.8 * r, height=0.16, stroke_width=0, fill_color=REWARD,
                                  fill_opacity=0.85) for r in rw]).arrange(DOWN, buff=0.08,
                                                                          aligned_edge=LEFT)
        x = bars.get_left()[0] + 1.8 * FX["grpo"]["mean"]
        mean = DashedLine([x, bars.get_bottom()[1] - 0.08, 0], [x, bars.get_top()[1] + 0.08, 0],
                          color=TEXT, stroke_width=2, dash_length=0.06)
        return VGroup(bars, mean)

    def icon_pair(self):
        good = VGroup(*[tok(POS, 0.26, 0.4, 1.8) for _ in range(5)]).arrange(RIGHT, buff=0.08)
        bad = VGroup(*[tok(NEG, 0.26, 0.4, 1.8) for _ in range(5)]).arrange(RIGHT, buff=0.08)
        return VGroup(good, bad).arrange(DOWN, buff=0.16)

    # -------------------------------------------------------------------- PPO
    def ppo(self):
        fx = FX["ppo"]
        V, R, D, A, eps = fx["values"], fx["reward"], fx["deltas"], fx["advantages"], fx["eps"]
        path = np.array(fx["ratio_path"])
        TX = [-2.7 + 0.84 * i for i in range(6)]
        yv = lambda v: 0.75 + 1.9 * v            # noqa: E731
        YB = -0.85

        self.head = header("PPO", "online RL with a critic")
        self.play(FadeIn(self.head, shift=RIGHT * 0.2), *self.install_tracker(
            ["sample", "score", "credit", "update"], loop=True), run_time=0.9)

        # 1. sample ------------------------------------------------------------
        prompt = self.prompt_card().move_to([-6.1, TY - 0.2, 0])
        prompt.shift(UP * (TY - prompt[0].get_center()[1]))
        pol = block(r"\pi_\theta", POLICY).move_to([-4.45, TY, 0])
        pol_name = T("policy", 20, SUB).next_to(pol, DOWN, buff=0.12)
        a1 = Arrow(prompt[0].get_right(), pol.get_left(), buff=0.08, color=DIM, stroke_width=3,
                   max_tip_length_to_length_ratio=0.3)
        self.play(*self.step(0), FadeIn(prompt), FadeIn(pol), FadeIn(pol_name), GrowArrow(a1),
                  *self.roster_to(["policy"]), run_time=1.0)
        toks = VGroup(*[tok(POLICY, 0.56).move_to([x, TY, 0]) for x in TX])
        self.play(LaggedStart(*[GrowFromPoint(t, pol.get_right()) for t in toks], lag_ratio=0.3),
                  run_time=2.1)
        samp = M(r"y\sim\pi_\theta(\cdot\,|\,x)", 0.46, SUB).next_to(toks, DOWN, buff=0.22)
        self.play(FadeIn(samp), run_time=0.5)
        self.wait(0.3)
        self.mark("ppo_sampled", tokens=len(toks))

        # 2. score -------------------------------------------------------------
        rm = block(r"r", REWARD, frozen=True).move_to([4.6, TY, 0])
        rm_name = T("reward model", 20, SUB).next_to(rm, DOWN, buff=0.12)
        a2 = Arrow(toks.get_right() + RIGHT * 0.05, rm.get_left(), buff=0.12, color=DIM,
                   stroke_width=3, max_tip_length_to_length_ratio=0.12)
        self.play(*self.step(1), FadeIn(rm), FadeIn(rm_name), GrowArrow(a2),
                  *self.roster_to(["policy", "reward"]), run_time=1.0)
        scan = Rectangle(width=toks.width + 0.3, height=0.84, stroke_width=0, fill_color=REWARD,
                         fill_opacity=0.2).move_to(toks)
        self.play(FadeIn(scan), run_time=0.3)
        self.play(scan.animate.stretch_to_fit_width(0.12).move_to(rm.get_left()), run_time=0.7)
        self.remove(scan)
        xR = TX[-1] + 0.42 + 0.36
        r_dot = Dot([xR, yv(R), 0], radius=0.09, color=REWARD)
        r_lab = M(rf"r={R:.2f}", 0.46, REWARD).next_to(r_dot, RIGHT, buff=0.14)
        flying = r_dot.copy().move_to(rm.get_top())
        self.play(flying.animate(path_arc=0.9).move_to(r_dot), run_time=0.8)
        self.add(r_dot)
        self.remove(flying)
        one = T("one score for the whole answer", 19, REWARD).next_to(r_lab, UP, buff=0.18)
        one.align_to(r_dot, LEFT).shift(LEFT * 0.4)
        self.play(FadeIn(r_lab), FadeIn(one), run_time=0.5)
        self.wait(0.6)
        self.mark("ppo_scored", reward=R)

        # 3. credit: the critic ------------------------------------------------
        cr = block(r"V_\phi", CRITIC, w=1.3, h=0.95, symbol_scale=0.55).move_to([-4.45, 2.0, 0])
        cr_name = T("critic", 20, SUB).next_to(cr, LEFT, buff=0.18)
        self.play(*self.step(2), FadeIn(cr), FadeIn(cr_name), FadeOut(samp), FadeOut(one),
                  *self.roster_to(["policy", "critic", "reward"]), run_time=1.0)
        # Level left of token t is V(s_t); token t moves it to V(s_{t+1}), so the jump
        # drawn above token t is delta_t (the last one lands on the final reward).
        levels = V + [R]
        plate, jump = VGroup(), VGroup()
        for t in range(6):
            xa = TX[t - 1] if t else TX[0] - 0.42
            plate.add(Line([xa, yv(V[t]), 0], [TX[t], yv(V[t]), 0], color=CRITIC,
                           stroke_width=4))
            jump.add(Line([TX[t], yv(V[t]), 0], [TX[t], yv(levels[t + 1]), 0],
                          color=POS if D[t] > 0 else NEG, stroke_width=4))
        tail = Line([TX[-1], yv(R), 0], [xR, yv(R), 0], color=REWARD, stroke_width=3)
        v_lab = M(r"V", 0.62, CRITIC).next_to(plate[0], LEFT, buff=0.16)
        scanner = Line([TX[0] - 0.42, TY - 0.36, 0], [TX[0] - 0.42, yv(0.84), 0], color=CRITIC,
                       stroke_width=2, stroke_opacity=0.7)
        guess = T("expected final score", 19, CRITIC).next_to(v_lab, UP, buff=0.5)
        guess.align_to(v_lab, LEFT)
        self.play(FadeIn(scanner), FadeIn(v_lab), FadeIn(guess), run_time=0.4)
        for t in range(6):
            self.play(Create(plate[t]), scanner.animate.set_x(TX[t]), run_time=0.3,
                      rate_func=linear)
            self.play(Create(jump[t]), toks[t].animate(rate_func=there_and_back)
                      .set_fill(CRITIC, 0.5), run_time=0.2)
        self.play(Create(tail), FadeOut(scanner), run_time=0.3)
        bad = 2
        d_lab = M(r"\delta_t", 0.6, NEG).next_to(jump[bad], RIGHT, buff=0.1)
        self.play(Indicate(jump[bad], color=NEG, scale_factor=1.25),
                  toks[bad].animate.set_stroke(NEG, 3.5), FadeIn(d_lab), FadeOut(guess),
                  run_time=0.9)
        self.mark("ppo_values", values=V)

        # Sparse final reward, gamma=1, lambda near 1: the advantage is close to
        # (final reward) - (critic's expectation before the token).
        rline = DashedLine([TX[0] - 0.42, yv(R), 0], [xR, yv(R), 0], color=REWARD,
                           stroke_width=2, dash_length=0.08)
        gaps = VGroup(*[Line([TX[t] - 0.17, yv(V[t]), 0], [TX[t] - 0.17, yv(R), 0],
                             color=POS if R > V[t] else NEG, stroke_width=6)
                        for t in range(6)])
        self.play(FadeOut(d_lab), Create(rline), run_time=0.5)
        self.play(LaggedStart(*[Create(g) for g in gaps], lag_ratio=0.12), run_time=0.9)
        base = Line([TX[0] - 0.45, YB, 0], [TX[-1] + 0.45, YB, 0], color=LINE, stroke_width=2)
        bars = VGroup()
        for t in range(6):
            h = 3.6 * A[t]
            bars.add(Rectangle(width=0.34, height=abs(h), stroke_width=0,
                               fill_color=POS if h > 0 else NEG, fill_opacity=0.9)
                     .move_to([TX[t], YB + h / 2, 0]))
        a_lab = M(r"\hat{A}_t", 0.5, TEXT).next_to(base, RIGHT, buff=0.2)
        self.play(Create(base), FadeIn(a_lab), run_time=0.4)
        self.play(LaggedStart(*[TransformFromCopy(gaps[t], bars[t]) for t in range(6)],
                              lag_ratio=0.12), run_time=1.6)
        gae = Mc([(r"\hat{A}_t=\sum_{k}\,(\gamma\lambda)^k\,\delta_{t+k}", TEXT),
                  (r"\ \approx\ r-V_\phi(s_t)", SUB)], 0.56).move_to([-0.6, -2.12, 0])
        self.play(FadeIn(gae, shift=UP * 0.1), run_time=0.6)
        self.mark("ppo_advantages", signs=[int(np.sign(a)) for a in A],
                  heights=[round(b.height * np.sign(a), 4) for b, a in zip(bars, A)])
        learn = CurvedArrow(r_dot.get_top() + UP * 0.12, cr.get_right() + UP * 0.2 + RIGHT * 0.06,
                            angle=0.55, color=CRITIC, stroke_width=2.5, tip_length=0.15)
        learn_lab = T("trained to predict the reward", 19, CRITIC)
        learn_lab.move_to(learn.point_from_proportion(0.5) + UP * 0.22)
        self.play(Create(learn), FadeIn(learn_lab), run_time=1.0)
        self.play(Indicate(cr, color=CRITIC, scale_factor=1.06), run_time=0.6)
        self.wait(0.5)

        # 4. update: clipped ratio ---------------------------------------------
        faded = VGroup(plate, jump, tail, rline, gaps)
        self.play(*self.step(3), faded.animate.set_stroke(opacity=0.25), FadeOut(learn),
                  FadeOut(learn_lab), toks[bad].animate.set_stroke(POLICY, 2.4), run_time=0.8)
        S = 2.5
        gauges, dots, pushes = VGroup(), VGroup(), VGroup()
        for t in range(6):
            x = TX[t]
            track = Line([x, YB - 0.72, 0], [x, YB + 0.72, 0], color=LINE, stroke_width=2)
            band = Rectangle(width=0.46, height=2 * eps * S, stroke_color=CLIP, stroke_width=1.6,
                             stroke_opacity=0.85, fill_color=CLIP, fill_opacity=0.1)
            gauges.add(VGroup(track, band.move_to([x, YB, 0])))
            dots.add(Dot([x, YB, 0], radius=0.085, color=POLICY))
            s = 1 if A[t] > 0 else -1
            pushes.add(Arrow([x + 0.36, YB - s * 0.02, 0], [x + 0.36, YB + s * 0.55, 0], buff=0,
                             color=POS if s > 0 else NEG, stroke_width=5,
                             max_tip_length_to_length_ratio=0.35))
        band_lab = M(r"1\pm\epsilon", 0.45, CLIP).next_to(base, RIGHT, buff=0.2).shift(UP * 0.5)
        rho_lab = M(r"\rho_t=1", 0.45, SUB).next_to(base, RIGHT, buff=0.2)
        self.play(*[ReplacementTransform(bars[t], pushes[t]) for t in range(6)], FadeIn(gauges),
                  FadeIn(dots), FadeOut(a_lab), FadeIn(band_lab), FadeIn(rho_lab), run_time=1.2)
        clip_eq = Mc([(r"\min(\rho_t\hat{A}_t,\ ", TEXT),
                      (r"\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)", CLIP),
                      (r"\,\hat{A}_t)", TEXT)], 0.58)
        ratio_eq = M(r"\rho_t=\dfrac{\pi_\theta(y_t\,|\,s_t)}{\pi_{\mathrm{old}}(y_t\,|\,s_t)}", 0.5,
                     SUB)
        eqs = VGroup(clip_eq, ratio_eq).arrange(RIGHT, buff=0.6).move_to([-0.25, -2.12, 0])
        self.play(FadeOut(gae, shift=DOWN * 0.1), FadeIn(eqs, shift=UP * 0.1), run_time=0.7)

        u = ValueTracker(0)
        last = len(path) - 1

        def rho(t):
            s = u.get_value()
            i = min(int(s), last - 1)
            return path[i, t] + (s - i) * (path[i + 1, t] - path[i, t])

        for t in range(6):
            dots[t].add_updater(lambda m, t=t: m.move_to([TX[t], YB + S * (rho(t) - 1), 0])
                                .set_color(CLIP if abs(rho(t) - 1) > eps - 1e-6 else POLICY))
            pushes[t].add_updater(
                lambda m, t=t: m.set_opacity(0.12 if abs(rho(t) - 1) > eps - 1e-6 else 1))
        self.play(u.animate.set_value(last), run_time=3.6, rate_func=linear)
        for m in [*dots, *pushes]:
            m.clear_updaters()
        self.mark("ppo_clipped", final_ratio=[round(float(rho(t)), 4) for t in range(6)])
        self.remove(u)
        stop = T("clipped: no further push", 19, CLIP).move_to([3.3, YB - 0.5, 0],
                                                              aligned_edge=LEFT)
        self.play(FadeIn(stop), run_time=0.4)

        self.play(*self.roster_to(["policy", "critic", "reward", "reference"]), run_time=0.9)
        self.play(*self.add_leash(), run_time=0.8)
        self.play(self.loop_flash(), Indicate(pol, color=POLICY, scale_factor=1.05), run_time=1.3)
        self.play(LaggedStart(*[self.pulse(k, 1.1) for k, _ in self.roster], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(0.6)
        self.mark("ppo_done", models=[k for k, _ in self.roster])
        self.stage_keep = [prompt, pol, pol_name, a1]

    # ------------------------------------------------------------------- GRPO
    def grpo(self):
        fx = FX["grpo"]
        rw, A, L, mean = fx["rewards"], fx["advantages"], fx["lengths"], fx["mean"]
        prompt, pol, pol_name, a1 = self.stage_keep
        self.clear_stage(keep=self.stage_keep)
        self.swap_header("GRPO", "online policy gradient without a critic",
                         *self.reset_tracker())
        cross = self.cross_out("critic")
        self.play(Create(cross), run_time=0.5)
        self.play(FadeOut(cross), *self.roster_to(["policy", "reward", "reference"]),
                  run_time=1.0)
        self.mark("grpo_no_critic", models=[k for k, _ in self.roster])

        rows_y = [TY + 1.2, TY + 0.4, TY - 0.4, TY - 1.2]
        x0, dx, s = -2.75, 0.62, 0.48
        rows = VGroup(*[VGroup(*[tok(POLICY, s).move_to([x0 + dx * j, y, 0]) for j in range(n)])
                        for n, y in zip(L, rows_y)])
        self.play(*self.step(0), LaggedStart(*[LaggedStart(
            *[GrowFromPoint(t, pol.get_right()) for t in row], lag_ratio=0.25) for row in rows],
            lag_ratio=0.18), run_time=2.8)
        cap = T("a group of answers to the same prompt", 19, SUB).next_to(rows, DOWN, buff=0.3)
        cap.align_to(rows, LEFT)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(0.3)
        self.mark("grpo_group", group_size=len(rows))

        XB0, KS = 2.1, 3.2
        rbars = VGroup(*[Rectangle(width=KS * r, height=0.34, stroke_width=0, fill_color=REWARD,
                                   fill_opacity=0.85).move_to([XB0 + KS * r / 2, y, 0])
                         for r, y in zip(rw, rows_y)])
        rnums = VGroup(*[T(f"{r:.1f}", 20, REWARD).next_to(b, RIGHT, buff=0.12)
                         for r, b in zip(rw, rbars)])
        rhead = T("reward (model or rule)", 19, REWARD).move_to([XB0, rows_y[0] + 0.55, 0],
                                                                aligned_edge=LEFT)
        self.play(*self.step(1), self.pulse("reward"), FadeIn(rhead), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in rbars], lag_ratio=0.15),
                  run_time=1.2)
        self.play(FadeIn(rnums), run_time=0.4)
        self.wait(0.4)

        xm = XB0 + KS * mean
        y_lo, y_hi = rows_y[-1] - 0.42, rows_y[0] + 0.3
        ghost = DashedLine([xm, y_lo, 0], [xm, y_hi, 0], color=CRITIC, stroke_width=3,
                           dash_length=0.1)
        ghost_lab = M(r"V_\phi", 0.45, CRITIC).next_to(ghost, DOWN, buff=0.1)
        mline = DashedLine([xm, y_lo, 0], [xm, y_hi, 0], color=TEXT, stroke_width=3,
                           dash_length=0.1)
        m_lab = T("group mean", 19, TEXT).next_to(mline, DOWN, buff=0.1)
        self.play(*self.step(2), Create(ghost), FadeIn(ghost_lab), run_time=0.8)
        self.play(ReplacementTransform(ghost, mline), FadeOut(ghost_lab, shift=DOWN * 0.1),
                  FadeIn(m_lab, shift=DOWN * 0.1), run_time=1.0)
        self.mark("grpo_mean", mean=mean)

        devs, advbars = VGroup(), VGroup()
        for r, a, y in zip(rw, A, rows_y):
            xe = XB0 + KS * r
            lo, hi = sorted([xm, xe])
            col = POS if r > mean else NEG
            devs.add(Rectangle(width=hi - lo, height=0.34, stroke_width=0, fill_color=col,
                               fill_opacity=0.9).move_to([(lo + hi) / 2, y, 0]))
            w = 0.95 * abs(a)
            advbars.add(Rectangle(width=w, height=0.34, stroke_width=0, fill_color=col,
                                  fill_opacity=0.9).move_to([xm + np.sign(a) * w / 2, y, 0]))
        ahead = T("advantage", 19, TEXT).move_to([xm, rhead.get_center()[1], 0])
        self.play(FadeIn(devs), rbars.animate.set_fill(opacity=0.22), run_time=0.9)
        self.play(ReplacementTransform(devs, advbars), FadeOut(rbars), FadeOut(rnums),
                  FadeOut(rhead), FadeIn(ahead), run_time=1.2)
        adv_eq = M(r"\hat{A}_i=\dfrac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}", 0.6)
        adv_eq.move_to([3.7, -2.3, 0])
        self.play(FadeIn(adv_eq, shift=UP * 0.1), run_time=0.6)
        self.mark("grpo_advantages", widths=[round(b.width * np.sign(a), 4)
                                             for b, a in zip(advbars, A)])

        amax = max(abs(a) for a in A)
        tints = []
        for row, a in zip(rows, A):
            col = POS if a > 0 else NEG
            op = 0.22 + 0.55 * abs(a) / amax
            tints.append(LaggedStart(*[t.animate.set_fill(col, op).set_stroke(col)
                                       for t in reversed(row)], lag_ratio=0.12))
        same = T("same advantage for every token of an answer", 19, SUB).move_to(cap,
                                                                                  aligned_edge=LEFT)
        self.play(*tints, FadeOut(cap), FadeIn(same), run_time=1.6)
        self.wait(0.5)
        self.mark("grpo_broadcast")

        ups = VGroup()
        for row, a, y in zip(rows, A, rows_y):
            x = row[-1].get_right()[0] + 0.28
            ln = 0.22 + 0.3 * abs(a) / amax
            sg = np.sign(a)
            ups.add(Arrow([x, y - sg * ln / 2, 0], [x, y + sg * ln / 2, 0], buff=0,
                          color=POS if a > 0 else NEG, stroke_width=5,
                          max_tip_length_to_length_ratio=0.4))
        upd = T("then PPO's clipped update, with a KL term", 19, SUB)
        upd.move_to(same, aligned_edge=LEFT)
        self.play(*self.step(3), LaggedStart(*[GrowArrow(a) for a in ups], lag_ratio=0.15),
                  FadeOut(same), FadeIn(upd), run_time=1.1)
        self.play(self.loop_flash(), Indicate(pol, color=POLICY, scale_factor=1.05),
                  self.leash.animate(rate_func=there_and_back).set_stroke(width=5), run_time=1.3)
        self.wait(0.8)
        self.mark("grpo_done")

    # -------------------------------------------------------------------- DPO
    def dpo(self):
        traj = FX["dpo"]["trajectory"]
        self.clear_stage()
        old_strip, old_arrow = self.trk_strip, self.trk_arrow
        start, end = loop_ends(self.trk_words)
        arc = ArcBetweenPoints(start, end, angle=LOOP_ANGLE, color=DIM, stroke_width=2)
        left = arc.copy().pointwise_become_partial(arc, 0.0, 0.4)
        right = arc.copy().pointwise_become_partial(arc, 0.6, 1.0)
        self.remove(old_arrow)
        self.trk_arrow = None
        self.add(left, right)
        self.play(left.animate.set_color(NEG).shift(LEFT * 0.12 + DOWN * 0.06),
                  right.animate.set_color(NEG).shift(RIGHT * 0.12 + DOWN * 0.06), run_time=0.8)
        self.swap_header("DPO", "direct preference optimization · offline",
                         FadeOut(left), FadeOut(right), FadeOut(old_strip),
                         FadeOut(self.trk_bar))
        self.trk_bar = None
        self.play(*self.install_tracker(["fixed pairs", "score", "credit", "update"], loop=False),
                  run_time=0.5)
        cross = self.cross_out("reward")
        self.play(Create(cross), run_time=0.5)
        self.play(FadeOut(cross), *self.roster_to(["policy", "reference"]), run_time=1.0)
        self.mark("dpo_offline", models=[k for k, _ in self.roster])

        # 1. fixed pairs --------------------------------------------------------
        deck = VGroup(*[self.mini_pair_card().move_to([-5.15 + 0.09 * k, 0.9 - 0.09 * k, 0])
                        for k in range(5)])
        deck_lab = T("collected before training", 19, SUB).next_to(deck, DOWN, buff=0.25)
        self.play(*self.step(0), LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in deck],
                                            lag_ratio=0.1), FadeIn(deck_lab), run_time=1.0)
        top = deck[-1]
        RX = [-2.55 + 0.62 * j for j in range(5)]
        YW, YL = 0.95, -0.25
        chosen = VGroup(*[tok(POS, 0.52).move_to([x, YW, 0]) for x in RX])
        rejected = VGroup(*[tok(NEG, 0.52).move_to([x, YL, 0]) for x in RX])
        prompt = self.prompt_card()
        prompt.shift([-3.6, (YW + YL) / 2, 0] - prompt[0].get_center())
        w_lab = VGroup(M(r"y_w", 0.62, POS), T("chosen", 19, POS)).arrange(RIGHT, buff=0.12)
        l_lab = VGroup(M(r"y_l", 0.62, NEG), T("rejected", 19, NEG)).arrange(RIGHT, buff=0.12)
        w_lab.next_to(chosen, UP, buff=0.16).align_to(chosen, LEFT)
        l_lab.next_to(rejected, DOWN, buff=0.16).align_to(rejected, LEFT)
        self.play(top.animate.move_to([-1.3, 0.35, 0]).scale(1.6), run_time=0.8)
        self.play(ReplacementTransform(top[2], chosen), ReplacementTransform(top[3], rejected),
                  ReplacementTransform(top[1], prompt), FadeOut(top[0]), run_time=1.0)
        self.play(FadeIn(w_lab), FadeIn(l_lab), run_time=0.5)
        self.play(FadeOut(deck[:-1]), FadeOut(deck_lab), run_time=0.5)
        self.mark("dpo_pair")

        # 2. score: both models read both answers -----------------------------
        pol = block(r"\pi_\theta", POLICY).move_to([-5.5, 1.3, 0])
        ref = block(r"\pi_{\mathrm{ref}}", REF, frozen=True).move_to([-5.5, -0.6, 0])
        self.play(*self.step(1), FadeIn(pol), FadeIn(ref), run_time=0.8)
        for model, col in ((pol, POLICY), (ref, REF)):
            self.play(Indicate(model, color=col, scale_factor=1.05),
                      LaggedStart(*[t.animate(rate_func=there_and_back).set_fill(col, 0.75)
                                    for t in [*chosen, *rejected]], lag_ratio=0.05),
                      run_time=0.9)
        XM, Y0, K = 0.85, 0.35, 0.85
        axis = Line([XM, -1.25, 0], [XM, 1.62, 0], color=LINE, stroke_width=2)
        axis_lab = M(r"\hat{r}", 0.6, SUB).next_to(axis.get_top(), RIGHT, buff=0.14)
        zero = DashedLine([XM - 0.45, Y0, 0], [XM + 0.45, Y0, 0], color=SUB, stroke_width=2,
                          dash_length=0.08)
        zero_lab = M(r"\pi_\theta=\pi_{\mathrm{ref}}", 0.5, SUB).next_to(zero, RIGHT, buff=0.36)
        gdot = Dot([XM - 0.16, Y0, 0], radius=0.1, color=POS)
        rdot = Dot([XM + 0.16, Y0, 0], radius=0.1, color=NEG)
        rhat_eq = M(r"\hat{r}(x,y)=\beta\,\log\dfrac{\pi_\theta(y\,|\,x)}{\pi_{\mathrm{ref}}(y\,|\,x)}",
                    0.55).move_to([0.62, 2.3, 0])
        self.play(Create(axis), FadeIn(axis_lab), Create(zero), FadeIn(zero_lab), FadeIn(gdot),
                  FadeIn(rdot), FadeIn(rhat_eq, shift=DOWN * 0.1), run_time=1.0)
        self.wait(0.6)
        self.mark("dpo_implicit_reward", start=[traj[0]["r_w"], traj[0]["r_l"]])

        # 3. credit: margin and loss -----------------------------------------
        ax = Axes(x_range=[-1, 3, 1], y_range=[0, 1.4, 0.7], x_length=3.5, y_length=2.5,
                  tips=False, axis_config=dict(color=LINE, stroke_width=2, include_ticks=False))
        ax.move_to([4.55, 0.3, 0])
        curve = ax.plot(softplus_neg, x_range=[-1, 3], color=TEXT, stroke_width=3)
        c_lab = M(r"-\log\sigma(m)", 0.45, TEXT).next_to(ax.c2p(-1, 1.32), RIGHT, buff=0.15)
        x_lab = VGroup(T("margin", 19, SUB), M("m", 0.6, SUB)).arrange(RIGHT, buff=0.1)
        x_lab.next_to(ax.c2p(3, 0), DOWN, buff=0.18).align_to(ax.c2p(3, 0), RIGHT)
        y_lab = T("loss", 19, SUB).next_to(ax.c2p(-1, 1.4), UP, buff=0.1)
        loss_eq = Mc([(r"\mathcal{L}_{\mathrm{DPO}}=-\log\sigma(", TEXT), (r"\hat{r}_w", POS),
                      (r"-", TEXT), (r"\hat{r}_l", NEG), (r")", TEXT)], 0.6).move_to([0.2, -2.2, 0])
        m_def = Mc([(r"m=", TEXT), (r"\hat{r}_w", POS), (r"-", TEXT), (r"\hat{r}_l", NEG)], 0.56)
        self.play(*self.step(2), Create(ax), Create(curve), FadeIn(c_lab), FadeIn(x_lab),
                  FadeIn(y_lab), FadeIn(loss_eq, shift=UP * 0.1), run_time=1.2)

        k = ValueTracker(0)
        last = len(traj) - 1

        def st():
            s = k.get_value()
            i = min(int(s), last - 1)
            f = s - i
            a, b = traj[i], traj[i + 1]
            return {key: a[key] + f * (b[key] - a[key]) for key in a}

        gdot.add_updater(lambda m: m.move_to([XM - 0.16, Y0 + K * st()["r_w"], 0]))
        rdot.add_updater(lambda m: m.move_to([XM + 0.16, Y0 + K * st()["r_l"], 0]))

        def bracket():
            yw, yl = gdot.get_center()[1], rdot.get_center()[1]
            x = XM + 0.52
            g = VGroup(Line([x, yl, 0], [x, yw, 0]), Line([x - 0.07, yw, 0], [x + 0.07, yw, 0]),
                       Line([x - 0.07, yl, 0], [x + 0.07, yl, 0])).set_stroke(CLIP, 2.5)
            return g

        br = always_redraw(bracket)

        def loss_dot():
            m = st()["margin"]
            return Dot(ax.c2p(m, softplus_neg(m)), radius=0.1, color=CLIP)

        def tangent():
            m = st()["margin"]
            slope = -st()["weight"]
            p = np.array(ax.c2p(m, softplus_neg(m)))
            d = np.array(ax.c2p(m + 0.55, softplus_neg(m) + 0.55 * slope)) - p
            return Line(p - d, p + d, color=CLIP, stroke_width=2.5, stroke_opacity=0.8)

        def push(dot, sign, col):
            def make():
                ln = 0.12 + 1.1 * st()["weight"]
                c = dot.get_center() + np.array([0, sign * 0.12, 0])
                return Arrow(c, c + np.array([0, sign * ln, 0]), buff=0, color=col,
                             stroke_width=5, max_tip_length_to_length_ratio=0.4)
            return always_redraw(make)

        ld, tg = always_redraw(loss_dot), always_redraw(tangent)
        pw, pl = push(gdot, 1, POS), push(rdot, -1, NEG)
        m_def.next_to(zero_lab, DOWN, buff=0.35).align_to(zero_lab, LEFT)
        self.play(FadeIn(ld), FadeIn(tg), FadeIn(br), FadeIn(m_def), run_time=0.7)
        self.wait(0.4)
        self.mark("dpo_start", margin=traj[0]["margin"], loss=round(traj[0]["loss"], 4))

        # 4. update: the pair pulls apart, the loss slides down ---------------
        self.play(*self.step(3), FadeIn(pw), FadeIn(pl), FadeOut(zero_lab), run_time=0.5)
        self.play(k.animate.set_value(last), run_time=6.0, rate_func=linear)
        end_state = st()
        for m in (gdot, rdot):
            m.clear_updaters()
        self.mark("dpo_trained", margin=round(end_state["margin"], 4),
                  loss=round(end_state["loss"], 4), weight=round(end_state["weight"], 4))
        wrong = T("the push fades as the margin grows", 19, CLIP)
        wrong.next_to(ax, DOWN, buff=0.6)
        self.play(FadeIn(wrong), run_time=0.5)
        own = T("the policy is its own reward model", 17, TEXT).next_to(rhat_eq, RIGHT, buff=0.32)
        self.play(FadeIn(own, shift=LEFT * 0.1), Indicate(pol, color=POLICY, scale_factor=1.05),
                  run_time=1.0)
        self.wait(1.6)
        self.mark("dpo_done")

    # ---------------------------------------------------------------- summary
    def summary(self):
        self.clear_stage()
        persist = [m for m in self.persistent() if m.family_members_with_points()]
        for m in persist:
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in persist], run_time=0.7)
        self.remove(*self.mobjects)

        goal = Mc([(r"\max_{\theta}\ \mathbb{E}\,[\,r(x,y)\,]", REWARD),
                   (r"\ -\ \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})", CLIP)], 0.62)
        goal.move_to([0, 3.05, 0])
        same = T("one goal, three ways to estimate the learning signal", 21, SUB)
        same.next_to(goal, DOWN, buff=0.2)
        cols = []
        specs = [
            ("PPO", "online · critic", self.mini_ppo(), "a critic credits each token",
             [("policy", POLICY, False), ("critic", CRITIC, False), ("reward", REWARD, True),
              ("reference", REF, True)]),
            ("GRPO", "online · no critic", self.mini_grpo(), "group mean replaces the critic",
             [("policy", POLICY, False), ("reward", REWARD, True), ("reference", REF, True)]),
            ("DPO", "offline · pairs", self.mini_dpo(), "no sampling, no reward model",
             [("policy", POLICY, False), ("reference", REF, True)]),
        ]
        syms = {"policy": r"\pi_\theta", "critic": r"V_\phi", "reward": r"r",
                "reference": r"\pi_{\mathrm{ref}}"}
        for x, (name, sub, mini, cap, models) in zip((-4.55, 0, 4.55), specs):
            box = RoundedRectangle(corner_radius=0.25, width=4.15, height=5.0, stroke_color=LINE,
                                   stroke_width=2, fill_color=PANEL, fill_opacity=1)
            box.move_to([x, -0.55, 0])
            nm = T(name, 40, TEXT, MEDIUM).move_to([x, 1.4, 0]).align_to([0, 1.62, 0], UP)
            sb = T(sub, 20, SUB).move_to([x, 0.95, 0]).align_to([0, 1.02, 0], UP)
            mini.scale_to_fit_height(min(mini.height, 1.25)).move_to([x, -0.1, 0])
            chips = VGroup()
            for key, col, frozen in models:
                sq = RoundedRectangle(corner_radius=0.12, width=0.74, height=0.62,
                                      stroke_color=col, stroke_width=2, fill_color=col,
                                      fill_opacity=0.06 if frozen else 0.2)
                if frozen:
                    sq = VGroup(sq.copy().set_stroke(width=0),
                                DashedVMobject(sq.copy().set_fill(opacity=0), num_dashes=22))
                chips.add(VGroup(sq, M(syms[key], 0.7, col).move_to(sq)))
            chips.arrange(RIGHT, buff=0.14).move_to([x, -2.3, 0])
            cp = T(cap, 19, TEXT).move_to([x, -1.35, 0]).align_to([0, -1.23, 0], UP)
            cols.append(VGroup(box, nm, sb, mini, cp, chips))
        note = T("Toy numbers for intuition  ·  PPO: Schulman et al. 2017  ·  "
                 "GRPO: Shao et al. 2024  ·  DPO: Rafailov et al. 2023", 15, DIM)
        note.to_edge(DOWN, buff=0.18)
        self.play(FadeIn(goal, shift=DOWN * 0.1), FadeIn(same), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cols], lag_ratio=0.25),
                  run_time=1.8)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(5.0)
        self.mark("summary")
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.9)
        self.wait(0.3)

    def mini_ppo(self):
        A = FX["ppo"]["advantages"]
        g = VGroup()
        for i, a in enumerate(A):
            x = -1.0 + 0.4 * i
            g.add(tok(POLICY, 0.28, 0.16, 1.6).move_to([x, 0.62, 0]))
            h = 2.2 * a
            g.add(Rectangle(width=0.16, height=abs(h), stroke_width=0,
                            fill_color=POS if a > 0 else NEG, fill_opacity=0.9)
                  .move_to([x, -0.05 + h / 2, 0]))
        g.add(Line([-1.25, -0.05, 0], [1.25, -0.05, 0], color=LINE, stroke_width=1.5))
        return g

    def mini_grpo(self):
        A = FX["grpo"]["advantages"]
        L = FX["grpo"]["lengths"]
        amax = max(abs(a) for a in A)
        rows = VGroup()
        for a, n in zip(A, L):
            col = POS if a > 0 else NEG
            rows.add(VGroup(*[tok(col, 0.22, 0.22 + 0.55 * abs(a) / amax, 1.4)
                              for _ in range(n)]).arrange(RIGHT, buff=0.06))
        return rows.arrange(DOWN, buff=0.1, aligned_edge=LEFT)

    def mini_dpo(self):
        good = VGroup(*[tok(POS, 0.26, 0.4, 1.6) for _ in range(5)]).arrange(RIGHT, buff=0.07)
        bad = VGroup(*[tok(NEG, 0.26, 0.4, 1.6) for _ in range(5)]).arrange(RIGHT, buff=0.07)
        pair = VGroup(good, bad).arrange(DOWN, buff=0.22)
        up = Arrow(ORIGIN, UP * 0.45, buff=0, color=POS, stroke_width=4,
                   max_tip_length_to_length_ratio=0.4).next_to(good, RIGHT, buff=0.15)
        dn = Arrow(ORIGIN, DOWN * 0.45, buff=0, color=NEG, stroke_width=4,
                   max_tip_length_to_length_ratio=0.4).next_to(bad, RIGHT, buff=0.15)
        return VGroup(pair, up, dn)
