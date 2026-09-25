"""One request, one arrival timeline: TTFT followed by three distinct ITLs."""
from functools import partial
import json
from pathlib import Path
import sys
import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE, COMPOSED, ACCENT
label = partial(label, layout_scale=16)


def cn(text, size=26, color=FOREGROUND):
    return label(text, size, color, cjk=True)


class TwoWaits(Scene):
    def mark(self, name, **data):
        for m in self.mobjects:
            assert m.get_left()[0] > -7.1 and m.get_right()[0] < 7.1, (name, 'x')
            assert m.get_bottom()[1] > -3.95 and m.get_top()[1] < 3.95, (name, 'y')
        self.events.append(dict(name=name, seconds=round(float(self.time), 3), **data))

    def construct(self):
        self.camera.background_color = BACKGROUND
        self.events = []
        example = json.loads((P / 'example.json').read_text())
        title = label('Two phases. Two kinds of waiting.', 34).to_edge(UP, buff=.35).to_edge(LEFT, buff=.55)
        note = VGroup(cn('用词语示意', 18, MUTED), label('token', 18, MUTED),
                      cn('· 分词与时长均为示意', 18, MUTED)).arrange(RIGHT, buff=.1).to_edge(DOWN, buff=.25)
        self.add(title, note)
        doc = RoundedRectangle(width=3.55, height=1.4, corner_radius=.12,
            stroke_color=GRID, stroke_width=1.5, fill_color=BASE, fill_opacity=.025).move_to([-4.45, 1.5, 0])
        lines = VGroup(cn(example['document_title'], 27, BASE),
                       cn(example['plan'], 23, MUTED),
                       cn(example['risk'], 23, MUTED)).arrange(DOWN, buff=.12).move_to(doc)
        query = RoundedRectangle(width=3.55, height=.88, corner_radius=.12,
            stroke_color=BASE, stroke_width=1.5, fill_color=BACKGROUND, fill_opacity=1).move_to([-4.45, .18, 0])
        query_text = VGroup(label('User query', 19, BASE),
                            cn(example['request'], 26)).arrange(DOWN, buff=.1).move_to(query)
        assert doc.get_bottom()[1] - query.get_top()[1] > .15
        for frame, content in [(doc, lines), (query, query_text)]:
            assert content.width < frame.width-.25 and content.height < frame.height-.1
        prefill = label('Prefill', 27, BASE).move_to([-4.45, 2.55, 0])
        decode = label('Decode', 27, COMPOSED).move_to([2.65, 2.55, 0])
        self.play(FadeIn(doc), FadeIn(lines), FadeIn(prefill), run_time=.8)
        self.wait(.6)
        self.play(FadeIn(query), FadeIn(query_text), run_time=.5)
        self.wait(.4)
        xs = [-5.9, -.5, 1.6, 3.7, 5.8]
        y = -1.15
        rail = Line([xs[0], y, 0], [xs[-1], y, 0], color=GRID, stroke_width=2)
        ticks = VGroup(*[Line([x, y-.12, 0], [x, y+.12, 0], color=GRID, stroke_width=2) for x in xs])
        sent = cn('请求发出', 23, MUTED).move_to([xs[0], y+.5, 0])
        first = cn('首个输出', 23, ACCENT).move_to([xs[1], y+.5, 0])
        later = cn('后续输出', 23, COMPOSED).move_to([3.7, y+.5, 0])
        self.play(Create(rail), FadeIn(ticks), FadeIn(sent), run_time=.6)
        cards = []
        for x, word in zip(xs[1:], example['output_units']):
            box = RoundedRectangle(width=1.65, height=.8, corner_radius=.1,
                stroke_color=COMPOSED, stroke_width=1.7, fill_color=COMPOSED, fill_opacity=.045).move_to([x, 1.1, 0])
            cards.append(VGroup(box, cn(word, 32, COMPOSED).move_to(box)))
        cursor = Line([xs[1]-.55, .85, 0], [xs[1]-.55, 1.35, 0], color=MUTED, stroke_width=2)
        clock = Dot([xs[0], y, 0], color=ACCENT, radius=.08)
        self.add(clock, cursor)
        self.mark('request_sent', visible_outputs=0, x=xs[0])

        def interval(start, end, color, text, duration, index):
            bottom = -2.0
            growing = Line([start, bottom, 0], [start+.001, bottom, 0], color=color, stroke_width=4)
            cap = Line([start, bottom-.1, 0], [start, bottom+.1, 0], color=color, stroke_width=2)
            caption = label(text, 29, color).move_to([(start+end)/2, -2.55, 0])
            self.add(growing, cap, caption)
            growing.add_updater(lambda m: m.put_start_and_end_on(
                np.array([start, bottom, 0]), np.array([max(start+.001, clock.get_x()), bottom, 0])))
            self.mark('ttft_start' if index == 0 else f'itl_{index}_start',
                      start_x=start, end_x=end, visible_outputs=index)
            self.play(clock.animate.move_to([end, y, 0]), run_time=duration, rate_func=linear)
            growing.clear_updaters()
            growing.put_start_and_end_on([start, bottom, 0], [end, bottom, 0])
            self.add(Line([end, bottom-.1, 0], [end, bottom+.1, 0], color=color, stroke_width=2))
            # Instant reveal pins arrival to the timer endpoint without fade-in ambiguity.
            self.add(cards[index], Dot([end, y, 0], color=color, radius=.07))
            self.add(DashedLine([end, .58, 0], [end, y+.9, 0], color=GRID, stroke_width=1.3))
            if index == 0:
                self.remove(cursor)
                self.add(first, later, decode)
                prefill.set_opacity(.4)
                clock.set_color(COMPOSED)
            self.mark(f'output_{index+1}_received', visible_outputs=index+1, x=end,
                      interval='TTFT' if index == 0 else f'ITL {index}')

        interval(xs[0], xs[1], ACCENT, 'TTFT', 5.0, 0)
        for i in range(1, 4):
            interval(xs[i], xs[i+1], COMPOSED, 'ITL', 2.0, i)
        self.wait(3.5)
        self.mark('complete', visible_outputs=4)
        self.wait(.5)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=.6)
        self.wait(.3)
        (P / 'two-waits-timeline.json').write_text(json.dumps({
            'events': self.events, 'checks': {'inside_canvas': True},
            'duration_seconds': float(self.time)}, ensure_ascii=False, indent=2)+'\n')
