"""Render three README mechanism loops; requires Pillow and FFmpeg, no model.

Run from the repository root:
    uv run --project visuals/animations python visuals/readme/render.py

All numerical inputs are illustrative fixtures. The decoder drawing is a
schematic, not a forward pass or a visualization of measured activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
W, H, SCALE, FPS = 1120, 560, 2, 12
INK = "#111827"
MUTED = "#64748B"
LINE = "#E2E8F0"
BLUE = "#2563EB"
VIOLET = "#7C3AED"
GREEN = "#047857"
AMBER = "#B45309"
RED = "#BE4253"
WHITE = "#FFFFFF"

VOCAB = ["The", "cat", "sat", "on", "the", "mat", "."]
LOGITS = [
    [-2.0, -1.5, -1.2, 2.4, 0.8, -0.2, 0.5],
    [-2.0, -1.3, -1.0, -0.3, 2.5, 1.1, 0.2],
    [-2.0, -1.0, -0.8, 0.2, -0.5, 2.6, 1.0],
]
Q = [[1.0, 0.0, 0.2], [0.2, 1.0, 0.1], [0.8, 0.3, 0.4],
     [0.4, 0.9, 0.6], [0.6, 0.6, 0.2]]
K = [[0.8, 0.1, 0.4], [0.1, 0.9, 0.2], [0.7, 0.4, 0.3],
     [0.3, 0.8, 0.5], [0.9, 0.2, 0.1]]
V = [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
TOKENS = VOCAB[:5]
EPS = 1e-8
GROUPS = [(["4", "5", "4", "3"], [1.0, 0.0, 1.0, 0.0]),
          (["4"] * 4, [1.0] * 4)]


def softmax(values):
    maximum = max(values)
    exps = [math.exp(x - maximum) for x in values]
    return [x / sum(exps) for x in exps]


def attention():
    rows = []
    for i, query in enumerate(Q):
        scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(3)
                  for key in K[:i + 1]]
        rows.append(softmax(scores) + [0.0] * (len(K) - i - 1))
    return rows


def group_values(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    advantages = [(r - mean) / (std + EPS) if std else 0.0 for r in rewards]
    return mean, std, advantages


PROBS = [softmax(row) for row in LOGITS]
ATTENTION = attention()
OUTPUTS = [[sum(a * v[j] for a, v in zip(row, V)) for j in range(2)]
           for row in ATTENTION]


def verify_fixture():
    assert [VOCAB[max(range(len(row)), key=row.__getitem__)] for row in PROBS] == ["on", "the", "mat"]
    assert all(abs(sum(row) - 1) < 1e-12 for row in PROBS + ATTENTION)
    assert all(ATTENTION[i][j] == 0 for i in range(5) for j in range(i + 1, 5))
    # Perturb future values: an earlier query's weighted output must not change.
    for i in range(5):
        changed = [v if j <= i else [1e6, -1e6] for j, v in enumerate(V)]
        out = [sum(a * v[j] for a, v in zip(ATTENTION[i], changed)) for j in range(2)]
        assert out == OUTPUTS[i]
    mean, std, adv = group_values(GROUPS[0][1])
    assert (mean, std) == (0.5, 0.5)
    assert all(abs(a - b) < 1e-7 for a, b in zip(adv, [1, -1, 1, -1]))
    assert group_values(GROUPS[1][1]) == (1.0, 0.0, [0.0] * 4)
    return {"probability_normalization": True, "greedy_tokens": ["on", "the", "mat"],
            "causal_future_weights_zero": True, "future_value_invariance": True,
            "population_std": True, "constant_group_zero_advantages": True}


def clamp(value):
    return min(1.0, max(0.0, value))


def ease(value):
    v = clamp(value)
    return v * v * (3 - 2 * v)


def rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, amount):
    return tuple(round(x + (y - x) * clamp(amount)) for x, y in zip(rgb(a), rgb(b)))


class Canvas:
    def __init__(self, font):
        self.image = Image.new("RGB", (W * SCALE, H * SCALE), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.font = font

    def box(self, rect, fill=WHITE, outline=None, radius=14, width=1):
        self.draw.rounded_rectangle(tuple(round(x * SCALE) for x in rect),
                                    radius=round(radius * SCALE), fill=fill,
                                    outline=outline, width=round(width * SCALE))

    def line(self, points, fill=LINE, width=2):
        self.draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
                       fill=fill, width=round(width * SCALE))

    def circle(self, x, y, radius, fill):
        self.draw.ellipse(((x - radius) * SCALE, (y - radius) * SCALE,
                           (x + radius) * SCALE, (y + radius) * SCALE), fill=fill)

    def text(self, x, y, text, size=22, fill=INK, anchor="la"):
        font = ImageFont.truetype(str(self.font), round(size * SCALE))
        self.draw.text((round(x * SCALE), round(y * SCALE)), text, font=font,
                       fill=fill, anchor=anchor)

    def card(self, rect):
        x0, y0, x1, y1 = rect
        self.box((x0, y0 + 3, x1, y1 + 3), "#F1F5F9", radius=18)
        self.box(rect, WHITE, LINE, radius=18)

    def arrow(self, x0, y, x1, color=LINE):
        self.line([(x0, y), (x1, y)], color, 2)
        self.line([(x1 - 7, y - 5), (x1, y), (x1 - 7, y + 5)], color, 2)

    def title(self, title, subtitle):
        self.text(48, 32, title, 34)
        self.text(48, 81, subtitle, 21, MUTED)
        self.line([(48, 121), (W - 48, 121)], LINE, 1)

    def finish(self, fade=0):
        image = self.image.resize((W, H), Image.Resampling.LANCZOS)
        if fade:
            image = Image.blend(image, Image.new("RGB", (W, H), WHITE), clamp(fade))
        return image


def next_token(t, font):
    c = Canvas(font)
    c.title("A language model, one token at a time",
            "Read the prefix. Score the vocabulary. Append one token.")
    step = min(int(t / 4.2), 2)
    local = min(t - step * 4.2, 4.2)
    prefix = VOCAB[:3] + ["on", "the", "mat"][:step]
    selected = ["on", "the", "mat"][step]
    landed = local >= 3.7
    shown = prefix + ([selected] if landed else [])
    for i in range(6):
        x = 194 + i * 124
        if i < len(shown):
            color = BLUE if i < 3 else GREEN
            c.box((x, 153, x + 112, 204), mix(WHITE, color, .08), mix(WHITE, color, .25))
            c.text(x + 56, 179, shown[i], 27, color, "mm")
        else:
            c.box((x, 153, x + 112, 204), "#FAFBFD", LINE)
            c.text(x + 56, 179, "·", 26, "#CBD5E1", "mm")

    c.card((48, 256, 325, 481))
    c.text(72, 276, "Prefix", 25, BLUE)
    c.text(72, 320, "The cat sat", 29)
    if step:
        c.text(72, 360, " ".join(prefix[3:]), 29)
    c.text(72, 437, f"{len(prefix)} context tokens", 20, MUTED)
    c.arrow(341, 369, 397)
    c.card((412, 256, 666, 481))
    c.text(436, 276, "Decoder", 25, VIOLET)
    for i, label in enumerate(["Embeddings", "Causal blocks", "Output head"]):
        active = int(clamp((local - 1.0) / 1.1) * 2.99) == i and 1.0 <= local < 2.2
        c.box((436, 321 + i * 44, 642, 355 + i * 44),
              mix(WHITE, VIOLET, .1 if active else .035), mix(WHITE, VIOLET, .25 if active else .1), 8)
        c.text(539, 338 + i * 44, label, 20, VIOLET if active else MUTED, "mm")
    c.arrow(682, 369, 738)
    c.card((752, 256, 1072, 481))
    c.text(776, 276, "Next-token chances", 25, GREEN)
    order = sorted(range(7), key=lambda j: PROBS[step][j], reverse=True)[:3]
    reveal = ease((local - 2.0) / .55)
    for rank, j in enumerate(order):
        y = 321 + rank * 44
        color = GREEN if rank == 0 else BLUE
        c.text(776, y, VOCAB[j], 22, color)
        c.box((832, y + 6, 999, y + 20), "#F1F5F9", radius=7)
        length = 167 * PROBS[step][j] * reveal
        if length > 2:
            c.box((832, y + 6, 832 + length, y + 20), color, radius=7)
        c.text(1050, y + 12, f"{PROBS[step][j]:.0%}" if reveal > .95 else "—", 20, color, "rm")
    c.text(776, 451, "Top 3 of 7 toy tokens", 18, MUTED)
    if .3 <= local < 1.0:
        c.circle(341 + 56 * ease((local - .3) / .7), 369, 5, BLUE)
    if 1.7 <= local < 2.2:
        c.circle(682 + 56 * ease((local - 1.7) / .5), 369, 5, VIOLET)
    if 3.1 <= local < 3.7:
        u = ease((local - 3.1) / .6)
        x = (1 - u) ** 2 * 912 + 2 * u * (1 - u) * 1050 + u ** 2 * (250 + len(prefix) * 124)
        y = (1 - u) ** 2 * 334 + 2 * u * (1 - u) * 212 + u ** 2 * 179
        c.box((x - 48, y - 23, x + 48, y + 23), GREEN, radius=12)
        c.text(x, y, selected, 26, WHITE, "mm")
    c.text(48, 517, "Greedy decoding · illustrative logits · schematic decoder", 18, MUTED)
    c.text(1072, 517, f"{step + 1:02d} / 03", 18, MUTED, "ra")
    return c.finish(ease((t - 13.0) / .5))


def causal_attention(t, font):
    c = Canvas(font)
    c.title("Attention has an information boundary",
            "Each query can read its past and itself. Future positions stay masked.")
    row = min(int(t / 2.2), 4)
    local = min(t - row * 2.2, 2.2)
    x0, y0, cell = 133, 201, 53
    c.text(133, 148, "Keys →", 20, BLUE)
    c.text(48, 177, "Queries ↓", 18, VIOLET)
    for j, token in enumerate(TOKENS):
        c.text(x0 + j * cell + cell / 2, 179, token, 19, BLUE, "mm")
        c.text(112, y0 + j * cell + cell / 2, token, 19,
               VIOLET if j == row else MUTED, "rm")
    for i in range(5):
        for j in range(5):
            x, y = x0 + j * cell, y0 + i * cell
            if j > i:
                c.box((x, y, x + cell - 4, y + cell - 4), "#F1F5F9", radius=7)
                c.text(x + 24, y + 24, "×", 24, "#CBD5E1", "mm")
            else:
                color = VIOLET if i == row else BLUE
                c.box((x, y, x + cell - 4, y + cell - 4),
                      mix(WHITE, color, .12 + .72 * ATTENTION[i][j]), radius=7)
                c.text(x + 24, y + 24, f"{ATTENTION[i][j]:.2f}", 18,
                       WHITE if ATTENTION[i][j] > .65 else color, "mm")
    c.box((x0 - 5, y0 + row * cell - 5, x0 + 5 * cell,
           y0 + (row + 1) * cell), None, VIOLET, 10, 2)
    c.text(133, 490, "Causal weights [5 × 5]", 20, MUTED)

    c.card((461, 151, 1072, 495))
    c.text(487, 170, f'Query: “{TOKENS[row]}”', 27, VIOLET)
    c.text(487, 214, "Where its attention goes", 20, MUTED)
    baseline = 372
    for j, token in enumerate(TOKENS):
        x = 524 + j * 113
        c.line([(x - 28, baseline), (x + 28, baseline)], LINE, 2)
        weight = ATTENTION[row][j]
        if j <= row:
            height = 102 * weight
            c.box((x - 23, baseline - max(height, 2), x + 23, baseline), BLUE, radius=5)
            c.text(x, baseline - height - 19, f"{weight:.0%}", 21, BLUE, "mm")
            if .25 < local < 1.45:
                u = ease((local - .25) / 1.2)
                c.circle(x, baseline + 37 + u * 32, 3 + 4 * weight, GREEN)
        else:
            c.text(x, 316, "×", 28, "#CBD5E1", "mm")
            c.text(x, 348, "masked", 16, MUTED, "mm")
        c.text(x, 393, token, 23, BLUE if j <= row else MUTED, "mm")
    c.box((487, 437, 1046, 475), "#ECFDF5", radius=9)
    out = OUTPUTS[row]
    c.text(766, 456, f"Weighted value  =  [{out[0]:.2f}, {out[1]:.2f}]", 23, GREEN, "mm")
    c.text(48, 527, "softmax(mask(QKᵀ / √d)) × V · one head · toy vectors", 19, MUTED)
    return c.finish(ease((t - 11.5) / .5))


def group_rewards(t, font):
    c = Canvas(font)
    group = 0 if t < 7.2 else 1
    local = t if group == 0 else t - 7.2
    answers, rewards = GROUPS[group]
    mean, std, advantages = group_values(rewards)
    subtitle = ("One prompt. Four tries. Compare each reward with the group."
                if group == 0 else "A new group: equal rewards give zero relative advantages.")
    c.title("A learning signal from the group", subtitle)
    c.text(560, 153, "2 + 2 = ?", 32, INK, "mm")
    reward_reveal = ease((local - 1.0) / .5)
    advantage_reveal = ease((local - 3.3) / .55)
    for j, (answer, reward, advantage) in enumerate(zip(answers, rewards, advantages)):
        x = 48 + j * 262
        center = x + 119
        c.card((x, 193, x + 238, 348))
        c.text(x + 20, 210, f"Try {j + 1}", 20, MUTED)
        c.text(center, 272, answer, 44, INK, "mm")
        if reward_reveal > 0:
            color = GREEN if reward else RED
            c.box((x + 35, 310, x + 203, 337), mix(WHITE, color, .1 * reward_reveal), radius=8)
            c.text(center, 324, f"reward {reward:.0f}", 20,
                   mix(WHITE, color, reward_reveal), "mm")
        if local > 2.0:
            c.line([(center, 356), (center, 372), (560, 372), (560, 379)], LINE, 2)
        if advantage_reveal > 0:
            color = GREEN if advantage > 0 else RED if advantage < 0 else MUTED
            c.line([(center, 426), (center, 443)], mix(WHITE, color, advantage_reveal), 2)
            c.text(center, 470, f"{advantage:+.2f}" if advantage else "0.00", 34,
                   mix(WHITE, color, advantage_reveal), "mm")
            c.text(center, 504, "advantage", 18, MUTED, "mm")
    if local > 2.0:
        c.box((351, 381, 769, 421), "#FFFBEB", "#FDE9B0", 10)
        c.text(560, 401, f"mean {mean:.1f}  ·  population std {std:.1f}", 22, AMBER, "mm")
    c.text(48, 538, "A = (reward − mean) / (population std + ε)", 18, MUTED)
    c.text(1072, 538, "Toy rewards · reward term only", 18, MUTED, "ra")
    fade = ease((t - 14.5) / .5)
    if 6.85 <= t < 7.2:
        fade = ease((t - 6.85) / .35)
    elif 7.2 <= t < 7.55:
        fade = 1 - ease((t - 7.2) / .35)
    return c.finish(fade)


SCENES = {
    "next-token": (next_token, 13.5, [0, 2.8, 3.9, 7.9, 10.8, 12.6]),
    "causal-attention": (causal_attention, 12.0, [.4, 2.8, 5.0, 7.2, 9.4, 11.0]),
    "group-relative-rewards": (group_rewards, 15.0, [.6, 1.8, 3.0, 5.0, 9.0, 12.5]),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_font(requested):
    candidates = [Path(requested)] if requested else [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("No supported font found; pass --font /path/to/font.ttf")


def render(name, font, qa_dir):
    scene, duration, samples = SCENES[name]
    frames = round(duration * FPS)
    output = HERE / f"{name}.gif"
    with tempfile.TemporaryDirectory(prefix=f"dongxi-readme-{name}-") as temp:
        temp = Path(temp)
        for i in range(frames):
            scene(i / FPS, font).save(temp / f"{i:04d}.png")
        palette = temp / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(temp / "%04d.png"), "-vf",
                        "palettegen=max_colors=128:reserve_transparent=1", "-frames:v", "1",
                        str(palette)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(temp / "%04d.png"), "-i", str(palette), "-lavfi",
                        "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                        "-loop", "0", str(output)], check=True)
    poster = HERE / f"{name}.png"
    scene(5.0 if name == "group-relative-rewards" else samples[-1], font).save(poster)
    # Sample the decoded GIF, so contact sheets check palette encoding too.
    sheet = Image.new("RGB", (W, 3 * (H // 2 + 28)), "#F8FAFC")
    draw = ImageDraw.Draw(sheet)
    with Image.open(output) as gif:
        delays, unique = [], set()
        for i in range(gif.n_frames):
            gif.seek(i)
            delays.append(gif.info.get("duration", 0))
            unique.add(hashlib.sha256(gif.convert("RGB").tobytes()).hexdigest())
        for j, at in enumerate(samples):
            total = 0
            for index, delay in enumerate(delays):
                total += delay
                if total > at * 1000:
                    break
            gif.seek(index)
            x, y = (j % 2) * W // 2, (j // 2) * (H // 2 + 28)
            sheet.paste(gif.convert("RGB").resize((W // 2, H // 2)), (x, y + 28))
            draw.text((x + 12, y + 7), f"{name} · {at:.1f}s", fill=INK)
        assert gif.size == (W, H)
        assert gif.info.get("loop") == 0
        assert len(unique) >= 25
        assert abs(sum(delays) / 1000 - duration) < .15
        assert output.stat().st_size < 3 * 1024 * 1024
        info = {"file": output.name, "dimensions": [W, H], "frames": gif.n_frames,
                "duration_seconds": sum(delays) / 1000, "loop": 0,
                "unique_frames": len(unique), "bytes": output.stat().st_size,
                "sha256": digest(output), "poster": poster.name,
                "poster_sha256": digest(poster)}
    qa_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(qa_dir / f"{name}-sheet.png")
    print(f"{name}: {info['frames']} frames, {info['duration_seconds']:.2f}s, "
          f"{info['bytes'] / 1024:.0f} KiB", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font")
    parser.add_argument("--qa-dir", type=Path, default=Path(tempfile.gettempdir()) / "dongxi-readme-qa")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("FFmpeg is required for palette-optimized GIF export.")
    font = find_font(args.font)
    checks = verify_fixture()
    outputs = [render(name, font, args.qa_dir) for name in SCENES]
    manifest = {
        "purpose": "README mechanism previews, not measured model behavior or training results",
        "renderer": "visuals/readme/render.py", "renderer_sha256": digest(Path(__file__)),
        "command": "uv run --project visuals/animations python visuals/readme/render.py",
        "executed_command": [str(Path(sys.executable).relative_to(ROOT)), *sys.argv],
        "environment": {"python": platform.python_version(), "pillow": PIL.__version__,
                        "platform": platform.platform(), "font": str(font), "font_sha256": digest(font),
                        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]},
        "fixtures": {"vocabulary": VOCAB, "logits": LOGITS, "probabilities": PROBS,
                     "Q": Q, "K": K, "V": V, "attention": ATTENTION, "outputs": OUTPUTS,
                     "groups": [{"answers": a, "rewards": r, "mean_std_advantages": group_values(r)}
                                for a, r in GROUPS], "epsilon": EPS},
        "checks": checks, "outputs": outputs,
        "limitations": ["Fixed logits and a schematic decoder; no model forward pass or training.",
                        "One attention head with five positions and two-dimensional toy values; positional operations omitted.",
                        "Population-normalized group advantages only; no policy loss, clipping, KL or optimizer update.",
                        "The fade to white is an editorial loop reset, not a state transition."],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
