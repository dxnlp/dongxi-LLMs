"""DONGXI LABS mark for every published GIF and MP4 (learner, 2026-10-30: "we should protect our IP").

The mark is stamped at encode time, after Manim, so scenes need no layout changes and the GIF and
the MP4 carry the same mark. It is the cover's letter-spaced wordmark, small, in the muted slate of
the white-background visual system, at the bottom-right corner.

Use in a project's render.py:

    from brand import write_mark, gif_filter, brand_mp4
    mark = write_mark(w, tmp / 'mark.png')
    ffmpeg -i render.mov -i mark.png -filter_complex gif_filter(w, h, fps) -loop 0 out.gif
    brand_mp4(manim_mp4, out_mp4, tmp)

`brand_existing_gif` stamps a GIF that was rendered before this rule, without re-rendering.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEXT = 'DONGXI LABS'
AVENIR = '/System/Library/Fonts/Avenir Next.ttc'
DEMI = 2
COLOR = (100, 116, 139)          # manim_style.MUTED #64748B
ALPHA = 0.9
SIZE = 0.0135                    # font px = SIZE × video width (17 px at 1280, 26 at 1920)
MARGIN = 0.016                   # gap to the right and bottom edges, × video width


def spaced(text=TEXT):
    """Letter-spaced as on the covers: one space between letters, two between words."""
    return ' '.join(text).replace('   ', '  ')


def mark_image(width):
    """RGBA wordmark for a video `width` px wide, drawn at 4× and downsampled for clean edges."""
    k = 4
    font = ImageFont.truetype(AVENIR, round(SIZE * width * k), index=DEMI)
    text = spaced()
    l, t, r, b = font.getbbox(text)
    img = Image.new('RGBA', (r - l + 2 * k, b - t + 2 * k), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((k - l, k - t), text, font=font, fill=COLOR + (round(255 * ALPHA),))
    return img.resize((img.width // k, img.height // k), Image.LANCZOS)


def write_mark(width, path):
    path = Path(path)
    mark_image(width).save(path)
    return path


def margin(width):
    return round(MARGIN * width)


def overlay(width, src='[0:v]', mark='[1:v]', out='[m]'):
    m = margin(width)
    return f'{src}{mark}overlay=W-w-{m}:H-h-{m}{out}'


def gif_filter(w, h, fps):
    """filter_complex for a transparent qtrle render (input 0) and the mark (input 1) → GIF."""
    return (f'color=white:s={w}x{h}:r={fps}[bg];[bg][0:v]overlay=shortest=1[c];'
            + overlay(w, '[c]', '[1:v]', '[m]') + ';[m]format=rgb24,split[a][b];'
            '[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=0[p];'
            '[b][p]paletteuse=dither=none:diff_mode=rectangle')


def video_width(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width', '-of', 'csv=p=0', str(path)],
                         check=True, text=True, capture_output=True).stdout
    return int(out.strip())


def brand_mp4(src, dst, tmp=None):
    """Re-encode an MP4 with the mark (H.264, high quality, yuv420p, same frame rate)."""
    with tempfile.TemporaryDirectory(prefix='brand-') as t:
        w = video_width(src)
        mark = write_mark(w, Path(tmp or t) / f'mark-{w}.png')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-i', str(mark),
                        '-filter_complex', overlay(w) + ';[m]format=yuv420p[v]', '-map', '[v]',
                        '-c:v', 'libx264', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart',
                        str(dst)], check=True)
    return Path(dst)


def brand_existing_gif(src, dst):
    """Stamp a GIF rendered before the mark rule: overlay, then rebuild the palette."""
    with tempfile.TemporaryDirectory(prefix='brand-') as t:
        w = video_width(src)
        mark = write_mark(w, Path(t) / 'mark.png')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-i', str(mark),
                        '-filter_complex', overlay(w) + ';[m]format=rgb24,split[a][b];'
                        '[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=0[p];'
                        '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', str(dst)],
                       check=True)
    return Path(dst)
