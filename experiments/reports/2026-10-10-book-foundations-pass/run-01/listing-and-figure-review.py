"""Execute new standalone chapter listings and make a saved-figure review sheet.

This goal-local review consumer preserves canonical sources and old evidence.
The contact sheet supports human visual inspection; it does not automate it.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import json
import re
import subprocess

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent
baseline = json.loads((RUN / 'baseline.json').read_text())
records, figures = [], []
for path in sorted((ROOT / 'book/chapters').glob('*.md')):
    relative = str(path.relative_to(ROOT))
    before = subprocess.check_output(
        ['git', 'show', f"{baseline['base_commit']}:{relative}"], cwd=ROOT, text=True)
    after = path.read_text()
    pattern = r'(?ms)^```python\s*\n(.*?)^```\s*$'
    old_blocks = re.findall(pattern, before)
    for index, block in enumerate(re.findall(pattern, after)):
        if block in old_blocks:
            continue
        output = StringIO()
        with redirect_stdout(output):
            exec(compile(block, f'{relative}:listing-{index}', 'exec'), {})
        records.append(dict(chapter=relative, listing_index=index,
                            sha256=hashlib.sha256(block.encode()).hexdigest(),
                            status='passed', stdout=output.getvalue()))
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    old_images = {target for _, target in re.findall(image_pattern, before)}
    for alt, target in re.findall(image_pattern, after):
        if target in old_images:
            continue
        asset = (path.parent / target).resolve()
        figures.append(dict(chapter=relative, path=str(asset.relative_to(ROOT)),
                            sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                            alt=alt))

tile_width, tile_height = 900, 640
columns = 2
rows = (len(figures) + columns - 1) // columns
sheet = Image.new('RGB', (columns * tile_width, rows * tile_height), 'white')
draw = ImageDraw.Draw(sheet)
for index, record in enumerate(figures):
    x, y = (index % columns) * tile_width, (index // columns) * tile_height
    with Image.open(ROOT / record['path']) as original:
        panel = ImageOps.contain(original.convert('RGB'), (tile_width - 20, tile_height - 70))
    sheet.paste(panel, (x + (tile_width - panel.width) // 2, y + 60))
    draw.text((x + 12, y + 8), record['chapter'], fill='black')
    draw.text((x + 12, y + 28), Path(record['path']).name, fill='black')
sheet.save(RUN / 'promoted-figure-review.png')
receipt = dict(scope='New chapter listings executed verbatim on CPU; saved figures collected for human inspection',
               listings=records, figures=figures,
               figure_inspection='pending visual inspection; contact sheet is not an automated readability judgment',
               status='passed', contact_sheet='promoted-figure-review.png')
(RUN / 'listing-and-figure-review.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(status='passed', executed_new_listings=len(records), promoted_figures=len(figures))))
