"""Extract the original Figure 3 as a shared, white 16:9 player poster.

Requires pypdfium2 and Pillow. Does not modify either animation or source PDF.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import pypdfium2 as pdfium
from PIL import Image

P = Path(__file__).resolve().parent
REVISION = 'df42c109f1defefcbfcedbe7d905718a12266e40'
BASE = f'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash'
SOURCE = f'{BASE}/blob/{REVISION}/DeepSeek_V41_Tech_Report.pdf'
CROP = (68, 82, 528, 323)  # PDF points, top-left origin; excludes caption.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, help='Existing copy of the pinned report')
    args = parser.parse_args()
    raw = args.pdf.read_bytes() if args.pdf else urlopen(SOURCE.replace('/blob/', '/resolve/')).read()
    assets = P / 'assets'
    assets.mkdir(exist_ok=True)
    document = pdfium.PdfDocument(io.BytesIO(raw))
    page = document[6]
    scale = 5
    bitmap = page.render(scale=scale).to_pil().convert('RGB')
    figure = bitmap.crop(tuple(round(x * scale) for x in CROP))
    figure.thumbnail((1792, 952), Image.Resampling.LANCZOS)
    poster = Image.new('RGB', (1920, 1080), 'white')
    poster.paste(figure, ((1920-figure.width)//2, (1080-figure.height)//2))
    output = assets / 'deepseek-v41-architecture-poster.png'
    poster.save(output)
    license_url = f'{BASE}/raw/{REVISION}/LICENSE'
    (assets / 'LICENSE.deepseek.txt').write_bytes(urlopen(license_url).read())
    provenance = dict(
        title='DeepSeek-V4.1-Flash architecture, Figure 3', source_url=SOURCE,
        revision=REVISION, source_pdf_sha256=hashlib.sha256(raw).hexdigest(),
        pdf_page=7, crop_pdf_points_top_left=CROP, render_scale=scale,
        transformations='Crop Figure 3; uniform resize; center on a white 1920x1080 canvas. No redrawing.',
        license='MIT (source repository license, including associated documentation)',
        license_url=license_url, copyright='Copyright (c) 2023 DeepSeek',
        poster=output.name, poster_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        dimensions=[1920, 1080], pypdfium2_version=pdfium.PYPDFIUM_INFO.version,
    )
    (assets / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
