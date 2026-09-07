"""Export generated PNG outputs from a verified run into notebook preview assets.

Usage: python scripts/render_decoder_notebook_figures.py /tmp/chapter5-reference-...
This writes only named generated PNG assets, never notebook sources or learner
outputs. Run verify_decoder_notebooks.py first to reproduce the current plots.
"""
import argparse
import base64
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executed_directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.executed_directory/'manifest.json').read_text())
    if manifest.get('status') != 'passed' or len(manifest['notebooks']) != 11:
        raise ValueError('A successful eleven-notebook verification manifest is required')
    destination = root/'notebooks/figures/chapter-05'
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in manifest['notebooks']:
        source = Path(entry['path'])
        notebook = json.loads((args.executed_directory/source.parent.name/source.name).read_text())
        for cell in notebook['cells']:
            if 'visual-explanation' not in cell.get('metadata',{}).get('tags',[]):
                continue
            for output in cell.get('outputs',[]):
                data=output.get('data',{}).get('image/png')
                if data:
                    name=f'{source.parent.name}-{source.stem}-{cell["id"]}.png'
                    if Path(name).name != name:
                        raise ValueError('Invalid figure name')
                    (destination/name).write_bytes(base64.b64decode(data))
                    count += 1
    expected=sum(entry.get('rendered_figures',0) for entry in manifest['notebooks'])
    if not expected or count != expected:
        raise ValueError(f'Expected {expected} verified figures, found {count}')
    print(f'Exported {count} generated reference previews to {destination}')


if __name__ == '__main__':
    main()
