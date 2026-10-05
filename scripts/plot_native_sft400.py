#!/usr/bin/env python3
"""Plot measured fixed-recipe learning curves, never fabricated model outputs."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'experiments/reports/native-sft400-comparison-figures'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bindings={}
    rows={}
    for mode in ('full','lora'):
        stem=f'native-sft-{mode}-pilot400-20261005-run-01'
        acceptance=ROOT/'experiments/reports'/stem/'acceptance.json'
        metrics=ROOT/'outputs'/stem/'metrics.jsonl'
        accepted=json.loads(acceptance.read_text())
        values=[json.loads(line) for line in metrics.read_text().splitlines()]
        if (accepted['status']!='passed' or accepted['result']['updates']!=400
                or [row['update'] for row in values]!=list(range(1,401))
                or values[-1]['cumulative_supervised_targets']!=9321
                or any(not math.isfinite(row[key]) or row[key]<=0 for row in values
                    for key in ('answer_nll','gradient_norm'))):
            raise ValueError('Complete actual finite original400 metrics required')
        rows[mode]=values
        for path in (acceptance,metrics):
            bindings[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    OUTPUT.mkdir(mode=0o700,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for mode,label,color in (('full','Full','black'),('lora','Rank8 Q/V LoRA','#B56720')):
        x=[row['cumulative_supervised_targets'] for row in rows[mode]]
        for axis,key,title in zip(axes,('answer_nll','gradient_norm'),
                ('Training answer NLL (nats/label)','Pre-clip gradient norm')):
            axis.semilogy(x,[row[key] for row in rows[mode]],label=label,color=color,linewidth=1.25)
            axis.set_xlabel('Cumulative successful training labels')
            axis.set_ylabel(title)
            axis.spines[['top','right']].set_visible(False)
            axis.grid(axis='y',alpha=.18)
    axes[0].legend(frameon=False)
    fig.suptitle('Same9,321 labels; different trainable update spaces — one fixed recipe/seed',fontsize=11)
    fig.savefig(OUTPUT/'learning-curves.png',dpi=170)
    plt.close(fig)
    for name,expected in bindings.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Actual plot inputs changed during rendering')
    with (OUTPUT/'inputs.json').open('x') as handle:
        json.dump(dict(status='completed',input_sha256=bindings,
            plot_sha256=hashlib.sha256((OUTPUT/'learning-curves.png').read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Measured training curves, not publication accuracy or universal method ranking'),handle,indent=2)
        handle.write('\n')
    print(OUTPUT/'learning-curves.png')


if __name__=='__main__':
    main()
