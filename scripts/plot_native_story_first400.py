#!/usr/bin/env python3
"""CPU-only plot of the accepted first400 story tranche, never story-quality scores.

Run only after both fixed producers have ended. No Torch, tokenizer, weights,
GPU, network, future schedule samples or producer mutation are used.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'experiments/reports/native-story-first400-figures'
RUN_ID='20261005-01'
PYTHON='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
MAX_BYTES=64*1024**2
ARMS=('control','half-lr')
REQUIRED_SOURCES={'scripts/run_native_story_stages.py','scripts/run_deterministic_story_child.py',
    'scripts/train_stories.py','src/dongxi_llms/stories_training.py','src/dongxi_llms/pretraining_lab.py'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def read_bytes(path):
    path=Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=MAX_BYTES:
            raise ValueError('Bounded nonempty regular plot input required: '+str(path))
        chunks=[];count=0
        while chunk:=os.read(fd,1024**2):
            count+=len(chunk)
            if count>MAX_BYTES:raise ValueError('Plot input byte bound crossed')
            chunks.append(chunk)
        after=os.fstat(fd)
        if count!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Plot input changed while reading')
    finally:os.close(fd)
    raw=b''.join(chunks)
    return raw,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def parse_json(raw):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate plot-input JSON key')
            value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite plot-input JSON')))


class Inputs:
    def __init__(self):self.bindings={}
    def read(self,path):
        path=Path(path);raw,binding=read_bytes(path)
        key=str(path.relative_to(ROOT))
        if key in self.bindings and self.bindings[key]!=binding:raise ValueError('Plot input changed between reads')
        self.bindings[key]=binding;return raw
    def json(self,path):
        value=parse_json(self.read(path))
        if not isinstance(value,dict):raise ValueError('Plot receipt JSON object required')
        return value
    def verify(self):
        after={name:read_bytes(ROOT/name)[1] for name in self.bindings}
        if after!=self.bindings:raise ValueError('Actual plot inputs changed during rendering')
        return after


def expected_command(arm,evidence,output):
    peak,floor=('0.0003','0.00003') if arm=='control' else ('0.00015','0.000015')
    return [PYTHON,str(ROOT/'scripts/run_deterministic_story_child.py'),'train',
        '--data',str(ROOT/'data/cache/day09-full-v2'),'--output',str(output/'pilot'),
        '--device','cuda','--total','14000','--warmup','200','--microbatch','16','--accumulation','1',
        '--valid-windows','512','--sample-tokens','256','--checkpoint-every','400',
        '--max-seconds','14400','--peak-lr',peak,'--floor-lr',floor,'--valid-target-budget','50000000',
        '--work-limits',str(evidence/'work-caps.json'),'--work-journal',str(output/'pilot-work.jsonl'),
        '--stop-after','400']


def scheduled_lr(update,arm):
    """Original zero-based14k schedule; not a schedule compressed to400."""
    peak=3e-4 if arm=='control' else 1.5e-4;floor=peak/10
    step=update-1
    if step<200:return peak*(step+1)/200
    progress=(step-200+1)/(14000-200)
    return floor+.5*(peak-floor)*(1+math.cos(math.pi*progress))


def validate_rows(rows,completion,arm):
    if len(rows)!=400 or any(type(row.get('update')) is not int for row in rows) or [row['update'] for row in rows]!=list(range(1,401)):
        raise ValueError('Exactly400 ordered actual updates1..400 required')
    targets=positions=0
    for row in rows:
        for key in ('loss','lr','gradient_norm','seconds'):
            if type(row.get(key)) not in (int,float) or not math.isfinite(row[key]) or row[key]<=0:
                raise ValueError('Finite positive actual online metric required: '+key)
        if type(row.get('valid_targets')) is not int or not 0<row['valid_targets']<=16*1024:
            raise ValueError('Actual valid training labels required')
        if type(row.get('processed_positions')) is not int or row['processed_positions']!=16*1024:
            raise ValueError('Original16×1024 padded training positions required')
        targets+=row['valid_targets'];positions+=row['processed_positions']
        if (type(row.get('cumulative_targets')) is not int or row['cumulative_targets']!=targets or
                type(row.get('cumulative_processed_positions')) is not int or row['cumulative_processed_positions']!=positions or
                row.get('processed_positions_accounting')!='measured-input-numel'):
            raise ValueError('Actual cumulative target/position accounting differs')
        if not math.isclose(row['lr'],scheduled_lr(row['update'],arm),rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError('Actual LR differs from original14000/200-warmup schedule')
    if (any(type(completion.get(key)) is not int for key in ('completed_updates','valid_target_budget',
            'cumulative_targets','cumulative_processed_positions')) or
            completion.get('completed_updates')!=400 or completion.get('requested_stop_reached') is not True or
            completion.get('schedule_complete') is not False or completion.get('valid_target_budget')!=50_000_000 or
            completion.get('stopped_for_target_budget') is not False or completion.get('stopped_for_time_budget') is not False or
            completion.get('cumulative_targets')!=targets or completion.get('cumulative_processed_positions')!=positions or
            completion.get('processed_positions_accounting')!='measured-input-numel'):
        raise ValueError('Actual requested first400 boundary with incomplete14000 schedule required')
    completed=completion.get('work_ledger',{}).get('completed',{})
    if (completed.get('train_updates')!=400 or completed.get('training_valid_targets')!=targets or
            completed.get('training_windows')!=400*16 or completion['work_ledger'].get('open_tickets') or
            completion['work_ledger'].get('failed_tickets')):
        raise ValueError('Accepted completed training work differs from metric trajectory')


def read_arm(arm,inputs):
    stem=f'native-story-{arm}-pilot-{RUN_ID}'
    evidence=ROOT/'experiments/reports'/stem;output=ROOT/'outputs'/stem
    acceptance=inputs.json(evidence/'acceptance.json')
    if (acceptance.get('status')!='passed' or acceptance.get('stage')!='story-'+arm+'-pilot' or
            acceptance.get('completion_level')!='first400-tranche-not-full14000' or
            acceptance.get('full_schedule_complete') is not False):
        raise ValueError('Actual accepted fixed first400 story arm required')
    prepared=inputs.json(evidence/'preparation.json')
    preparation_sha=canonical_hash({key:value for key,value in prepared.items() if key!='preparation_sha256'})
    if (prepared.get('preparation_sha256')!=preparation_sha or acceptance.get('preparation_sha256')!=preparation_sha or
            prepared.get('stage')!='story-'+arm+'-pilot' or prepared.get('run_id')!=RUN_ID or
            acceptance.get('bindings_after')!=prepared.get('bindings') or
            prepared.get('execution_tranche',{}).get('requested_stop')!=400 or
            prepared['execution_tranche'].get('total_horizon')!=14000 or
            prepared['execution_tranche'].get('full_schedule_completion') is not False or
            prepared.get('commands')!={'pilot':expected_command(arm,evidence,output)}):
        raise ValueError('Actual source-bound preparation/accepted first400 command differs')
    sources=prepared['bindings'].get('sources',{})
    if not REQUIRED_SOURCES<=set(sources):raise ValueError('Actual producer source bindings required')
    for name,binding in sources.items():
        path=ROOT/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or not isinstance(binding,dict) or
                binding.get('path')!=str(path) or re.fullmatch(r'[0-9a-f]{64}',str(binding.get('sha256',''))) is None):
            raise ValueError('Actual declared producer source path/SHA required')
        raw=inputs.read(path)
        if hashlib.sha256(raw).hexdigest()!=binding['sha256'] or len(raw)!=binding.get('bytes'):
            raise ValueError('Actual bound producer source bytes changed')
    launch=inputs.json(evidence/'launch-pilot.json');returned=inputs.json(evidence/'returned-supervision-pilot.json')
    terminal=inputs.json(evidence/'supervision-pilot/result.json')
    command=prepared['commands']['pilot']
    if launch.get('argv')!=command or launch.get('preparation_sha256')!=preparation_sha:
        raise ValueError('Actual pilot launch does not join preparation')
    fields=('status','actual_exit_code','child_pid','child_command','stop_reason','actual_native_profile_executed','limits')
    invocations=acceptance.get('invocations',[])
    if (any(terminal.get(key)!=returned.get(key) for key in fields) or returned.get('status')!='completed' or
            returned.get('actual_exit_code')!=0 or returned.get('actual_native_profile_executed') is not True or
            type(returned.get('child_pid')) is not int or returned['child_pid']<=0 or returned.get('child_command')!=command or
            returned.get('cleanup_errors') or returned.get('limits',{}).get('external_seconds')!=14400 or
            len(invocations)!=1 or invocations[0].get('label')!='pilot' or
            any(invocations[0].get(key)!=returned.get(key) for key in ('status','actual_exit_code','child_pid'))):
        raise ValueError('Actual completed native child/acceptance join required')
    first=inputs.read(evidence/'supervision-pilot/stdout.txt').split(b'\n',1)[0]
    entry=parse_json(first)
    if (not isinstance(entry,dict) or entry.get('event')!='actual-story-deterministic-entry' or
            entry.get('deterministic_algorithms') is not True or entry.get('cublas_workspace')!=':4096:8' or
            entry.get('attention_backend')!='SDPA MATH only' or entry.get('tf32') is not False):
        raise ValueError('Actual deterministic MATH producer entry required')
    completion=inputs.json(output/'pilot/completion.json')
    raw=inputs.read(output/'pilot/metrics.jsonl')
    if not raw.endswith(b'\n'):raise ValueError('Incomplete actual training metric tail')
    rows=[parse_json(line) for line in raw.splitlines()]
    if any(not isinstance(row,dict) for row in rows):raise ValueError('Actual training metric objects required')
    validate_rows(rows,completion,arm)
    return dict(rows=rows,bindings=prepared['bindings'],producer_entry=entry,preparation_sha256=preparation_sha,
        completion={key:completion[key] for key in ('completed_updates','cumulative_targets','cumulative_processed_positions',
            'requested_stop_reached','schedule_complete')})


def load_inputs():
    inputs=Inputs();arms={arm:read_arm(arm,inputs) for arm in ARMS}
    if arms['control']['bindings']!=arms['half-lr']['bindings']:
        raise ValueError('Actual producer data/source/environment bindings differ between arms')
    fields=('update','valid_targets','cumulative_targets','processed_positions','cumulative_processed_positions')
    if [[row[key] for key in fields] for row in arms['control']['rows']]!=[[row[key] for key in fields] for row in arms['half-lr']['rows']]:
        raise ValueError('Actual first400 label/position sequence differs between fixed-seed arms')
    return inputs,arms


def render(arms,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    try:
        for arm,label,color in (('control','Control LR','black'),('half-lr','Half LR','#B56720')):
            rows=arms[arm]['rows']
            axes[0].plot([row['cumulative_targets'] for row in rows],[row['loss'] for row in rows],
                label=label,color=color,linewidth=1.25)
            axes[1].plot([row['update'] for row in rows],[row['lr'] for row in rows],
                label=label,color=color,linewidth=1.25)
        axes[0].set_xlabel('Cumulative valid training-label presentations')
        axes[0].set_ylabel('Online batch NLL (nats/valid label)')
        axes[0].set_title('Measured training batches — not story quality',fontsize=10)
        axes[1].set_xlabel('Completed update (first400 only)');axes[1].set_ylabel('Learning rate')
        axes[1].set_title('Original14,000-update schedule;200-update warmup',fontsize=10)
        axes[1].axvline(200,color='gray',linestyle=':',linewidth=.8);axes[1].set_xlim(1,400)
        for axis in axes:
            axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18)
        axes[0].legend(frameon=False)
        fig.suptitle('Same data/order/label exposure; first400 tranche, not completed14k training',fontsize=11)
        fig.savefig(path,dpi=170)
    finally:plt.close(fig)


def retain(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise FileExistsError('Actual plot output must be new: '+str(OUTPUT))
    source=Path(__file__).resolve();source_before=read_bytes(source)[1]
    inputs,arms=load_inputs();inputs.verify()
    OUTPUT.mkdir(mode=0o700,exist_ok=False)
    try:
        plot=OUTPUT/'learning-curves.png';render(arms,plot)
        after=inputs.verify()
        if read_bytes(source)[1]!=source_before:raise ValueError('Plot consumer source changed during rendering')
        retain(OUTPUT/'inputs.json',dict(status='completed',input_bindings_before=inputs.bindings,
            input_bindings_after=after,input_sha256={name:value['sha256'] for name,value in inputs.bindings.items()},
            plot_sha256=read_bytes(plot)[1]['sha256'],source_sha256=source_before['sha256'],
            observed_arms={arm:{key:value for key,value in record.items() if key not in ('rows','bindings')} for arm,record in arms.items()},
            schedule=dict(original_total_updates=14000,warmup_updates=200,plotted_updates=400,full_schedule_complete=False),
            scope='Actual online training batch NLL and observed LR only; matched valid-label presentations and padded positions. Not held-out NLL, story ratings, completed14k/50M training, causal LR superiority or universal recipe ranking.',
            cpu_boundary='Small receipt/source/metric reads and Matplotlib rendering only; no weights, model, tokenizer, Torch/GPU/network or producer-journal work'))
        inputs.verify()
        if read_bytes(source)[1]!=source_before:raise ValueError('Plot consumer source changed during receipt write')
    except BaseException as error:
        retain(OUTPUT/'failure.json',dict(type=type(error).__name__,message=str(error),
            input_bindings_before=inputs.bindings,source_sha256=source_before['sha256'],
            scope='Invalid/incomplete plot prefix retained; producer receipts and any image remain unchanged'))
        raise
    print(plot)
    return 0


if __name__=='__main__':raise SystemExit(main())
