"""Explicit two-checkpoint evaluation; fixed inputs, no model download or sweep."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest,_native_probe,_supervise
from dongxi_llms.run_identity import artifact_hashes
EVIDENCE=Path(__file__).parent
PYTHON='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
BASE=Path('/home/dongxi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
SFT=ROOT/'outputs/native-base-profile-20261005-run01/policy'
INPUTS=(ROOT/'fixtures/reasoning-evaluation/items.json',EVIDENCE/'contract.json',
    ROOT/'experiments/configs/pretrained-evaluation-replay-20261005.json',
    ROOT/'experiments/specs/2026-10-05-pretrained-evaluation-replay.md')
SOURCES=tuple(ROOT/p for p in ('src/dongxi_llms/reasoning_generation.py',
    'src/dongxi_llms/reasoning_evaluation.py','src/dongxi_llms/run_identity.py',
    'src/dongxi_llms/sampling_likelihood_lab.py','scripts/generate_reasoning_records.py',
    'src/dongxi_llms/native_profile_supervisor.py'))+(Path(__file__),)


def retain(name,value):
    with (EVIDENCE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    bindings={str(p):_digest(str(p)) for p in INPUTS+SOURCES}
    model_bindings={label:artifact_hashes(path) for label,path in (('base',BASE),('sft20',SFT))}
    retain('prepared-pair.json',dict(bindings=bindings,models=model_bindings,
        scope='existing15itemdevelopmentinstrument; disposable20updateprofile notselectedSFTparent',
        max_attempts_per_child=15,max_new_tokens_per_child=960,
        external_seconds_per_child=900,reserve_bytes=25*1024**3))
    for label,checkpoint in (('base',BASE),('sft20',SFT)):
        if any(_digest(str(p))!=bindings[str(p)] for p in INPUTS+SOURCES):
            raise ValueError('Fixed input/source binding changed')
        if artifact_hashes(checkpoint)!=model_bindings[label]:
            raise ValueError('Actual checkpoint bytes changed')
        argv=[PYTHON,str(ROOT/'scripts/generate_reasoning_records.py'),
            '--items',str(INPUTS[0]),'--checkpoint',str(checkpoint),
            '--contract',str(EVIDENCE/'contract.json'),'--output',str(EVIDENCE/label),
            '--environment-lock','/home/dongxi/dgx-spark-dongxi/uv.lock','--allow-cuda']
        declaration='October5 goal-authorized realpretrained developmentevaluation afterpassednativeprofile; fixed15items/greedy64/BF16/localonly/900s/25GiB; rawresponses/errorsretained; notpublicationor400updatecomparison.'
        retain(label+'-launch.json',dict(argv=argv,bindings=bindings,
            checkpoint_files=model_bindings[label],operator_declaration=declaration))
        result=_supervise(argv,EVIDENCE/(label+'-supervision'),native=True,seconds=900,
            probe=_native_probe,operator_declaration=declaration)
        retain(label+'-returned-supervision.json',result)
        print(json.dumps(dict(label=label,status=result['status'],exit_code=result['actual_exit_code'],
            seconds=result['child_seconds'],minimum_available_bytes=result['minimum_sampled_available_bytes'])),flush=True)
        if result['status']!='completed' or result['actual_exit_code']!=0:
            raise RuntimeError('Actual generation invocation failed; raw evidence remains')
    return 0


if __name__=='__main__':raise SystemExit(main())
