"""Fixed owned-child supervision for the separately declared merge diagnostic."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest,_native_probe,_supervise
EVIDENCE=Path(__file__).parent

def main():
    command=['/home/dongxi/dgx-spark-dongxi/.venv/bin/python',str(EVIDENCE/'merge-child.py')]
    bindings={str(p):_digest(str(p)) for p in (Path(__file__),EVIDENCE/'merge-child.py',
        ROOT/'experiments/specs/2026-10-05-native-lora-merge-reload.md',
        ROOT/'src/dongxi_llms/native_profile_supervisor.py')}
    with (EVIDENCE/'launch.json').open('x') as f:json.dump(dict(argv=command,bindings=bindings),f,indent=2);f.write('\n')
    result=_supervise(command,EVIDENCE/'supervision',native=True,seconds=900,probe=_native_probe,
        operator_declaration='October5 goal-authorized exactBase20updateLoRA merge/reload diagnostic; originalbyteinterface/tolerancesfixed;localonly/900s/25GiB/nopilot/nopublication')
    with (EVIDENCE/'returned-supervision.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','actual_exit_code','child_seconds','minimum_sampled_available_bytes')}))
    return 0 if result['status']=='completed' and result['actual_exit_code']==0 else 1

if __name__=='__main__':raise SystemExit(main())
