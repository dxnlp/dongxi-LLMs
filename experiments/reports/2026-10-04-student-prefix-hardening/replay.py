"""Predeclared hardening replay; preserve the original measurement and failures."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
import torch
from dongxi_llms.student_prefix_lab import run_reference,validate_contract
from dongxi_llms.reasoning_generation import GenerationJournal
from dongxi_llms.run_identity import collect_run_identity


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical(value):
    if isinstance(value,dict):
        return {k:numerical(v) for k,v in value.items()
                if not k.endswith('seconds') and k!='payload_sha256'}
    if isinstance(value,list):return [numerical(v) for v in value]
    return value


def main():
    original=ROOT/'experiments/reports/2026-10-04-student-prefix-distillation'
    directory=Path(__file__).resolve().parent/'replay'
    if directory.exists():raise FileExistsError('New replay directory required')
    journal=GenerationJournal(directory)
    files=[ROOT/p for p in ('src/dongxi_llms/student_prefix_lab.py','tests/test_student_prefix_lab.py',
        'src/dongxi_llms/decoder_lab.py','src/dongxi_llms/teacher_data_lab.py',
        'src/dongxi_llms/reasoning_generation.py','src/dongxi_llms/run_identity.py',
        'fixtures/student-prefix/items.json','experiments/specs/2026-10-04-student-prefix-distillation.md',
        'experiments/specs/2026-10-04-student-prefix-hardening.md')]+[Path(__file__).resolve()]
    original_files=[original/name for name in ('results.json','contract.json','input-identity.json','responses.jsonl','events.jsonl')]
    before={str(p.relative_to(ROOT)):digest(p) for p in [*files,*original_files]}
    started=time.perf_counter()
    try:
        identity=collect_run_identity(ROOT,source_files=files[:6]+[Path(__file__).resolve()],input_files=files[6:9]+original_files)
        journal.write_new('input-identity.json',identity)
        fixture=json.loads(files[6].read_text());contract=json.loads((original/'contract.json').read_text())
        validate_contract(contract,fixture);journal.write_new('contract.json',contract)
        torch.set_num_threads(1)
        result=run_reference(fixture,contract,journal=journal)
        journal.write_new('results.json',result)
        historical=json.loads((original/'results.json').read_text())
        runs_equal=numerical(result['campaigns'])==numerical(historical['campaigns'])
        controls_equal=numerical(result['controls'])==numerical(historical['controls'])
        new=[json.loads(line) for line in (directory/'responses.jsonl').read_text().splitlines()]
        old=[json.loads(line) for line in (original/'responses.jsonl').read_text().splitlines()]
        records_equal=numerical(new)==numerical(old)
        unchanged=all(digest(ROOT/p)==h for p,h in before.items())
        verification={'schema':'dongxi-student-prefix-hardening-replay-v1',
            'original_and_current_hashes':before,'original_artifacts_unchanged':unchanged,
            'all_three_campaigns_eight_arms_equal':runs_equal,'exact_controls_equal':controls_equal,
            'all_raw_response_records_equal':records_equal,'response_records':len(new),
            'updates':sum(a['updates'] for c in result['campaigns'] for a in c['arms']),
            'comparison_exclusions':['keys ending seconds','payload_sha256 derived from durations'],
            'memory_and_provenance_boundary':'Campaign body memory/identity outside compared campaigns; original artifacts bytewise checked',
            'wall_seconds':time.perf_counter()-started,'actual_invocation_identity':identity}
        journal.write_new('comparison.json',verification)
        if not all((runs_equal,controls_equal,records_equal,unchanged)):
            raise AssertionError('Original/current numerical or bytewise preservation comparison failed')
        journal.event('completed',updates=verification['updates'],response_records=len(new))
        print(json.dumps({k:v for k,v in verification.items() if k not in ('original_and_current_hashes','actual_invocation_identity')},indent=2))
    except (Exception,KeyboardInterrupt) as error:
        journal.event('failure',error_type=type(error).__name__,error=str(error))
        journal.write_new('failure.json',{'type':type(error).__name__,'message':str(error),'partial_work_retained':True})
        raise
    finally:journal.close()


if __name__=='__main__':main()
