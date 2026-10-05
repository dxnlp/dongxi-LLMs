"""Bounded root checks of the original-requirements follow-up, not model pilots."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    env = dict(os.environ, PYTHONPATH='src:tests', CUDA_VISIBLE_DEVICES='',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    paths = sorted([*ROOT.glob('src/dongxi_llms/*.py'), *ROOT.glob('scripts/*.py'),
                    *ROOT.glob('tests/*.py')])
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    inputs = [ROOT / 'fixtures/reasoning-evaluation' / name for name in
              ('items.json', 'settings.json', 'contract.json', 'responses.jsonl')]
    inputs += [ROOT / 'fixtures/chapter11' / name for name in
               ('train.jsonl', 'validation.jsonl', 'evaluation.jsonl')]
    input_before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    commands, memory = [], []
    def reserve():
        available = next(int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:'))
        memory.append(available)
        if available < 25 * 1024**3:
            raise RuntimeError('Sampled host reserve below 25GiB before/after child')
    def run(name, arguments):
        reserve()
        started = time.monotonic()
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=60)
        log = output / (name + '.log')
        with log.open('x') as handle:
            handle.write(result.stdout + result.stderr)
        row = dict(name=name, command=[sys.executable, *arguments],
            exit_code=result.returncode, seconds=time.monotonic()-started,
            log=str(log.relative_to(ROOT)), log_sha256=sha(log))
        commands.append(row)
        reserve()
        if result.returncode:
            raise RuntimeError(name + ' failed; retained actual log')
        return result.stdout + result.stderr
    try:
        test_log = run('focused-tests', ['-m', 'unittest', 'test_evaluation_model_card',
            'test_reasoning_evaluation', 'test_chosen_sft_control', 'test_staged_campaign', '-q'])
        match = re.search(r'Ran (\d+) tests in ([0-9.]+)s', test_log)
        if not match or not test_log.rstrip().endswith('OK'):
            raise ValueError('Unittest completion footer missing')
        for label in ('card-a', 'card-b'):
            run(label, ['scripts/export_evaluation_model_card.py',
                '--items', 'fixtures/reasoning-evaluation/items.json',
                '--contract', 'fixtures/reasoning-evaluation/contract.json',
                '--records', 'fixtures/reasoning-evaluation/responses.jsonl',
                '--compare', 'authored-baseline', 'authored-candidate',
                '--draws', '2000', '--seed', '1010', '--output', str(output / label)])
        card_files = ('model-card.json', 'model-card.md', 'replay.json')
        card_hashes = {name: sha(output / 'card-a' / name) for name in card_files}
        if any(card_hashes[name] != sha(output / 'card-b' / name) for name in card_files):
            raise ValueError('Repeated same-input exports differ')
        card = json.loads((output / 'card-a/model-card.json').read_text())
        assert card['evaluation']['replayed_record_count'] == 30
        assert card['evaluation']['errors'] == 1
        assert card['approval']['value'] is None
        for model in card['models'].values():
            assert model['genealogy']['value'] is None
            assert model['unlinked_record_count'] == 15
            assert all(c['known_rows'] == 0 and c['unknown_rows'] == 15
                for c in model['recorded_costs'].values())
        run('math', ['scripts/check_book_math.py'])
        route_log = run('routes', ['scripts/check_course_integrity.py', '--json'])
        routes = json.loads(route_log)
        assert not routes['problems']
        ledger = json.loads((ROOT / 'docs/course_improvements.json').read_text())
        assert sum(x['status'] == 'complete' for x in ledger['items']) == 13
        assert 'Day9' in ledger['learner_position']
        from dongxi_llms.staged_campaign import prepare_campaign
        campaign = prepare_campaign()
        rows = campaign['stage_rows']
        assert len(rows) == 45 and campaign['jobs_started'] == 0
        assert campaign['actual_genealogy'] == []
        assert sum(len(row['actual']) for row in rows) == 945
        assert all(value is None for row in rows for value in row['actual'].values())
        after = {name: sha(ROOT / name) for name in before}
        assert before == after
        assert input_before == {name: sha(ROOT / name) for name in input_before}
        report = dict(status='passed', date_utc=datetime.now(timezone.utc).isoformat(),
            scope='Isolated Linux CPU focused tests and offline card replay; not full suite, notebooks, pretrained, CUDA, Mac or hosted CI',
            executable=sys.executable, commands=commands,
            tests=int(match[1]), unittest_seconds=float(match[2]),
            source_sha256_before=before, source_sha256_after=after,
            sources_unchanged=True, input_sha256=input_before, inputs_unchanged=True,
            card_bundle_sha256=card_hashes, repeated_export_byte_identical=True,
            required_reserve_gib=25, minimum_sampled_available_bytes=min(memory),
            memory_sampling='Before/after each child, not continuous containment',
            external_campaign=dict(rows=45, actual_null_fields=945, jobs=0, genealogy=[]),
            goal_complete=False, learner_position=ledger['learner_position'])
        with (output / 'verification.json').open('x') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print(json.dumps(dict(status='passed', tests=report['tests'], output=str(output))))
    except BaseException as error:
        with (output / 'failure.json').open('x') as handle:
            json.dump(dict(type=type(error).__name__, message=str(error), commands=commands), handle, indent=2)
        raise


if __name__ == '__main__':
    main()
