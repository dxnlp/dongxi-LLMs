"""Report-only identity/criterion supplement; no tensor or model imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/dongxi/dongxi_ai/Dongxi_LLMs')
REPORT = ROOT/'experiments/reports'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    before = path.stat()
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024*1024):
            sha.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return dict(path=str(path), bytes=before.st_size, sha256=sha.hexdigest())


def main():
    native = read(REPORT/'2026-10-05-native-acceptance-independent.json')
    assert native['status'] == 'passed' and native['torch_imported'] is False
    identities = []
    base_cache = {}
    for mode, suffix in (('full', 'original'), ('full', 'resumed02'),
                         ('lora', 'original'), ('lora', 'resumed')):
        directory = ROOT/f'outputs/native-sft-{mode}-replay-20261005-{suffix}'
        paths = list(directory.glob('identity-*.json'))
        assert len(paths) == 1
        saved = read(paths[0])
        identity = saved['identity']
        assert saved['status'] == 'completed'
        assert identity['identity_sha256'] == canonical(
            {key: value for key, value in identity.items() if key != 'identity_sha256'})
        assert identity['git']['commit']['status'] == 'measured'
        assert identity['git']['dirty'] is True and identity['git']['status_sha256']
        assert identity['environment']['environment_lock']['status'] == 'hashed'
        assert identity['device']['name'] == 'NVIDIA GB10'
        assert identity['gpu_driver']['status'] == 'measured'
        assert identity['command'] and identity['checkpoint_files']
        config = identity['config']
        assert config['dtype'] == 'BF16 weights/autocast; FP32 cross entropy'
        assert config['loss_policy'] == 'assistant body + template end tokens; one explicit shift'
        assert config['mode'] == mode and config['updates'] == 20
        interface = identity['checkpoint_interface']
        assert interface['interface_sha256'] == canonical({key: interface[key] for key in
            ('schema_version', 'tokenizer', 'template_sha256', 'generation_stop_ids')})
        genealogy = read(directory/'policy/course-genealogy.json')
        assert genealogy['checkpoint_interface'] == interface
        assert genealogy['base_checkpoint_files'] == identity['checkpoint_files']
        assert genealogy['base_revision'] == config['revision']
        for relative, expected in identity['source_sha256'].items():
            assert digest(ROOT/relative)['sha256'] == expected
        for relative, expected in identity['input_sha256'].items():
            path = Path(relative)
            assert digest(path if path.is_absolute() else ROOT/path)['sha256'] == expected
        base = Path(identity['checkpoint_path'])
        for name, expected in identity['checkpoint_files'].items():
            path = base/name
            key = str(path)
            if key not in base_cache:
                base_cache[key] = digest(path)
            assert base_cache[key]['sha256'] == expected
        identities.append(dict(binding=digest(paths[0]), genealogy=digest(directory/'policy/course-genealogy.json'),
            mode=mode, invocation=suffix, canonical_identity_sha256=identity['identity_sha256'],
            recorded_git_commit=identity['git']['commit'], recorded_dirty=True,
            recorded_dirty_status_sha256=identity['git']['status_sha256'],
            recorded_environment=identity['environment'], recorded_device=identity['device'],
            recorded_driver=identity['gpu_driver'], command=identity['command'],
            interface=interface, loss_policy=config['loss_policy'],
            dtype=config['dtype'], objective_configuration_sha256=canonical(config),
            declared_upstream_revision=config['revision'], actual_local_base_files_verified=True,
            sources_and_inputs_current_bytes_verified=True))
    cpu = read(REPORT/'2026-10-04-run-identity.json')
    result = dict(schema='dongxi-original-dxi01-criteria-independent-v1',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), status='passed-original-scope-review',
        recommendation='Complete DXI-01 only against its four unchanged original acceptance criteria. '
            'Preserve all historical pending strings; scope-route campaign/model-quality and broader '
            'containment work to DXI-03/readiness rather than deleting or claiming it completed.',
        unchanged_acceptance=read(ROOT/'docs/course_improvements.json')['items'][0]['acceptance'],
        actual_native_identities=identities, independently_hashed_base_inventory=list(base_cache.values()),
        historical_cpu_identity_report=digest(REPORT/'2026-10-04-run-identity.json'),
        historical_cpu_identity_scope=cpu['scope'], historical_cpu_identity_checks=cpu['checks'],
        native_acceptance_binding=digest(REPORT/'2026-10-05-native-acceptance-independent.json'),
        criterion_mapping=[
            dict(criterion=1, conclusion='Actual four native SFT identity records independently verified; '
                'Git/dirty/source/input/command/environment/driver/device/objective/interface recorded. '
                'Full/LoRA parent Base ancestry is byte-bound; merge input ancestry separately verified.'),
            dict(criterion=2, conclusion='Historical actual CPU refusal/save-reload/explicit legacy-adoption '
                'checks retained; current unchanged tests/source inspected, not newly rerun here. '
                'Actual native interface retained through full/LoRA replay and explicit FP32 merged export.'),
            dict(criterion=3, conclusion='Declared upstream revision is separate from independently hashed '
                'local Base/tokenizer/adapter bytes. Historical partial-failure journaling checks and actual '
                'conflict/BF16 failures retained. Abrupt conflict child identity remains partial/running; '
                'returned external failure receipt is authoritative, not an invented completed child journal.'),
            dict(criterion=4, conclusion='Actual pinned native full checkpoint and rank8 Q/V LoRA exact '
                'fresh10-to20 recovery established. Explicit separately predeclared FP32 LoRA merge/reload '
                'has actual saved byte/interface proof and recorded eight-prefix numerical checks. '
                'Original BF16 merge fails; FP32 is not BF16 policy equivalence.')],
        not_established=['400-update pilot or selected downstream parent', 'actual DPO/RLVR pretrained chain',
            'Mac/hosted CI', 'whole-job physical hard quota', 'general model quality or safety',
            'cross-machine or arbitrary pickle metadata equivalence'],
        torch_imported='torch' in sys.modules, model_loaded=False, collector=digest(Path(__file__)))
    assert not result['torch_imported']
    output = REPORT/'2026-10-05-native-acceptance-criteria-independent.json'
    with output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'], identities=len(identities),
        actual_base_files=len(base_cache), torch_imported=result['torch_imported']), indent=2))


if __name__ == '__main__':
    main()
