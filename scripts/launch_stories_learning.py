#!/usr/bin/env python3
"""Wait for audited data, verify launch gates, then run one fixed learning recipe."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch(config_path):
    config = json.loads(Path(config_path).read_text())
    status_path = Path(config['status_file'])
    status_path.parent.mkdir(parents=True, exist_ok=False)
    def status(stage, **extra):
        record = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(status_path)
        print(json.dumps(record), flush=True)
    try:
        status('waiting_for_full_data', data=config['data'])
        start = time.monotonic()
        manifest_path = Path(config['data'])/'manifest.json'
        while not manifest_path.exists():
            active = subprocess.check_output(['systemctl','--user','show',config['data_service'],
                                               '-p','ActiveState','--value'], text=True).strip()
            if active not in ('active','activating'):
                raise RuntimeError('Data preparation ended without a complete manifest; training NOT launched')
            if time.monotonic()-start > config['data_wait_seconds']:
                raise TimeoutError('Data preparation wait budget exhausted; training NOT launched')
            time.sleep(5)
        manifest = json.loads(manifest_path.read_text())
        if manifest['subset'] or manifest['train']['documents'] < 100000 or manifest['valid']['documents'] < 10000:
            raise RuntimeError('Full-data gate failed')
        for name, expected in config['sources'].items():
            if sha256(name) != expected:
                raise RuntimeError(f'Code changed after launch preparation: {name}')
        if not json.loads(Path(config['recovery_report']).read_text())['bitwise_identical']:
            raise RuntimeError('Recovery gate failed')
        if shutil.disk_usage('.').free < 100*1024**3:
            raise RuntimeError('Less than100GiB disk available')
        gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
        if gpu:
            raise RuntimeError('GPU has another compute process; training NOT launched')
        status('checking_regressions', train_documents=manifest['train']['documents'])
        tests = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],
                               capture_output=True, text=True, timeout=180)
        (status_path.parent/'tests.txt').write_text(tests.stdout+tests.stderr)
        if tests.returncode:
            raise RuntimeError('Regression tests failed; training NOT launched')
        # Runtime validates data-file hashes and checks memory before allocating the model.
        command = [sys.executable,'scripts/train_stories.py','train','--data',config['data'],
                   '--output',config['output'], *config['training_args']]
        status('training_starting', command=command,
               train_documents=manifest['train']['documents'],
               available_training_targets=manifest['train']['valid_targets'])
        result = subprocess.run(command, timeout=14700)
        status('training_exited', exit_code=result.returncode, output=config['output'])
        return result.returncode
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    raise SystemExit(launch(parser.parse_args().config))
