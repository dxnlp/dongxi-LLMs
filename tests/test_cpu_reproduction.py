"""CPU reproduction source contracts, failure retention and lock boundaries."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib
import unittest

import psutil
import yaml
from scripts.run_cpu_verification import CPU_VERIFICATION_SCOPE, execute

ROOT = Path(__file__).resolve().parents[1]


class CPUReproductionTests(unittest.TestCase):
    def test_scope_follows_the_recorded_host_without_claiming_other_platforms(self):
        self.assertIn('recorded invocation platform', CPU_VERIFICATION_SCOPE)
        self.assertIn('not other-platform reproduction', CPU_VERIFICATION_SCOPE)
        self.assertIn('pretrained/GPU outcomes', CPU_VERIFICATION_SCOPE)
        self.assertIn('learner mastery', CPU_VERIFICATION_SCOPE)
        self.assertNotIn('not hosted CI/Mac', CPU_VERIFICATION_SCOPE)

    def test_teaching_lock_cpu_linux_and_mac_wheels(self):
        project = tomllib.loads((ROOT/'pyproject.toml').read_text())
        lock = tomllib.loads((ROOT/'uv.lock').read_text())
        self.assertEqual(project['tool']['uv']['sources']['torch'][0]['marker'], "sys_platform == 'linux'")
        index = project['tool']['uv']['index'][0]
        self.assertTrue(index['explicit'])
        self.assertEqual(index['url'], 'https://download.pytorch.org/whl/cpu')
        cpu = next(p for p in lock['package'] if p['name']=='torch' and p['version'].endswith('+cpu'))
        self.assertEqual(cpu['source']['registry'], index['url'])
        self.assertFalse(any(p['name'].startswith('nvidia-') for p in lock['package']))
        self.assertTrue(any('aarch64' in w['url'] for w in cpu['wheels']))
        pypi = next(p for p in lock['package'] if p['name']=='torch' and p['source']['registry']=='https://pypi.org/simple')
        self.assertTrue(any('macosx' in w['url'] and 'arm64' in w['url'] for w in pypi['wheels']))
        for package in lock['package']:
            for wheel in package.get('wheels', []):
                self.assertRegex(wheel['hash'], r'^sha256:[0-9a-f]{64}$')

    def test_workflow_is_read_only_pinned_cpu_and_retains_failures(self):
        workflow = yaml.load((ROOT/'.github/workflows/course-cpu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(workflow['permissions'], {'contents':'read'})
        job = workflow['jobs']['references']
        self.assertEqual(job['strategy']['matrix']['os'], ['ubuntu-24.04','macos-15'])
        self.assertEqual(job['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(job['env']['HF_HUB_OFFLINE'], '1')
        for step in job['steps']:
            if 'uses' in step:
                self.assertRegex(step['uses'], r'^[\w/-]+@[0-9a-f]{40}$')
        checkout = job['steps'][0]
        self.assertEqual(checkout['with']['persist-credentials'], 'false')
        install = next(s['run'] for s in job['steps'] if s.get('name','').startswith('Install locked'))
        self.assertIn('--locked --extra course', install)
        upload = job['steps'][-1]
        self.assertEqual(upload['if'], 'always()')
        text = (ROOT/'.github/workflows/course-cpu.yml').read_text()
        self.assertNotIn('pull_request_target', text)
        self.assertNotIn('secrets.', text)

    def test_command_failure_and_timeout_outputs_are_kept(self):
        with tempfile.TemporaryDirectory(prefix='dongxi-command-test-') as directory:
            log = Path(directory)/'failure.log'
            result = execute([sys.executable, '-c', "import sys; print('retained'); sys.exit(7)"],
                             log, env=dict(os.environ), timeout=10)
            self.assertEqual(result['exit_code'], 7)
            self.assertFalse(result['timed_out'])
            self.assertIn('retained', log.read_text())
            result = execute([sys.executable, '-c', "import time; print('before-timeout',flush=True); time.sleep(10)"],
                             Path(directory)/'timeout.log', env=dict(os.environ), timeout=0.5)
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['exit_code'], 0)
            self.assertIn('before-timeout', (Path(directory)/'timeout.log').read_text())
            self.assertLess(result['seconds'], 8)

    def test_missing_executable_is_evidence_not_a_pass(self):
        with tempfile.TemporaryDirectory(prefix='dongxi-command-test-') as directory:
            result = execute(['/does-not-exist/dongxi-test'], Path(directory)/'missing.log',
                             env=dict(os.environ), timeout=10)
            self.assertIsNone(result['exit_code'])
            self.assertIn('FileNotFoundError', Path(result['log']).read_text())

    def test_timeout_stops_descendant_in_an_independent_session(self):
        with tempfile.TemporaryDirectory(prefix='dongxi-command-test-') as directory:
            code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); print(p.pid,flush=True); time.sleep(60)"
            log = Path(directory)/'descendant.log'
            result = execute([sys.executable,'-c',code], log, env=dict(os.environ), timeout=0.5)
            self.assertTrue(result['timed_out'])
            child_pid = int(log.read_text().splitlines()[0])
            try:
                child = psutil.Process(child_pid)
                self.assertEqual(child.status(), psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    unittest.main()
