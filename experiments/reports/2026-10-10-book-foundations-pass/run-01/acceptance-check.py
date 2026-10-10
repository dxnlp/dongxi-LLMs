"""Consolidate final source-bound review, execution and preservation evidence.

Run only after the CPU panel and final documentation updates. This does not
replace the semantic reviews or execute tests; it checks their exact inputs.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent
def read(name):
    return json.loads((RUN / name).read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def check_hashes(hashes):
    return [name for name, digest in hashes.items()
            if not (ROOT/name).is_file() or sha(ROOT/name) != digest]

issues = []
baseline = read('baseline.json')
issues += check_hashes(baseline['protected_hashes'])
assert sha(RUN/'original-proposal.md') == baseline['initial_plan_sha256']
issues += check_hashes(read('lesson-freeze.json')['book_hashes'])
authors = {}
for row in read('depth-edits.json')['chapters']:
    authors[row['path']] = row['final_sha256']
authors.update(read('operations-edits.json')['owned_files'])
for row in read('routes-edits.json')['chapters']:
    authors[row['path']] = row['source_sha256']
issues += check_hashes(authors)

peer_sources, accepted_chapters = {}, set()
peers = []
for name, key, digest_key in [
    ('depth-peer-review.json', 'chapters', 'reviewed_sha256'),
    ('operations-peer-review.json', 'reviewed_sources', 'sha256'),
    ('routes-peer-review.json', 'sources', 'sha256'),
]:
    receipt = read(name)
    assert receipt['status'] == 'accepted'
    hashes = {row['path']: row[digest_key] for row in receipt[key]}
    issues += check_hashes(hashes)
    peer_sources.update(hashes)
    accepted_chapters.update(int(Path(path).name[:2]) for path in hashes
                             if path.startswith('book/chapters/'))
    peers.append(dict(receipt=name, accepted_sources=len(hashes), sha256=sha(RUN/name)))
assert accepted_chapters == set(range(1, 16))

reader = read('root-reader-review.json')
issues += check_hashes({row['path']: row['source_sha256'] for row in reader['chapters']})
assert all(all(row['acceptance'].values()) for row in reader['chapters'])
integration = read('integration-check.json')
assert integration['status'] == 'passed'
assert sum(row['before_sha256'] != row['after_sha256'] for row in integration['chapters']) == 15
assert all(not row['missing_original_question_lines'] for row in integration['chapters'])
assert all(row['preserved'] for row in integration['solutions'])
assert read('numerical-replay.json')['status'] == 'passed'
listings = read('listing-and-figure-review.json')
assert listings['status'] == 'passed'
assert {(row['chapter'], row['sha256']) for row in listings['listings']} == {
    (row['chapter'], row['sha256']) for row in integration['new_python_listings']}
assert read('visual-review.json')['status'] == 'passed'
issues += check_hashes({row['path']: row['sha256'] for row in listings['figures']})

cpu = read('final-cpu/cpu-verification.json')
assert cpu['status'] == 'passed'
assert len(cpu['commands']) == 6
assert all(row['exit_code'] == 0 and not row['timed_out'] for row in cpu['commands'])
for key in ['source_hashes', 'lesson_hashes', 'input_hashes']:
    issues += check_hashes(cpu[key])
assert not cpu['changed_sources_during_run'] and not cpu['changed_lessons_during_run']
notebooks = read('final-cpu/notebooks/manifest.json')
assert len(notebooks['notebooks']) == 76 and not notebooks['failures']
assert notebooks['minimum_observed_available_gib'] >= notebooks['required_host_reserve_gib']
issues += check_hashes({row['path']: row['sha256'] for row in notebooks['notebooks']})
for command in cpu['commands']:
    assert sha(Path(command['log'])) == command['log_sha256']
test_log = (RUN/'final-cpu/tests.log').read_text()
tests = int(re.search(r'Ran (\d+) tests', test_log).group(1))
assert not issues, issues

closure_paths = ['docs/BOOK_FOUNDATIONS_PLAN.md', 'BOOK.md', 'PROGRESS.md',
    'LEARNING_MEMORY.md', 'visuals/animations/PROPOSALS.md',
    'experiments/reports/2026-10-10-book-foundations-pass.md']
assert 'Status: complete.' in (ROOT/closure_paths[0]).read_text()
assert 'Status: complete.' in (ROOT/closure_paths[-1]).read_text()
artifacts = {str(path.relative_to(ROOT)): sha(path) for path in sorted(RUN.rglob('*'))
             if path.is_file() and path.name != 'completion.json'}
record = dict(schema_version=1, goal_id='BOOK-FOUNDATIONS-2026-10',
    date_utc=datetime.now(timezone.utc).isoformat(), base_commit=baseline['base_commit'],
    status='passed', accepted_chapters=sorted(accepted_chapters),
    peer_reviews=peers, author_source_hashes=authors, peer_source_hashes=peer_sources,
    preserved_baseline_files=len(baseline['protected_hashes']), preservation_issues=issues,
    local_targets_checked=integration['local_targets_checked'],
    preserved_original_numbered_lines=sum(row['original_numbered_question_lines'] for row in integration['chapters']),
    preserved_original_solution_ids=sum(len(row['original_ids']) for row in integration['solutions']),
    cpu_tests=tests, fresh_notebooks=len(notebooks['notebooks']),
    source_code_cells=sum(row['code_cells'] for row in notebooks['notebooks']),
    executed_reference_cells=sum(row['executed_code_cells'] for row in notebooks['notebooks']),
    preserved_unfinished_cells=sum(len(row['skipped_unfinished_exercise_cells']) for row in notebooks['notebooks']),
    emitted_images=sum(row['images'] for row in notebooks['notebooks']),
    minimum_observed_available_gib=notebooks['minimum_observed_available_gib'],
    cpu_source_hashes=cpu['source_hashes'], cpu_lesson_hashes=cpu['lesson_hashes'],
    closure_hashes={name: sha(ROOT/name) for name in closure_paths}, artifact_hashes=artifacts,
    limits='Agent reader judgments and actual Spark CPU checks; no measured reader study, learner mastery, new GPU campaign, Mac verification, hosted CI or publication')
(RUN/'completion.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({key: record[key] for key in ['status', 'accepted_chapters',
    'preserved_baseline_files', 'local_targets_checked', 'cpu_tests', 'fresh_notebooks',
    'executed_reference_cells', 'preserved_unfinished_cells', 'emitted_images']}))
