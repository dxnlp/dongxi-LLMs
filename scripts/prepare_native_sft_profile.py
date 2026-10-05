"""Prepare one native SFT profile; default does not spawn any process.

--execute is a separately scoped operator invocation, not authenticated permission.
Course source verification never uses it and never loads pretrained weights.
"""
import argparse
import json
from pathlib import Path

from dongxi_llms.native_profile_supervisor import (
    STAGE, MODEL, REVISION, RECIPE, prepare_native_profile, execute_native_profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('interpreter', 'train', 'dev', 'template', 'environment_lock',
                'work_limits', 'snapshot_io_limits', 'work_journal',
                'snapshot_io_ledger', 'output'):
        parser.add_argument('--'+key.replace('_', '-'), required=True)
    parser.add_argument('--snapshot-max-bytes', type=int, required=True)
    parser.add_argument('--work-journal-max-bytes', type=int, required=True)
    parser.add_argument('--preparation-record', type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicit operator-only native launch; never automatic approval')
    parser.add_argument('--operator-declaration', help='Retained operator scope text, NOT authenticated authority')
    parser.add_argument('--supervision-output', help='Separate new watchdog evidence directory')
    args = parser.parse_args(argv)
    if args.execute and (args.operator_declaration is None or args.supervision_output is None):
        parser.error('--execute requires scoped --operator-declaration and --supervision-output')
    if not args.execute and (args.operator_declaration is not None or args.supervision_output is not None):
        parser.error('Execution fields must not silently change preparation mode')
    request = {key: value for key, value in vars(args).items() if key not in (
        'preparation_record', 'execute', 'operator_declaration', 'supervision_output')}
    request.update(stage_id=STAGE, model=MODEL, revision=REVISION,
                   tokenizer_revision=REVISION, recipe=dict(RECIPE))
    prepared = prepare_native_profile(request)
    if args.preparation_record:
        with args.preparation_record.open('x') as handle:
            handle.write(json.dumps(prepared, indent=2, sort_keys=True)+'\n')
    print(json.dumps(prepared, indent=2, sort_keys=True))
    if args.execute:
        result = execute_native_profile(prepared, operator_declaration=args.operator_declaration,
                                        supervision_output=args.supervision_output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result['status'] == 'completed' else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
