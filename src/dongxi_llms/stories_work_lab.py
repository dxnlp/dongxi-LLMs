"""Native story-work microscope, not a corpus/model-scale training launcher."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from .batched_cache_lab import digest
from .decoder_lab import DecoderConfig
from .stories_data import IGNORE
from .stories_training import Recipe, Session, session_contract
from .story_work_budget import (STORY_WORK_KEYS, StoryWorkBudget, read_story_receipt,
                                story_work_contract)
from .work_budget import WorkBudgetExceeded


class OrderedWindows:
    """The original boundary geometry: four authored ID windows, not stories."""
    identity = 'original-story-budget-ids-lengths-3-6-1-7-v1'
    length = 8

    def __init__(self):
        self.x = torch.full((4, 8), 15, dtype=torch.long)
        self.y = torch.full((4, 8), IGNORE, dtype=torch.long)
        for row, count in enumerate((3, 6, 1, 7)):
            self.x[row, :count] = torch.tensor([14, *range(1, count)])
            self.y[row, :count] = torch.tensor([*range(1, count), 15])

    def __len__(self):
        return 4

    def batch(self, indices):
        return self.x[indices], self.y[indices]


def _fixture():
    config = DecoderConfig(vocab=16, width=8, heads=2, kv_heads=2, head_dim=4,
        layers=1, hidden=16, max_length=8, modern=True, qk_norm=False, tied=True)
    recipe = Recipe(total_updates=5, warmup=1, peak_lr=.003, floor_lr=.0003,
        accumulation=2, seed=909, activation_checkpointing=False)
    return config, recipe, OrderedWindows()


def _session(config, recipe, data, budget=None):
    result = Session(config, recipe, data, valid_target_budget=1000, work_budget=budget)
    result.stream.order = torch.arange(4)
    return result


def _numerical(session):
    return deepcopy(dict(model=session.model.state_dict(), optimizer=session.optimizer.state_dict(),
        stream=session.stream.state_dict(), rng=torch.get_rng_state(),
        gradients=[p.grad for p in session.model.parameters()], training=session.model.training,
        updates=session.step, targets=session.tokens, positions=session.positions,
        position_accounting=session.positions_accounting))


def _record(row):
    """All original numerical fields, excluding only clocks and resource metadata."""
    names = ('update', 'loss', 'lr', 'gradient_norm', 'valid_targets', 'cumulative_targets',
        'first_q_max_parameter_change', 'processed_positions', 'cumulative_processed_positions',
        'processed_positions_accounting')
    return {name:row[name] for name in names}


def work_recovery_example(attempt_target_cap=25):
    """Run the predeclared 9/fail8/restore/retry8 sequence; restore caller RNG.

    Capacity is explicitly authored for this microscope. Loader planning, byte
    I/O, serialization and physical resources are outside these logical units.
    Temporary files belong only to this invocation and are not learner outputs.
    """
    if type(attempt_target_cap) is not int or attempt_target_cap not in (24, 25):
        raise ValueError('This authored lesson compares only attempted-target capacities24 and25')
    with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(prefix='dongxi-story-clocks-') as temporary:
        config, recipe, data = _fixture()
        original = _session(config, recipe, data)
        reference_records = [_record(original.update()), _record(original.update())]
        reference_sha = digest(_numerical(original))
        capacities = dict.fromkeys(STORY_WORK_KEYS, 1000)
        capacities['training_valid_targets'] = attempt_target_cap
        contract = story_work_contract(capacities, 1024**2)
        science = session_contract(config, recipe, data, valid_target_budget=1000,
                                   work_contract=contract)
        root = Path(temporary)
        journal = root/'model-work.jsonl'
        boundary = root/'nine-targets.pt'
        timeline = []

        def event(label, session, budget):
            state = budget.ledger.snapshot()
            timeline.append(dict(event=label, successful_targets=session.tokens,
                reserved_targets=state['reserved']['training_valid_targets'],
                completed_updates=session.step, journal_sequence=state['sequence']))

        budget = StoryWorkBudget.create(journal, contract=contract, scientific_contract=science,
                                       invocation_id='lesson-original')
        try:
            actual = _session(config, recipe, data, budget)
            first = _record(actual.update())
            if first != reference_records[0]:
                raise AssertionError('First native update changed its original numerical record')
            event('complete9', actual, budget)
            actual.save(boundary)
            receipt = read_story_receipt(str(boundary)+'.work.json')
            calls = []
            native_loss = actual.summed_loss

            def fail_second_forward(x, y):
                calls.append(int(x.numel()))
                if len(calls) == 2:
                    raise RuntimeError('declared failure before second forward')
                return native_loss(x, y)

            with patch.object(actual, 'summed_loss', side_effect=fail_second_forward):
                try:
                    actual.update()
                except RuntimeError as error:
                    if str(error) != 'declared failure before second forward':
                        raise
                else:
                    raise AssertionError('The declared entered-update failure must occur')
            if len(calls) != 2 or not any(p.grad is not None for p in actual.model.parameters()):
                raise AssertionError('Failure must follow a real native forward/backward')
            event('fail8', actual, budget)
            retained = budget.ledger.snapshot()
            if not retained['failed_tickets']:
                raise AssertionError('The admitted failed attempt must stay charged')
            try:
                actual.save(root/'forbidden-partial.pt')
            except RuntimeError:
                pass
            else:
                raise AssertionError('An entered failed update cannot publish')
            if (root/'forbidden-partial.pt').exists():
                raise AssertionError('Poisoned state publication wrote a payload')
        finally:
            budget.ledger.close()

        budget = StoryWorkBudget.open(journal, contract=contract, scientific_contract=science,
                                     invocation_id='lesson-recovery', receipt=receipt)
        try:
            retained_before_initialization = budget.ledger.snapshot() == retained
            restored = _session(config, recipe, data, budget)
            restored.restore(boundary, work_receipt=receipt)
            event('restore9', restored, budget)
            if attempt_target_cap == 24:
                before = digest(_numerical(restored))
                prefix = budget.ledger.snapshot()
                with patch.object(restored, 'summed_loss', wraps=restored.summed_loss) as spy:
                    try:
                        restored.update()
                    except WorkBudgetExceeded:
                        pass
                    else:
                        raise AssertionError('An eight-target retry cannot fit seven remaining places')
                    calls = spy.call_count
                numerical_unchanged = before == digest(_numerical(restored))
                prefix_unchanged = prefix == budget.ledger.snapshot()
                if not retained_before_initialization or calls or not numerical_unchanged or not prefix_unchanged:
                    raise AssertionError('Lower-cap retry must refuse without state change or work refill')
                event('refuse8', restored, budget)
                state = budget.ledger.snapshot()
                return dict(scope='Actual native CPU retry refusal, not a successful two-update replay',
                    attempt_target_cap=24, retry_refused=True, timeline=timeline,
                    successful_targets=restored.tokens, reserved_targets=state['reserved']['training_valid_targets'],
                    later_failed_work_retained=retained_before_initialization,
                    next_refusal_forward_calls=calls, refusal_numerical_state_unchanged=numerical_unchanged,
                    refusal_work_prefix_unchanged=prefix_unchanged,
                    reserved=state['reserved'], completed=state['completed'],
                    known_partial=state['known_partial'], uncertain_upper=state['uncertain_upper'],
                    failed_tickets=state['failed_tickets'], actual_successful_updates=restored.step,
                    recipe_horizon=recipe.total_updates)
            retry = _record(restored.update())
            event('retry8', restored, budget)
            final_sha = digest(_numerical(restored))
            if not retained_before_initialization or retry != reference_records[1] or final_sha != reference_sha:
                raise AssertionError('Fresh native recovery changed the original trajectory or refunded work')
            before = digest(_numerical(restored))
            prefix_before_refusal = budget.ledger.snapshot()
            with patch.object(restored, 'summed_loss', wraps=restored.summed_loss) as spy:
                try:
                    restored.update()
                except WorkBudgetExceeded:
                    pass
                else:
                    raise AssertionError('The exhausted whole-update allowance must refuse')
                refused_forward_calls = spy.call_count
            unchanged = before == digest(_numerical(restored))
            prefix_unchanged = prefix_before_refusal == budget.ledger.snapshot()
            if not unchanged or not prefix_unchanged or refused_forward_calls:
                raise AssertionError('Cap refusal must leave numerical and admitted work state unchanged')
            final_work = budget.ledger.snapshot()
            return dict(scope='Actual tiny native CPU model-work clocks; no language-quality or physical-resource claim',
                attempt_target_cap=25, retry_refused=False, timeline=timeline, successful_targets=restored.tokens,
                reserved_targets=final_work['reserved']['training_valid_targets'],
                original_record_equal=retry == reference_records[1], exact_numerical_recovery=True,
                original_numerical_sha256=reference_sha, restored_numerical_sha256=final_sha,
                later_failed_work_retained=retained_before_initialization,
                failed_tickets=final_work['failed_tickets'], reserved=final_work['reserved'],
                completed=final_work['completed'], known_partial=final_work['known_partial'],
                uncertain_upper=final_work['uncertain_upper'], next_refusal_forward_calls=refused_forward_calls,
                refusal_numerical_state_unchanged=unchanged, refusal_work_prefix_unchanged=prefix_unchanged,
                recipe_horizon=recipe.total_updates, actual_successful_updates=restored.step)
        finally:
            budget.ledger.close()
