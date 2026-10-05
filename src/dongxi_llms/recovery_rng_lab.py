"""A private history replay is work, but it is not live training sampling.

Original CPU generator microscope, not a model/checkpoint validator. Draw counts
are actual scalar randint invocations, not token, FLOP or runtime measurements.
"""
from __future__ import annotations

import hashlib

import torch


def _integer(value, name, maximum, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Bounded exact integer required for {name}")


def _draw(generator, count, source_count):
    return [int(torch.randint(source_count, (), generator=generator))
            for _ in range(count)]


def _state_sha256(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def sampler_verification(*, seed=1818, completed_updates=3, accumulation=2,
                         source_count=5, future_updates=3):
    """Return observed histories for a safe and a deliberately broken check.

    All generators are private CPU objects. The broken variant consumes its live
    sampler while attempting verification and does not roll back that damage.
    Equal draw values can occur by chance, so compare actual RNG state as well.
    """
    _integer(seed, "seed", 2**63-1)
    _integer(completed_updates, "completed_updates", 1000)
    _integer(accumulation, "accumulation", 16, 1)
    _integer(source_count, "source_count", 4096, 2)
    _integer(future_updates, "future_updates", 64)
    history_count = completed_updates * accumulation
    future_count = future_updates * accumulation
    live = torch.Generator(device="cpu").manual_seed(seed)
    history = _draw(live, history_count, source_count)
    saved = live.get_state().clone()
    before = _state_sha256(live)

    verifier = torch.Generator(device="cpu").manual_seed(seed)
    private_replay = _draw(verifier, history_count, source_count)
    after_private_check = _state_sha256(live)
    private_state_matches = torch.equal(verifier.get_state(), saved)
    next_ids = _draw(live, future_count, source_count)
    expected = torch.Generator(device="cpu").set_state(saved)
    expected_ids = _draw(expected, future_count, source_count)

    broken = torch.Generator(device="cpu").set_state(saved)
    broken_replay = _draw(broken, history_count, source_count)
    after_broken_check = _state_sha256(broken)
    broken_next = _draw(broken, future_count, source_count)
    return {
        "seed": seed, "source_count": source_count,
        "completed_updates": completed_updates, "accumulation": accumulation,
        "future_updates": future_updates,
        "history": history, "private_replay": private_replay,
        "expected_next_ids": expected_ids, "safe_next_ids": next_ids,
        "broken_check_ids": broken_replay, "broken_next_ids": broken_next,
        "history_matches": history == private_replay,
        "private_state_matches": bool(private_state_matches),
        "live_unchanged_by_private_check": before == after_private_check,
        "safe_next_matches": next_ids == expected_ids,
        "broken_live_advanced": before != after_broken_check,
        "broken_check_matches_history": broken_replay == history,
        "state_before_check_sha256": before,
        "state_after_private_check_sha256": after_private_check,
        "state_after_broken_check_sha256": after_broken_check,
        "total_scalar_draws": 3*history_count+3*future_count,
        "draws": {
            "retained_training_history": history_count,
            "private_validation": history_count,
            "safe_future_training": future_count,
            "independent_comparison_future": future_count,
            "broken_check_consumed_live": history_count,
            "broken_future_training": future_count,
        },
        "scope": "Private CPU RNG microscope; not actual model quality, checkpoint validation or production recovery",
    }
