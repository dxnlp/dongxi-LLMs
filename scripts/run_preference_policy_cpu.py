#!/usr/bin/env python3
"""Print reproducible Chapter 10–12 CPU evidence; never edits reports."""
import json
import platform
import time
import torch
from dongxi_llms.reward_model_lab import reward_comparison
from dongxi_llms.dpo_lab import categorical_dpo, tiny_sequence_dpo
from dongxi_llms.policy_gradient_lab import (exact_gradient, estimator_moments,
    group_estimator_moments, kl_estimators, procedural_policy)


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    logits = torch.tensor([.3, -.2, .1], dtype=torch.float64)
    reward = torch.tensor([0., 1., 3.], dtype=torch.float64)
    reference = exact_gradient(logits, reward)
    gradients = {"exact": reference.tolist()}
    for name, result in (("unbaselined", estimator_moments(logits, reward)),
                         ("constant_1.3", estimator_moments(logits, reward, 1.3)),
                         ("rloo_G3", group_estimator_moments(logits, reward, 3)),
                         ("inclusive_mean_G3", group_estimator_moments(logits, reward, 3, False))):
        gradients[name] = {"mean": result["mean"].tolist(), "variance_trace": result["variance_trace"]}
    kl = kl_estimators(torch.tensor([.2,.3,.5], dtype=torch.float64),
                       torch.tensor([.7,.2,.1], dtype=torch.float64))
    result = {"schema": 1, "mode": "bounded CPU mechanisms", "environment": {
        "python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "platform": platform.platform()}, "reward": reward_comparison(),
        "categorical_dpo": {mode: categorical_dpo(mode) for mode in ("dpo", "flipped", "sft")},
        "sequence_training": {mode: tiny_sequence_dpo(mode=mode) for mode in ("dpo", "sft")},
        "gradients": gradients,
        "kl": {key: {**item, "samples": item["samples"].tolist()} for key, item in kl.items()},
        "policies": [procedural_policy(mode, seed) for mode in ("reinforce", "baseline", "rloo", "ppo")
                     for seed in (1921, 1922, 1923)]}
    result["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
