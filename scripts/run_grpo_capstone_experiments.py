#!/usr/bin/env python3
"""Regenerate bounded Chapters13–15 evidence; CPU only, no servers/downloads."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import torch
from dongxi_llms.grpo_lab import decoder_rlvr
from dongxi_llms.optimization_diagnostics_lab import hacking_experiment
from dongxi_llms.distillation_lab import distill_distribution, inference_comparison


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    grpo = [decoder_rlvr(group_size=group, updates=12) for group in (4, 8)]
    hacking = [hacking_experiment(repair=repair) for repair in (False, True)]
    distillation = distill_distribution()
    sampling = inference_comparison()
    sources = ["src/dongxi_llms/grpo_lab.py", "src/dongxi_llms/optimization_diagnostics_lab.py",
               "src/dongxi_llms/distillation_lab.py", "src/dongxi_llms/decoder_lab.py",
               "scripts/run_grpo_capstone_experiments.py",
               "experiments/specs/2026-10-04-grpo-diagnostics-distillation.md"]
    report = {"mode": "cpu-mechanism", "command": f"{sys.executable} scripts/run_grpo_capstone_experiments.py",
              "python": platform.python_version(), "torch": torch.__version__,
              "machine": platform.machine(), "num_threads": torch.get_num_threads(),
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
              "seconds": time.monotonic()-started, "grpo": grpo, "reward_hacking": hacking,
              "distillation": distillation, "inference_simulation": sampling,
              "evidence_limits": "Bounded CPU mechanisms only; no Qwen, CUDA or natural-language capability result."}
    # JSON disallows nonfinite numbers; failed evidence cannot become valid JSON.
    text = json.dumps(report, indent=2, allow_nan=False)
    target = ROOT/"experiments/reports/2026-10-04-grpo-diagnostics-distillation.json"
    target.write_text(text+"\n")
    lines = ["# Chapters 13–15 CPU mechanism results", "", "Measured report; source hashes and every row are in the [JSON](2026-10-04-grpo-diagnostics-distillation.json).", "",
             f"Interpreter: `{sys.executable}`; Python {report['python']}; Torch {report['torch']}; {report['machine']}; one CPU thread.",
             f"Command: `{report['command']}`. Elapsed: {report['seconds']:.3f} seconds.", "",
             "## Autoregressive decoder GRPO", "", "One 16-wide layer, 12-token vocabulary; 60 SFT warm-start updates; 12 fresh RL updates per variant. Original 12 training/four held-out arithmetic prompts; frozen split, not natural-language reasoning.", "",
             "|Group|Valid response tokens|Initial train greedy|Final train greedy|Initial held-out greedy|Final held-out greedy|", "|---:|---:|---:|---:|---:|---:|"]
    for row in grpo:
        count = sum(step["valid_response_tokens"] for step in row["history"])
        lines.append(f"|{row['group_size']}|{count}|{row['initial']['train']['accuracy']:.3f}|{row['final']['train']['accuracy']:.3f}|{row['initial']['heldout']['accuracy']:.3f}|{row['final']['heldout']['accuracy']:.3f}|")
    lines += ["", "The comparison holds update count fixed, so G8 consumes more samples. It is not an equal-token algorithm ranking. Every held-out response is retained. Zero-variance groups and negative results are evidence, not discarded rows.", "",
              "## Reward misspecification", ""]
    for row in hacking:
        first, last = row["history"][0], row["history"][-1]
        lines.append(f"- {'Strict-reward restart' if row['repair'] else 'Broken proxy'}: expected proxy {first['proxy_reward']:.6f} → {last['proxy_reward']:.6f}; strict accuracy {first['strict_accuracy']:.6f} → {last['strict_accuracy']:.6f}.")
    lines += ["", "Both policies start from zero logits. This is actual exact-expectation SGD over three actions; repairing a previously hacked language model is not measured.", "",
              "## Distribution distillation", "",
              f"80 SGD updates at T=2: scaled forward KL {distillation['history'][0]['scaled_loss']:.9f} → {distillation['history'][-1]['scaled_loss']:.9f}. Teacher/student probabilities are retained. This establishes three-logit distribution fitting, not transferred reasoning.", "",
              "## Selection simulation", "", "Original categorical Monte Carlo, 1200 trials, seed 2628. Candidate cost is an illustrative 8 tokens each. Oracle availability assumes independent correctness; the imperfect ranker favors a wrong class.", "",
              "|Candidates|Majority accuracy|Proxy best-of-N accuracy|Oracle availability|Illustrative tokens|", "|---:|---:|---:|---:|---:|"]
    for row in sampling["rows"]:
        lines.append(f"|{row['n']}|{row['majority_accuracy']:.4f}|{row['proxy_best_n_accuracy']:.4f}|{row['oracle_any_correct']:.4f}|{row['candidate_token_budget']}|")
    lines += ["", "## Boundary", "", "No model downloads, Qwen execution, CUDA measurements, long training run, inference service, animation render or external publishing. Invariant/gradient tests and notebook execution have separate verification logs. Learner mastery is not inferred from this generated material."]
    target.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"report": str(target), "seconds": report["seconds"],
                      "grpo_variants": len(grpo)}, indent=2))


if __name__ == "__main__":
    main()
