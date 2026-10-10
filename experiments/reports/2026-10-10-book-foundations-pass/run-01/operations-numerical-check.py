"""Recompute book teaching examples with existing CPU helpers; no new campaign."""
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import psutil
import torch
from dongxi_llms import instruction_data_lab as data
from dongxi_llms.sft_lab import token_loss_sum, LoRALinear
from dongxi_llms.reward_model_lab import bt_loss, reward_comparison
from dongxi_llms.dpo_lab import dpo_loss, optimal_policy
from dongxi_llms.distillation_lab import distillation_loss
from dongxi_llms.student_prefix_lab import A, B, C, teacher_distribution, topk_tail

torch.set_num_threads(1)
if psutil.virtual_memory().available < 25 * 1024**3:
    raise RuntimeError("Host reserve below 25 GiB")
def scalar(value):
    return value.detach().item() if isinstance(value, torch.Tensor) else float(value)

root = Path.cwd()
results = {"scope": "Finite CPU teaching calculations and existing reference reproduction; no GPU or new model campaign",
           "python": sys.executable, "python_version": platform.python_version(),
           "torch": torch.__version__, "device": "cpu"}

example = data.instruction_fixture()[0]
batch = data.collate([example])
results["ch08"] = {"ids": example.ids, "labels": batch["labels"][0].tolist(),
    "scored_producer_target_positions": [[i-1, i] for i, flag in enumerate(example.supervised) if flag],
    "same_red_id_ownership": [[i, example.owners[i], example.supervised[i]]
                             for i, token in enumerate(example.ids) if token == data.VOCAB["red"]],
    "input_positions": len(example.ids), "shifted_scored_targets": int((batch["labels"][:, 1:] != -100).sum()),
    "mixture_share": data.mixture_exposure([.5, .5], [10, 90])}

prob = torch.tensor([[[.6, .3, .1], [.2, .3, .5], [1/3, 1/3, 1/3]]], dtype=torch.float64)
labels = torch.tensor([[-100, 0, 2]])
logits = prob.log().requires_grad_()
loss_sum, count = token_loss_sum(logits, labels)
gradient = torch.autograd.grad(loss_sum/count, logits)[0]
expected = torch.zeros_like(prob)
expected[0, 0] = (prob[0, 0] - torch.tensor([1., 0., 0.])) / 2
expected[0, 1] = (prob[0, 1] - torch.tensor([0., 0., 1.])) / 2
torch.testing.assert_close(gradient, expected)
changed = logits.detach().clone().requires_grad_()
without_end = labels.clone(); without_end[0, -1] = -100
changed_sum, changed_count = token_loss_sum(changed, without_end)
changed_gradient = torch.autograd.grad(changed_sum/changed_count, changed)[0]
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(1414)
    layer = LoRALinear(torch.nn.Linear(6, 5, bias=False), rank=2, alpha=2)
    x = torch.randn(3, 6)
    torch.testing.assert_close(layer(x), layer.base(x), rtol=0, atol=0)
    layer(x).sum().backward()
    initialization = {"a_gradient_norm": scalar(layer.a.grad.norm()),
                      "b_gradient_norm": scalar(layer.b.grad.norm()),
                      "base_gradient_absent": layer.base.weight.grad is None}
    with torch.no_grad(): layer.b.add_(.1)
    torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, layer.merged_weight()))
    layer.zero_grad(set_to_none=True)
    with torch.no_grad(): layer.a.zero_(); layer.b.zero_()
    layer(x).sum().backward()
    assert scalar(layer.a.grad.norm()) == scalar(layer.b.grad.norm()) == 0.
results["ch09"] = {"probabilities": prob.tolist(), "labels": labels.tolist(),
    "loss_sum": scalar(loss_sum), "target_count": int(count), "mean_nll": scalar(loss_sum/count),
    "token_nll": [-math.log(.6), -math.log(.5)], "logit_gradient": gradient.tolist(),
    "without_end": {"mean_nll": scalar(changed_sum/changed_count), "target_count": int(changed_count),
                    "logit_gradient": changed_gradient.tolist()},
    "lora_1024_rank8": {"base": 1024**2, "adapter": 8*(1024+1024)},
    "lora_initialization": initialization, "merge_assertion": "passed"}
results["ch09"]["both_lora_factors_zero_gradient_assertion"] = "passed"
results["ch09"]["complete_answer_probability"] = .6*.5

a = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
b = torch.tensor([-.6], dtype=torch.float64, requires_grad=True)
loss = bt_loss(a, b)
ga, gb = torch.autograd.grad(loss, (a, b))
soft = bt_loss(a, b, torch.tensor([.7], dtype=torch.float64))
soft_ga, soft_gb = torch.autograd.grad(soft, (a, b))
rewards = reward_comparison()
results["ch10"] = {"scores": [.4, -.6], "margin": 1., "preference_probability": scalar(torch.sigmoid(a-b)),
    "hard_loss": scalar(loss), "hard_score_gradients": [scalar(ga), scalar(gb)],
    "soft_q_07": {"loss": scalar(soft), "score_gradients": [scalar(soft_ga), scalar(soft_gb)],
                  "optimal_margin": math.log(.7/.3)},
    "reward_reference": {name: {k: row[k] for k in ("weights", "train_nll", "validation_nll", "adversarial_first_win_probability")}
                         for name, row in rewards.items()}}

reference = torch.tensor([.6, .3, .1], dtype=torch.float64)
reward = torch.tensor([0., .5, 1.], dtype=torch.float64)
beta = .5
tilted = optimal_policy(reference, reward, beta)
weight = reference * (reward/beta).exp()
torch.testing.assert_close(tilted, weight/weight.sum())
stationarity = reward-beta*(tilted.log()-reference.log()+1)
torch.testing.assert_close(stationarity, stationarity.mean().expand_as(stationarity))
torch.testing.assert_close(tilted, optimal_policy(reference, reward+50, beta))
before = torch.tensor([.2, .1, .7], dtype=torch.float64)
after = torch.tensor([.1, .01, .89], dtype=torch.float64)
counterexample = []
for p in (before, after):
    c = p[0].log().reshape(1).requires_grad_()
    r = p[1].log().reshape(1).requires_grad_()
    loss, margin = dpo_loss(c, r, before[0].log().reshape(1), before[1].log().reshape(1), beta=beta)
    grad = torch.autograd.grad(loss, (c, r))
    counterexample.append({"probability": p.tolist(), "pair_odds": scalar(p[0]/p[1]),
        "margin": scalar(margin), "loss": scalar(loss), "selected_score_derivatives": [scalar(g) for g in grad]})
results["ch11"] = {"reference": reference.tolist(), "reward": reward.tolist(), "beta": beta,
    "unnormalized_weight": weight.tolist(), "partition": scalar(weight.sum()), "optimal_policy": tilted.tolist(),
    "stationarity": stationarity.tolist(), "reward_offset_assertion": "passed",
    "beta_1_policy": optimal_policy(reference, reward, 1.).tolist(), "counterexample": counterexample}

teacher = torch.tensor([[2., .5, -1.]], dtype=torch.float64, requires_grad=True)
temperature_rows = []
for temperature in (1., 2., 4.):
    for scale in (False, True):
        student = torch.zeros_like(teacher, requires_grad=True)
        loss = distillation_loss(student, teacher, temperature, scale)
        gradient = torch.autograd.grad(loss, student)[0]
        q = (teacher.detach()/temperature).softmax(-1)
        expected = (student.detach()/temperature).softmax(-1)-q
        expected *= temperature if scale else 1/temperature
        torch.testing.assert_close(gradient, expected)
        temperature_rows.append({"temperature": temperature, "scaled": scale, "teacher_probability": q[0].tolist(),
                                 "loss": scalar(loss), "student_logit_gradient": gradient[0].tolist()})
assert teacher.grad is None
student = torch.zeros_like(teacher, requires_grad=True)
hard_sum, hard_count = token_loss_sum(torch.cat([student[:, None, :], student[:, None, :]], 1),
                                      torch.tensor([[-100, 0]]))
hard_gradient = torch.autograd.grad(hard_sum/hard_count, student)[0]
concentrated = torch.tensor([[.9, .05, .05]], dtype=torch.float64).log().requires_grad_()
concentrated_loss = distillation_loss(concentrated, teacher)
concentrated_gradient = torch.autograd.grad(concentrated_loss, concentrated)[0]
torch.testing.assert_close(concentrated_gradient, concentrated.detach().softmax(-1)-teacher.detach().softmax(-1))
plain = teacher_distribution([A, B], [C])
hint = teacher_distribution([A, B], [C], hint=[B, A])
results["ch15"] = {"teacher_logits": teacher.detach().tolist(), "temperature_rows": temperature_rows,
    "hard_token0_gradient": hard_gradient[0].tolist(), "teacher_gradient_absent": teacher.grad is None,
    "concentrated_student_probability": concentrated.detach().softmax(-1)[0].tolist(),
    "concentrated_soft_gradient": concentrated_gradient[0].tolist(),
    "wrong_prefix_teacher": {"symbol_ids": [A, B], "past": [C], "ordinary": plain.tolist(), "hint": hint.tolist(),
                             "ordinary_highest_id": int(plain.argmax()), "hint_highest_id": int(hint.argmax())}}

tail_control = json.loads((root/"fixtures/student-prefix/items.json").read_text())["tail_control"]
tail_p = torch.tensor(tail_control["p"], dtype=torch.float64)
tail_q = torch.tensor(tail_control["q"], dtype=torch.float64)
tail = topk_tail(tail_p, tail_q, 2, "forward")
torch.testing.assert_close(tail["full"], tail["bucket"]+tail["lost_detail"], atol=1e-14, rtol=0)
results["ch15"]["tail_decomposition"] = {k: scalar(tail[k]) for k in ("full", "bucket", "lost_detail", "tail_student_mass", "tail_teacher_mass")}

paths = ["src/dongxi_llms/"+n for n in ("instruction_data_lab.py", "sft_lab.py", "reward_model_lab.py",
         "dpo_lab.py", "distillation_lab.py", "student_prefix_lab.py")]
results["source_sha256"] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
results["status"] = "passed"
print(json.dumps(results, indent=2, allow_nan=False))
