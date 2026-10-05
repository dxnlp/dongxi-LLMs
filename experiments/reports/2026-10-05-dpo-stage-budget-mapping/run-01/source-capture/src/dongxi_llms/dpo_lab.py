"""Chapter 11 DPO contracts. Predictive logits are already aligned to labels."""
import torch
from torch.nn import functional as F


def sequence_logps(logits, labels, completion_mask, average=False):
    """[B,T,V] -> [B]; prompt/pad labels masked, EOS retained when valid.

    This function does NOT shift. Pass model(input_ids[:,:-1]), labels=input_ids
    [:,1:], and the corresponding shifted completion mask exactly once.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or completion_mask.shape != labels.shape:
        raise ValueError("Expected logits [B,T,V], labels and mask [B,T]")
    if completion_mask.dtype != torch.bool:
        raise ValueError("Completion mask must be boolean")
    if (completion_mask.sum(1) == 0).any():
        raise ValueError("Every sequence must contain a scored completion token")
    selected = labels[completion_mask]
    if ((selected < 0) | (selected >= logits.shape[-1])).any():
        raise ValueError("Scored labels are outside the vocabulary")
    safe = labels.masked_fill(~completion_mask, 0)
    token_logps = logits.log_softmax(-1).gather(-1, safe[..., None]).squeeze(-1)
    sums = token_logps.masked_fill(~completion_mask, 0).sum(-1)
    return sums / completion_mask.sum(-1) if average else sums


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
             beta=.2, preference=None):
    """Stable mean pairwise likelihood; references detached, beta positive."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if any(v.shape != policy_chosen.shape for v in values):
        raise ValueError("All log-probability vectors must have equal shape")
    margin = beta * (policy_chosen - policy_rejected -
                     reference_chosen.detach() + reference_rejected.detach())
    q = torch.ones_like(margin) if preference is None else preference
    if q.shape != margin.shape or not torch.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Invalid preference probabilities")
    return F.binary_cross_entropy_with_logits(margin, q), margin


def optimal_policy(reference, reward, beta):
    if beta <= 0 or reference.shape != reward.shape or (reference <= 0).any():
        raise ValueError("Need positive reference support and beta")
    if not torch.allclose(reference.sum(-1), torch.ones_like(reference.sum(-1))):
        raise ValueError("Reference must sum to one")
    return (reference.log() + reward / beta).softmax(-1)


def categorical_dpo(mode="dpo", steps=240, beta=.5):
    """Two contexts, three completions, all three pairs. Exact soft labels.

    dpo: known preference probabilities; flipped: reverse labels; sft: imitate
    the best completion. The latter sees demonstration data, not equivalent
    supervision. Equal optimizer updates do not imply equal information.
    """
    if mode not in ("dpo", "flipped", "sft"):
        raise ValueError(mode)
    reward = torch.tensor([[.7, 0., -.4], [-.3, .6, 0.]], dtype=torch.float64)
    reference = torch.tensor([[.5, .3, .2], [.2, .5, .3]], dtype=torch.float64)
    logits = reference.log().clone().requires_grad_()
    left, right = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])
    q = torch.sigmoid(reward[:, left] - reward[:, right])
    if mode == "flipped":
        q = 1 - q
    history = []
    for _ in range(steps):
        logp = logits.log_softmax(-1)
        if mode == "sft":
            loss = F.cross_entropy(logits, reward.argmax(-1))
        else:
            loss, _ = dpo_loss(logp[:, left], logp[:, right], reference.log()[:, left],
                               reference.log()[:, right], beta, q)
        (gradient,) = torch.autograd.grad(loss, logits)
        with torch.no_grad():
            logits -= 1. * gradient
        history.append(float(loss.detach()))
    p = logits.detach().softmax(-1)
    return {"probability": p.tolist(), "history": history,
            "expected_reward": float((p * reward).sum(-1).mean()),
            "reference_kl": float((p * (p.log() - reference.log())).sum(-1).mean()),
            "optimal_probability": optimal_policy(reference, reward, beta).tolist(),
            "beta": beta, "updates": steps}


def model_pair_loss(model, reference, chosen, rejected, beta=.2, preference=None):
    """Each branch is (ids [B,L], attention [B,L], completion [B,L]).

    Supports a Tensor-returning teaching model or HF output with .logits. A
    model requiring attention_mask receives it through the caller's wrapper.
    Completion masks refer to target TOKEN positions before the single shift.
    """
    def scores(network, branch):
        ids, attention, completion = branch
        output = network(ids[:, :-1], attention[:, :-1])
        logits = output.logits if hasattr(output, "logits") else output
        return sequence_logps(logits.float(), ids[:, 1:], completion[:, 1:] & attention[:, 1:].bool())
    chosen_logp, rejected_logp = scores(model, chosen), scores(model, rejected)
    with torch.no_grad():
        ref_chosen, ref_rejected = scores(reference, chosen), scores(reference, rejected)
    loss, margin = dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected, beta, preference)
    return loss, {"margin": margin.detach(), "chosen_logp": chosen_logp.detach(),
                  "rejected_logp": rejected_logp.detach()}


def tiny_sequence_dpo(steps=80, seed=1718, beta=.5, mode="dpo"):
    """Actual tiny decoder learns a one-byte answer style; new prompt byte held out.

    All prompts prefer A over B. Held-out P(A|new prompt) is an independently
    defined exact next-token metric, not a learned reward or a language claim.
    """
    from copy import deepcopy
    from dongxi_llms.decoder_lab import DecoderConfig, TinyDecoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyDecoder(DecoderConfig(vocab=12, width=16, heads=2, kv_heads=2,
                            head_dim=8, layers=1, hidden=32, max_length=8,
                            modern=True, qk_norm=False, tied=True))
    reference = deepcopy(model).eval().requires_grad_(False)
    reference_initial = {key: value.clone() for key, value in reference.state_dict().items()}
    model.eval()  # Disable stochastic layers; autograd remains enabled.
    def wrapper(network):
        return lambda ids, attention: network(ids)
    def branch(answer, prompts):
        ids = torch.tensor([[0, p, answer, 11] for p in prompts])
        return ids, torch.ones_like(ids), torch.tensor([[False, False, True, True] for _ in prompts])
    chosen, rejected = branch(9, range(1, 7)), branch(10, range(1, 7))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.01)
    history = []
    before = float(model(torch.tensor([[0, 8]]))[0, -1].softmax(-1)[9].detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if mode == "dpo":
            loss, observed = model_pair_loss(wrapper(model), wrapper(reference), chosen, rejected, beta)
            margin = float(observed["margin"].mean())
        elif mode == "sft":
            ids, attention, mask = chosen
            loss = -sequence_logps(model(ids[:, :-1]), ids[:, 1:], mask[:, 1:]).mean()
            margin = None
        else:
            raise ValueError(mode)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        history.append({"loss": float(loss.detach()), "margin": margin,
                        "heldout_A_probability": float(model(torch.tensor([[0, 8]]))[0, -1].softmax(-1)[9].detach())})
    ref_unchanged = all(torch.equal(value, reference_initial[key]) for key, value in reference.state_dict().items())
    return {"mode": mode, "updates": steps, "history": history,
            "heldout_A_before": before, "heldout_A_after": history[-1]["heldout_A_probability"],
            "reference_has_gradients": any(p.grad is not None for p in reference.parameters()),
            "reference_self_check": ref_unchanged}
