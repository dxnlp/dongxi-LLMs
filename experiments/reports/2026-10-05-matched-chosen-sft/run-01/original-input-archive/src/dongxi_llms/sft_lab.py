"""CPU SFT, gradient routing, LoRA and exact-token accumulation microscopes.

No checkpoint download. The reference starts randomly initialized; this is not
evidence about real base-to-assistant transfer. Run bounded examples as a module.
"""
import argparse
import copy
import json
import torch
from torch import nn
from torch.nn import functional as F
from .decoder_lab import DecoderConfig, TinyDecoder
from .instruction_data_lab import instruction_fixture, collate, SPECIAL, VOCAB


def token_loss_sum(logits, labels):
    aligned_logits, aligned_labels = logits[:, :-1], labels[:, 1:]
    count = int((aligned_labels != -100).sum())
    if not count:
        raise ValueError('Batch has no supervised next-token targets')
    loss = F.cross_entropy(aligned_logits.reshape(-1, logits.shape[-1]),
                           aligned_labels.reshape(-1), ignore_index=-100, reduction='sum')
    return loss, count


class LoRALinear(nn.Module):
    """Row-batch convention: y=x W^T + scale*(x A^T) B^T."""
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        if not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError('Invalid rank for teaching layer')
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.a, std=.02)
        self.scale = alpha/rank

    def forward(self, x):
        return self.base(x)+F.linear(F.linear(x, self.a), self.b)*self.scale

    def merged_weight(self):
        return self.base.weight+self.scale*self.b@self.a


def make_model(seed=1212):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDecoder(DecoderConfig(vocab=22, width=24, heads=3, kv_heads=3,
                                        head_dim=8, layers=1, hidden=48, max_length=32))


def add_lora(model, rank=4):
    model.requires_grad_(False)
    for block in model.blocks:
        block.attn.q = LoRALinear(block.attn.q, rank, rank)
        block.attn.v = LoRALinear(block.attn.v, rank, rank)
    return model


def gradient_routing(seed=1212):
    model = make_model(seed)
    batch = collate(instruction_fixture())
    loss, count = token_loss_sum(model(batch['input_ids']), batch['labels'])
    (loss/count).backward()
    return {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
            if parameter.grad is not None}


def accumulated_gradients(model, batches):
    """Backprop sums scaled by one common total, regardless of answer lengths."""
    model.zero_grad(set_to_none=True)
    count = sum(int((batch['labels'][:, 1:] != -100).sum()) for batch in batches)
    if not count:
        raise ValueError('Empty accumulation window')
    for batch in batches:
        loss, _ = token_loss_sum(model(batch['input_ids']), batch['labels'])
        (loss/count).backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


@torch.no_grad()
def complete(model, prefix, max_new_tokens=4):
    ids = torch.tensor([prefix], dtype=torch.long)
    generated = []
    for _ in range(max_new_tokens):
        token = int(model(ids)[0, -1].argmax())
        generated.append(token)
        ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
        if token == SPECIAL['end']:
            break
    return generated


def bounded_run(steps=100, seed=1212, lr=.015, mode='full'):
    if steps < 1 or steps > 400 or mode not in ('full', 'lora'):
        raise ValueError('Bounded demo uses 1..400 updates and full/lora')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_model(seed)
        if mode == 'lora':
            add_lora(model)
        batch = collate(instruction_fixture())
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
        with torch.no_grad():
            total, count = token_loss_sum(model(batch['input_ids']), batch['labels'])
            initial = float(total/count)
        history, norms = [], []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            total, count = token_loss_sum(model(batch['input_ids']), batch['labels'])
            loss = total/count
            loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError('Nonfinite optimization state')
            optimizer.step()
            history.append(float(loss.detach())); norms.append(float(norm))
        with torch.no_grad():
            total, count = token_loss_sum(model(batch['input_ids']), batch['labels'])
            final = float(total/count)
        outputs = [complete(model, example.ids[:-2]) for example in instruction_fixture()]
        expected = [[VOCAB[color], SPECIAL['end']] for color in ('red', 'blue', 'green', 'yellow')]
        # Verify restart data is sufficient at the immediate forward boundary.
        restored = make_model(seed)
        if mode == 'lora':
            add_lora(restored)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        replay_equal = bool(torch.equal(model(batch['input_ids']), restored(batch['input_ids'])))
        return model, dict(mode=mode, seed=seed, updates=steps, lr=lr,
                           initial_loss=initial, final_loss=final,
                           supervised_tokens_per_update=count,
                           trainable_parameters=sum(p.numel() for p in parameters),
                           total_parameters=sum(p.numel() for p in model.parameters()),
                           history=history, gradient_norms=norms,
                           outputs=outputs, expected=expected,
                           fixture_sequence_exact_match=sum(x == y for x,y in zip(outputs,expected))/len(expected),
                           state_forward_replay_equal=replay_equal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--mode', choices=['full','lora'], default='full')
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, result = bounded_run(args.steps, mode=args.mode)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
