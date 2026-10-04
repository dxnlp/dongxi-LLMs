"""A symbolic chat microscope with explicit ownership and segment boundaries.

These special IDs define a teaching format, not a Qwen/HF tokenizer template.
The assistant body AND end marker are supervised; headers are not.
"""
from dataclasses import dataclass
import torch
from .evaluation_lab import canonical_hash, normalize_answer

SPECIAL = {'pad': 0, 'bos': 1, 'system': 2, 'user': 3, 'assistant': 4, 'end': 5}
WORDS = ['red', 'blue', 'green', 'yellow', 'copy', 'reverse', 'one', 'two',
         'three', 'four', 'short', 'answer', 'please', 'color', 'is', 'kind']
VOCAB = {word: index+6 for index, word in enumerate(WORDS)}
ID_TO_WORD = {value: key for key, value in {**SPECIAL, **VOCAB}.items()}


@dataclass
class EncodedConversation:
    ids: list
    supervised: list
    owners: list


def encode_messages(messages):
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('Training conversation must end with an assistant response')
    ids, supervised, owners = [SPECIAL['bos']], [False], ['boundary']
    for message in messages:
        role = message['role']
        if role not in ('system', 'user', 'assistant'):
            raise ValueError('Unknown role')
        ids.append(SPECIAL[role]); supervised.append(False); owners.append(role+' header')
        for word in message['content'].split():
            if word not in VOCAB:
                raise ValueError(f'Teaching vocabulary has no word {word!r}')
            ids.append(VOCAB[word]); supervised.append(role == 'assistant'); owners.append(role)
        ids.append(SPECIAL['end']); supervised.append(role == 'assistant'); owners.append(role+' end')
    if not any(supervised):
        raise ValueError('No supervised targets')
    return EncodedConversation(ids, supervised, owners)


def collate(conversations):
    """Return full ids/labels; caller performs exactly one causal shift."""
    length = max(len(example.ids) for example in conversations)
    ids = torch.zeros((len(conversations), length), dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    for row, example in enumerate(conversations):
        ids[row, :len(example.ids)] = torch.tensor(example.ids)
        attention[row, :len(example.ids)] = True
        labels[row, :len(example.ids)] = torch.tensor([
            token if learn else -100 for token, learn in zip(example.ids, example.supervised)])
    return {'input_ids': ids, 'labels': labels, 'attention_mask': attention}


def packed_visibility(segment_ids):
    """Boolean [T,T], True=allowed: causal AND same independent conversation."""
    ids = torch.as_tensor(segment_ids)
    positions = torch.arange(len(ids))
    return (ids[:, None] == ids[None, :]) & (positions[None, :] <= positions[:, None])


def mixture_exposure(example_weights, response_lengths):
    if len(example_weights) != len(response_lengths) or min(example_weights) < 0 or sum(example_weights) <= 0:
        raise ValueError('Positive aligned mixture required')
    values = torch.tensor(example_weights, dtype=torch.float64)*torch.tensor(response_lengths)
    if values.sum() <= 0:
        raise ValueError('At least one supervised token required')
    return (values/values.sum()).tolist()


def group_split(records):
    """Stable exact-content groups. Similar semantic prompts still need audit."""
    groups = {}
    for record in records:
        key = canonical_hash([normalize_answer(record['prompt']), normalize_answer(record['answer'])])
        groups.setdefault(key, []).append(record['id'])
    return groups


def instruction_fixture():
    return [encode_messages([{'role': 'system', 'content': 'short answer'},
                             {'role': 'user', 'content': 'copy '+color},
                             {'role': 'assistant', 'content': color}])
            for color in ('red', 'blue', 'green', 'yellow')]
