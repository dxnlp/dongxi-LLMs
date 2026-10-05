"""Transparent evaluation accounting. Fixtures are illustrative, not model scores."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
import unicodedata


def normalize_answer(text):
    """Only NFC, casefold and whitespace; punctuation remains meaningful."""
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def exact_match(prediction, references, normalize=True):
    transform = normalize_answer if normalize else lambda x: x
    return any(transform(prediction) == transform(answer) for answer in references)


def pass_at_k(n, c, k):
    """Unbiased finite-pool estimator for iid samples; no selected-best oracle."""
    if not all(isinstance(x, int) for x in (n, c, k)) or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Need integer 0 <= c <= n and 1 <= k <= n')
    if n-c < k:
        return 1.0
    return 1.0-math.prod((n-c-i)/(n-i) for i in range(k))


def wilson_interval(successes, total, z=1.959963984540054):
    """Binomial score interval; independent items are an explicit assumption."""
    if total <= 0 or not 0 <= successes <= total or z <= 0:
        raise ValueError('Invalid binomial counts or z')
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return max(0., center-half), min(1., center+half)


def paired_bootstrap(a, b, draws=2000, seed=1010):
    """Resample item pairs; returns mean B-A and a percentile interval."""
    if len(a) != len(b) or not a or draws < 1:
        raise ValueError('Nonempty aligned item scores required')
    differences = [y-x for x, y in zip(a, b)]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(differences) for _ in differences)/len(differences)
                     for _ in range(draws))
    return sum(differences)/len(differences), (samples[int(.025*(draws-1))], samples[int(.975*(draws-1))])


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def contamination_groups(records):
    """Exact normalized prompt+reference collisions across splits."""
    groups = defaultdict(list)
    for item in records:
        key = canonical_hash([normalize_answer(item['prompt']), normalize_answer(item['answer'])])
        groups[key].append((item['id'], item['split']))
    return [group for group in groups.values() if len({split for _, split in group}) > 1]


def slice_scores(records):
    groups = defaultdict(list)
    for item in records:
        groups[item['slice']].append(int(item['correct']))
    return {name: {'n': len(scores), 'accuracy': sum(scores)/len(scores),
                   'interval': wilson_interval(sum(scores), len(scores))}
            for name, scores in sorted(groups.items())}


@dataclass(frozen=True)
class EvaluationContract:
    name: str
    split_hash: str
    model_revision: str
    template_hash: str
    metric: str = 'NFC-casefold-whitespace exact match'
    max_new_tokens: int = 64
    temperature: float = 0.
    seed: int = 1010

    @property
    def identity(self):
        return canonical_hash(self.__dict__)


def evaluation_fixture():
    """Purpose-built scores illustrating a slice regression; no pretrained run."""
    rows = []
    for index in range(40):
        difficult = index >= 30
        rows.append(dict(id=f'item-{index:02}', slice='rare-format' if difficult else 'common',
                         a=index % (2 if difficult else 3) != 0,
                         b=(index % 4 == 0) if difficult else (index % 8 != 0)))
    return rows
