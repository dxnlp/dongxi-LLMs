"""Executable prefix-identity fixture; groups are illustrative, not tokenizer output."""
import hashlib

REQUESTS = {
    'A': ('project', 'budget=100', 'schedule', 'delivery', 'review-risks'),
    'B': ('project', 'budget=100', 'schedule', 'delivery', 'request-budget-half'),
    'C': ('project', 'budget=50', 'schedule', 'delivery', 'review-new-budget'),
}


def shared_prefix(a, b):
    return next((i for i, (x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)))


def state_fingerprint(prefix):
    """Schematic visual identity only; never label these values activations."""
    return [b/255 for b in hashlib.sha256('|'.join(prefix).encode()).digest()[:8]]


def validate():
    a,b,c = (REQUESTS[k] for k in 'ABC')
    assert shared_prefix(a,b)==4 and shared_prefix(a,c)==shared_prefix(b,c)==1
    assert a[:4]==b[:4] and a[1]!=c[1] and a[2:4]==c[2:4]
    assert all(state_fingerprint(a[:i])==state_fingerprint(b[:i]) for i in range(1,5))
    assert state_fingerprint(a[:1])==state_fingerprint(c[:1])
    assert all(state_fingerprint(a[:i])!=state_fingerprint(c[:i]) for i in range(2,6))
    assert len({state_fingerprint(r)[0] for r in (a,b,c)})==3
    return dict(B_shared_groups=4,C_shared_groups=1,A_computed_groups=5,B_computed_groups=1,C_computed_groups=4,
                evidence='Illustrative grouped-prefix identity; no tokenizer, model or timing measurement')


if __name__=='__main__':print(validate())
