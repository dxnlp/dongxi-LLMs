# Original critique/revision fixtures

`protocol.json` is the frozen preregistered recipe. `adversarial.json` contains
twelve deliberately authored state-machine tests; its references are evaluation
labels, not instructions to the critique or revision callbacks. No upstream
assets, human judgments, API responses or pretrained model outputs are authored
here. The separately reused 864 DXI05 records are actual tiny-model responses,
not self-critiques, and are identified by their original immutable file digest.

The symbolic interface makes emission costs inspectable: EOS=2, zero=6, one=7,
KEEP=20, FLIP=21, REPAIR=22, INVALID=23. An action and EOS cost two serialized
tokens. Those new control symbols are not part of the DXI05 model's output
support. Programmatic revision emits no neural logits or model likelihood.

The fixed fixtures include accepted right→wrong, accepted wrong→right, a
double-flip return to the original answer, an incorrect format repair, empty and
capped proposals, and critique/revision exceptions. They test the instrument;
they are never pooled into an accuracy claim about real models.
