# Original color and style control fixtures

These ten prompts and pair labels were independently authored for the critic
lesson. No external dataset or generated teacher label is used. Train sources
are cup/key, calibration sources bag, final test sources box, and withheld
control sources mug; each has both red and blue labels. The fixed actor grammar
and printable-ASCII reward alphabet are declared before fitting. No word/UNK
alias collapses these prompts.

Balanced comparisons match style on each side. Confounded comparisons associate
correctness with fancy style and reverse only pair orientation; those two
observations are duplicates of one content contrast, not independent evidence.
The arms have equal source groups, pair counts and optimizer budgets, not equal
information. Unused process records do not enter preference fitting.

The independent quality rule checks requested color, at most one style marker
and delivered EOS. It is not supplied to either actor or critic. Reward scores
are actual fitted-model outputs, frozen/reloaded before policy updates, not
handwritten reward numbers. The finite response grammar is not open language
inference. The protocol and premeasurement specification define all budgets.

