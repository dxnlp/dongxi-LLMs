# Day 7 architecture-defense notebook — smoke specification

Add a core synthesis notebook alongside the existing optional recurrence lesson.
Use the modern tiny decoder (D=16, L=2, Hq=4, Hkv=2, d=4, F=32, V=16,
RMSNorm/SwiGLU/RoPE, QK norm and tied vocabulary head), seed 505, CPU float64.
Use the existing two-sequence six-position teaching inputs, correctly shifted.

The notebook must contain predictions, safe learner attempt cells, adjacent
runnable solutions, architecture maps, data-backed figures, and evidence limits.
It must trace actual module boundaries, account for unique parameters, expose
loss gradients, and diagnose finite-but-broken causal/cache variants. Include
a comparison-design worksheet, not an unrequested training campaign.

Acceptance: expected shapes/counts; all parameters connected with finite
gradients in this fixture; clean model passes finite/causal/cache checks;
wrong rotation offsets fail cache equivalence while remaining causal;
full-time centering fails future invariance and cache equivalence. Unit tests
check trace cleanup and side-effect boundaries. Use atol 1e-10, rtol 1e-8.

Execute all twelve Chapter 5 notebooks in fresh kernels without overwriting
sources; export saved previews; run all tests and inspect the new figures.
Preserve Day 6 prose edits and the existing recurrence notebook. Do not rename
its path or invalidate old links. No large model, benchmark, animation render,
trained recurrence comparison, or learner mastery claim.
