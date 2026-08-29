# Machine and Environment Manifests

Manifests identify the hardware, operating system, source revision, and software
environment used by an experiment. A manifest may describe the machine globally,
but each experiment report must still link the exact manifest it used.

Generated manifests must not contain credentials, access tokens, private cache
paths, or unrelated process command lines.
