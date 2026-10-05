# Keep all RLVR cap identity reads bounded

Declared before follow-up measurement on2026-10-05. Run04 and its38-test source
identity remain historical. The initial cap parse and direct rechecks are already
bounded/nonblocking/no-follow. Generic identity collection and closure hashing
would nevertheless reopen that same input through a blocking generic file reader.

Exclude the cap from generic input-file reads. Attach the actual bounded cap SHA
to the input identity with an explicit bounded-reader role; use that same reader
when rechecking source/input identity at closure. Recompute the captured identity
receipt after this additive field, preserving all existing input/source/parent
fields. A no-peer FIFO or link substituted after parsing must refuse before a
blocking read. Keep the numerical recipe, all23 caps and the journal unchanged.

Add a regression covering the actual identity-closure function with a replaced
no-peer FIFO and a spy forbidding generic cap reads. Collect existing38 checks
and this check into a separate run05; retain all earlier raw reports unchanged.
This only addresses the cap input role, not arbitrary hostile model files or
a physical process/filesystem sandbox. No shared helper edit or model acquisition
is included.
