# Queries, Keys, Values, and Learned Retrieval

- Day: 04
- Date opened: 2026-09-03
- Status: introduced
- Book destination: Chapter 4, queries/keys/values and content-addressed retrieval
- Related evidence: planned Day 4 attention notebook and implementation
- Related production tasks: `CAND-ANIM-008`

## Questions that drove the discussion

- Why does attention create queries, keys, and values rather than compare and
  mix the incoming states directly?
- What do the names query, key, and value actually mean?

## Learner's initial model

The learner recognized this as the foundational Day 4 question and requested a
mechanistic explanation. No settled explanation was claimed before instruction.

## Refined mental model

At one self-attention layer, every position can play both sides of a learned
retrieval operation. Its query is used when that position receives information;
its key determines how other positions can match it; and its value is the
message it can send when selected.

For incoming state $x_i$ at receiving position $i$ and incoming state $x_j$ at
candidate source position $j$:

$$
q_i=x_iW_Q,\qquad k_j=x_jW_K,\qquad v_j=x_jW_V.
$$

The routing score $q_i k_j^\top$ answers how compatible receiver $i$ is with
source $j$ under the learned projections. After scaling, masking, and row-wise
softmax, the resulting weight $a_{ij}$ controls how much of message $v_j$ enters
the output:

$$
o_i=\sum_j a_{ij}v_j.
$$

The projections separate two jobs: deciding where to read and deciding what is
returned. They also allow directed compatibility: the relation from $i$ to $j$
need not equal the relation from $j$ to $i$. Using the same unprojected state for
all three roles would force matching and payload into one representation and
greatly restrict what the layer can learn.

The names come from information retrieval and key-value memory, but they must
not be interpreted too literally. A query is not necessarily a human-readable
question, a key is not a token ID or dictionary word, and a value is not the
token's complete meaning. All three are learned continuous views of the current
hidden state. They begin without those semantic roles and acquire useful routing
behavior through training.

## Concrete examples and derivations

For a causal self-attention sequence, position $i$ emits one query and compares
it with keys at allowed source positions $j\leq i$. It then retrieves a weighted
mixture of their values. The same position also emits its own key and value so
that later positions may read from it.

This distinguishes attention tensors from the vocabulary interface studied on
Day 3. $Q$, $K$, and $V$ contain one continuous vector per sequence position;
they are not vocabulary-wide logit vectors and their indices are not token IDs.

## Demonstrated understanding

No explanation-back has yet been recorded. The mechanism is currently
introduced rather than demonstrated.

## Evidence and limitations

The equations are the intended Day 4 derivation, not yet verified by a preserved
implementation in this course. The database analogy explains role separation
but should not be used to claim that a head contains explicit symbolic records
or that individual attention weights faithfully explain model reasoning.

## Open edges

- Explain why dot products serve as compatibility scores.
- Derive every tensor shape and the full score matrix.
- Explain why $Q$ and $K$ must have compatible dimensions while $V$ may use a
  different width.
- Verify the role separation with a small hand-designed retrieval example.

## Reuse opportunities

- Foundational Chapter 4 section and worked exercise.
- Opening scene of `CAND-ANIM-008`: one state splits into receiver/query,
  address/key, and payload/value roles while object identity remains stable.
- First prediction and implementation section in the Day 4 notebook.
