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

On 2026-09-06 the learner initially identified V with attention scores. After
separating scores, weights, and values, the learner correctly explained that a
strong query-key match does not guarantee a useful result because the value
supplies the content. This demonstrates the routing/content distinction; full
tensor-shape understanding has not yet been assessed.

The learner next described the branches as three FFNs needed to store more
information. Refine this to three learned linear/affine projections, commonly
implemented by linear layers (sometimes fused), distinct from the decoder's
separate nonlinear feed-forward sublayer. They provide different learned views
of the same incoming state: receiver matching, source matching, and message
content. They do not introduce new context or act as three independent stores.
Three describes the standard query/key/value factorization, not a theorem that
attention requires exactly three independently parameterized networks. This
correction is introduced; learner restatement remains pending.

## Requested workflow and animation refinement — 2026-09-06

### Tensor-shape review after parameter-update discussion

Use four illustrative token positions, incoming width D=8, query/key width
d_k=3, and value width d_v=2. X is [4,8]; W_Q/W_K are [8,3] and W_V is [8,2].
The projected Q/K are [4,3], V is [4,2], scores and A are [4,4], and O is
[4,2]. A's rows are receiving queries and its columns are source positions.
The [4,4] matrix contains no vocabulary-candidate axis. Every source weight
multiplies the source's entire value vector, so one head uses the same routing
distribution across all its output features.

Longer supported inputs add position rows to activations and expand the
receiver/source score matrix without resizing the learned projection matrices.
This is a statement about the attention operation; positional handling, context
limits, and compute/memory still constrain a complete model. The review is
introduced; no new learner shape explanation is claimed. Its workflow is
already covered by `ANIM-ATTN-001`.

```text
                         Incoming states X
                     /          |          \\
                 X W_Q        X W_K        X W_V
                   Q            K            V
                    \          /             |
                       Q K^T                 |
                          |                  |
                   divide by sqrt(d_k)       |
                          |                  |
                   apply causal mask         |
                          |                  |
                   softmax over sources      |
                          |                  |
                    weights A -------------- V
                                  |
                                O = A V
```

Every position creates all three projected vectors. Each query row compares
against allowed source keys, including its own position, then mixes the matching
values. V is neither the compatibility scores nor their normalized weights.
The output O is a continuous feature matrix, not vocabulary logits or token IDs.

Extend the existing `CAND-ANIM-008` / `ANIM-ATTN-001` rather than duplicating it.
Keep the V branch visibly separate while Q and K form scores; transform those
scores into weights, then join the two branches for the value mixture. Hold
Q/K fixed while changing V to show unchanged weights and a changed output.
Conversely, hold V fixed while changing routing. These are storyboard requests;
new displayed numerical examples must be verified before Mac rendering.

## Evidence and limitations

The forward implementation and gradient/cache reports dated 2026-09-05 verify
the chapter's small examples. The database analogy explains role separation
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
