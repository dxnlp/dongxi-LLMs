# Worked solutions — Instruction Data as an Interface

These answers accompany [Chapter 8](../chapters/08-instruction-data-as-an-interface.md). The symbolic fixture makes masks visible; the real-model runner uses its separately audited template.

## 1. Serialize the roles

The teaching conversation is BOS, SYSTEM header, “short answer”, END, USER header, “copy red”, END, ASSISTANT header, “red”, END. Each word and marker has an explicit teaching ID. The same English text rendered by a Qwen tokenizer will generally have different IDs and token boundaries.

At training time include the completed answer. At generation time include context and the assistant-start prefix. Inspect raw IDs and decoded text; a Python message dictionary is not itself the model input.

## 2. First assistant target

The ASSISTANT header's logit predicts the first assistant body token. In an unshifted ownership array, mark that body token as supervised. Then align logits at positions $0,\ldots,T-2$ with labels at $1,\ldots,T-1$.

Do not mark the preceding header merely because its logit supplies the prediction. Ownership concerns the target being taught. The notebook prints the exact producing-position/target pairs; this catches an off-by-one error that a target-count check can miss.

## 3. Zero prompt loss and gradients

A prompt prediction can have zero direct loss while its hidden state affects later answer predictions through attention. The answer's derivative flows back through those context dependencies. Prompt embeddings can therefore receive a gradient.

Attention masking, loss masking and parameter freezing are distinct. To forbid a context source, change visibility. To omit a direct prediction, change its label. To prevent a parameter update, freeze the parameter. Conflating these controls can quietly change the objective or information available.

## 4. END-as-pad failure

Imagine IDs $[7,5,5,5]$ where the second position is a real END and the last two are padding, also represented by 5. Masking all IDs equal to 5 removes the real termination target. A length-derived validity mask $[1,1,0,0]$ retains it.

Build padding validity before combining it with assistant ownership. Confirm that all padded labels are ignored and that the final real assistant END remains supervised. Termination quality requires its own free-generation check too.

## 5. Overlength policy

One transparent bounded policy rejects any serialized example longer than the declared context. Record the number of rejected examples, their source/task distributions and the lost supervised-token fraction. This avoids silently removing task instructions or ending targets.

A production policy may instead truncate or segment, but must state which side is removed and what happens to incomplete answers. A length budget is a data-selection rule; it can disproportionately remove long reasoning tasks and change the mixture.

## 6. Isolated packing

Use

$$
M_{t,u}=\mathbf{1}\{u\le t\}\mathbf{1}\{s_u=s_t\}.
$$

For segment IDs $[0,0,1,1]$, query 3 can see positions 2 and 3, while query 2 cannot see 3 or either earlier segment position. Position IDs may reset within segments or continue, according to the chosen positional scheme. Compare with independent forwards to establish equivalence.

Also ignore boundary targets that would ask the last token of one segment to predict the first token of another. The teaching policy ignores BOS and headers; another policy needs its own explicit boundary mask.

## 7. Why END is insufficient

END supplies a token whose meaning is learned. An ordinary triangular mask still allows every later query to attend to earlier positions. No algebraic boundary appears merely because END occurs.

The visualization shows this difference directly: the ordinary lower-left triangle contains cross-conversation links; the segment mask removes them. A throughput recipe that allows those links has changed the conditioning distribution and should say so.

## 8. Token exposure

Equal example sampling with answer lengths 10 and 90 gives token shares

$$
\pi_{\mathrm{short}}=\frac{0.5\cdot10}{0.5\cdot10+0.5\cdot90}=0.1,
\qquad \pi_{\mathrm{long}}=0.9.
$$

Inverse-length weights produce equal expected token shares but sample short examples more often. Example averaging is another objective with different weights. The correct unit depends on the desired behavior and should be defended using the evaluation slices.

## 9. Group related examples

Group normalized identical instruction-answer pairs before splitting. For synthetic generation, keep records sharing the same value/source group together. If testing phrasing transfer, place related template families in separate groups too.

The original English fixture uses disjoint numeric values across train, development and test but shares task templates. Its test therefore measures transfer to unseen values under familiar task forms, not unseen instruction families. This limitation belongs in the card.

## 10. Data-card limitation

Write: “Four symbolic copy requests demonstrate role serialization, loss alignment and bounded optimization. They do not represent the distribution of general user requests and cannot support a general assistant-quality claim.”

State authorship, release license status, template, stopping marker, padding, rejection policy and grouping. Preserve the generator and hashes. A data card should make misuse of the evidence difficult, not advertise a dataset more broadly than its design supports.

## Notebook pathway

- [Roles and masks](../../notebooks/day-11/01_roles_templates_and_masks.ipynb).
- [Padding and packing boundaries](../../notebooks/day-11/02_padding_packing_boundaries.ipynb).
- [Mixtures and cards](../../notebooks/day-11/03_mixtures_and_data_cards.ipynb).

The corresponding plots display target alignment, actual visibility matrices and supervised-token shares.
