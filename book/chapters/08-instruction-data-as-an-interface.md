# 8. Instruction Data as an Interface

A user writes “Reply with one color.” The model answers “Here is a short story about a red balloon.” Its language may be fluent and its next-token loss may be low, yet it has failed the requested interface. Instruction tuning makes such interfaces part of the training distribution. The examples teach what an answer looks like, where it begins, whose words are context, and when it should end.

An instruction dataset is therefore a behavioral specification expressed as sequences. Its quality depends on more than the answer text. A wrong role marker, an incorrectly shifted mask, or contamination from the evaluation suite can change the experiment without changing a single natural-language answer.

Day 11 uses [roles and serialization](../../notebooks/day-11/01_roles_templates_and_masks.ipynb), [padding and packing](../../notebooks/day-11/02_padding_packing_boundaries.ipynb), and [mixtures and provenance](../../notebooks/day-11/03_mixtures_and_data_cards.ipynb). Follow the [lab guide](../labs/08-instruction-data-as-an-interface.md); [worked solutions](../solutions/08-instruction-data-as-an-interface.md) explain each exercise.

## 8.1 Begin with the user's desired behavior

Consider three examples: copy one word, reverse two words, and return an object with a required key. They all concern instruction following, but they require different output structure. A dataset with only copy tasks cannot establish a general assistant. A dataset with only beautifully formatted answers might teach presentation while leaving factual mistakes untouched.

Write a compact capability table first. Name the input family, answer policy, permitted transformations and regression risk. For a small course dataset, copying and extraction are useful because correctness can be checked exactly. Include held-out values and phrasings when testing transfer. Reusing every value in every training prompt makes a clean-looking score a memorization check.

The model sees demonstrations, not the data designer's intent. If every response begins “Certainly!”, the model can learn that phrase. If long answers dominate the supervised tokens, it receives more corrective pressure for verbosity than the example count suggests. If refusal examples are mixed without defining scope, benign questions may receive refusals. These effects should be anticipated in the card and evaluation suite.

## 8.2 Messages become one token sequence

A conversation often has system, user and assistant messages. A system message supplies behavior instructions or application context. A user message supplies the request. An assistant message supplies a demonstrated response. Tool messages are a separate interface and are outside this chapter's executable fixture.

Roles are encoded through text and special tokens according to a chat template. A decoder does not receive Python dictionaries; preprocessing converts them to IDs. The model learns the significance of those markers through training. The [Transformers chat-template guide](https://huggingface.co/docs/transformers/chat_templating) documents this serialization boundary.

The symbolic notebook format is intentionally small:

> BOS → SYSTEM → short answer → END → USER → copy red → END → ASSISTANT → red → END.

These labels denote discrete teaching IDs. They are not literal Qwen tokenization or an interchangeable industry standard. A real checkpoint's template can use distinct start and end markers, newlines, reasoning delimiters and tool structures. Inspect the actual decoded result from its pinned tokenizer.

For training a complete assistant demonstration, use the full conversation without an extra unfinished generation prompt. For inference, append whatever assistant-start prefix the model expects. A template that already contains BOS or end tokens must not be followed by another automatic special-token insertion. If formatting text first and tokenizing separately, verify the equivalent direct-template path and control additional special tokens.

## 8.3 Ownership is explicit

The role header is not the same as its body. A useful supervision policy learns assistant body tokens and the assistant's termination marker, while treating system/user content and role headers as context. Other policies can also learn headers; the choice must be written and tested.

Our encoder produces three aligned lists: token IDs, a boolean supervised flag, and a human-readable owner. The supervised flag is attached to the token being predicted. This detail matters when the model's logit at position $t$ predicts token $t+1$.

Let $x_{b,t}$ be the full conversation IDs, $m_{b,t}$ the ownership mask, and $a_{b,t}$ the padding-validity mask. Define labels

$$
\ell_{b,t}=
\begin{cases}
x_{b,t}, & m_{b,t}=1\ \text{and}\ a_{b,t}=1,\\
-100, & \text{otherwise}.
\end{cases}
$$

The sentinel $-100$ is an implementation convention for ignored labels, not a token ID. For logits $Z\in\mathbb{R}^{B\times T\times V}$, train using $Z[:,:-1,:]$ against $\ell[:,1:]$. Shift exactly once. Some high-level model APIs shift labels internally; the transparent teaching module performs the shift itself. Combining both shifts predicts two positions ahead.

In the symbolic example, the final assistant body “red” and its END marker are supervised. The ASSISTANT header is ignored. The logit at the header predicts “red”; the logit at “red” predicts END. The target belongs to the assistant even though the producing position may be the header.

## 8.4 Assistant-only loss still teaches from the prompt

Ignoring a user token's direct loss does not remove it from the forward graph. Its embedding and hidden states can influence later assistant positions through causal attention. The answer loss can therefore update shared parameters and prompt embeddings through those paths.

There are three different controls. The loss mask decides which predictions are scored. The attention mask decides which positions can exchange information. The parameter's trainable flag decides whether an optimizer updates that parameter. They solve different problems. A frozen embedding can participate in useful computation without receiving an update. A zero-loss prompt position can receive a gradient through later outputs. An attention-masked position can be excluded as context even if a careless label still asks the model to predict it.

This distinction links the present chapter to attention's causal boundary. An answer token is permitted to learn from earlier user text, while the earlier user representation cannot use the future answer in that forward pass. Backward information from the answer's loss updates parameters; it does not grant the forward computation access to future tokens.

## 8.5 Termination is part of the demonstration

A model that generates the right answer and continues indefinitely has not learned the complete interface. Include the intended assistant termination token as a supervised target. A maximum-token limit is a runtime cap, not the desired learned stopping behavior.

Do not assume a tokenizer's generic EOS ID equals every model's end-of-assistant marker. Inspect both template output and generation stop settings. Some formats distinguish a message end from conversation end. Suppressing all special tokens while inspecting text can hide whether the correct marker appeared, so retain raw IDs in the audit.

Truncation can erase the termination target or cut a response after only its opening. A recipe must say whether overlength examples are rejected, truncated, split or summarized. Report how many supervised targets survive and how many answers lose their natural ending. Prompt truncation should also preserve the task-defining instruction; chopping off the user's decisive condition changes the example.

The fixture keeps all sequences within a small bound. The Spark runner rejects examples that exceed its maximum length, rather than quietly selecting a different target policy.

## 8.6 Padding is not an instruction

Batches contain different lengths. Padding creates a rectangular tensor $X\in\mathbb{N}^{B\times T}$. Padded labels must be ignored and padded context must be masked appropriately. The attention-validity mask is separate from the supervision mask because valid user context is visible even when its labels are ignored.

Our right-padded microscope has no future influence on earlier real positions under a causal decoder. Its padding loss is still explicitly ignored. A production model should receive the actual attention mask; left padding changes position handling and generation conventions and requires an independent equivalence check.

If a model reuses EOS as its pad value, masking every token whose ID equals EOS can remove genuine end targets. Build padding masks from lengths or tokenizer-provided attention metadata, then combine them with ownership. Values alone cannot distinguish “real EOS here” from “padding placed here.”

A useful audit shows IDs, owners, labels and masks in the same heatmap. The reader can follow one assistant target across the causal shift. Summarized target counts are insufficient: an off-by-one mask can preserve the count while teaching the wrong token.

## 8.7 Packing changes the information boundary

Packing several short conversations into one long sequence reduces padding. A naive concatenation with only a triangular causal mask lets a later conversation attend to all earlier ones. An EOS marker supplies a learned clue; it does not mathematically prevent information flow.

For independent conversations, assign a segment identifier $s_t$ to each position. An allowed-attention mask is

$$
M_{t,u}=\mathbf{1}\{u\le t\}\mathbf{1}\{s_u=s_t\}.
$$

This block-diagonal causal structure allows a conversation to see its own past and excludes preceding independent examples. Whether position IDs reset within segments is another explicit choice. Test equivalence with independent forwards at the chosen positional convention.

Packing also creates an accidental cross-boundary next-token target unless the first token of each new segment is ignored. In our teaching policy BOS and role headers are ignored, so the boundary loss is excluded, but that property must be checked rather than assumed. Some throughput-oriented recipes intentionally allow cross-example context. Such recipes describe a different conditional distribution and should be compared as a controlled change.

The notebook visualizes ordinary causal visibility beside segment-isolated visibility. Change one segment ID and observe precisely which information paths open. No production packing backend is claimed by this schematic alone.

## 8.8 Mixture weights need a unit

Suppose half the sampled examples belong to short-answer tasks and half to explanation tasks. Short answers average ten supervised tokens; explanations average ninety. Under a global token mean, explanation tokens contribute approximately 90% of the objective, although example sampling is balanced.

If task family $r$ has example probability $w_r$ and mean supervised length $\bar n_r$, its expected token share is

$$
\pi_r=\frac{w_r\bar n_r}{\sum_q w_q\bar n_q}.
$$

To target equal supervised-token shares, one possible sampling strategy chooses $w_r$ proportional to $1/\bar n_r$. This changes the example distribution and increases the frequency of short tasks. Another approach averages each example's loss before averaging examples; that changes gradient weighting. Neither should be introduced silently.

Report examples, input tokens, assistant tokens and source proportions separately. A high fraction of explanation tokens is not necessarily bad; it becomes a problem when it contradicts the intended interface. Use an evaluation slice for concise-output instructions to see whether the behavior transfers.

Data-mixture changes and learning-rate changes should be tested separately when diagnosing a regression. If both change, a resulting score cannot identify the cause.

## 8.9 Provenance survives preprocessing

A data card should include creator/source, license, intended use, excluded use, collection or generation procedure, revisions, languages, task families, split rules, filtering, deduplication, template identity, mask policy, truncation counts and known limitations.

Content hashes are useful but do not replace source information. Identical answers to different tasks may be legitimate; identical instruction-answer pairs across train and test are suspicious. Group exact normalized pairs first, then audit near duplicates and shared document origins. Synthetic examples derived from the same template belong in related groups when assessing transfer.

The original fixture is authored for this course, without copied upstream examples; its data card records authorship and the release's declared license policy. It contains four symbolic copy conversations and intentionally lacks broad instruction diversity. The optional Spark dataset generator creates original deterministic copy/reverse/extraction examples with separate value groups. Its card explicitly limits the resulting claim to these task families.

Do not claim that synthetic authoring eliminates contamination risks. A generator may repeat a benchmark template or leak test values into a training vocabulary. Save the generation code and seeds, record grouping decisions, and verify train/development/test identities before any optimizer step.

## 8.10 From audited data to SFT

A compact acceptance audit checks that every example contains permitted roles, a nonempty assistant response, at least one surviving supervised target, valid vocabulary IDs, a correct termination target, no padded labels, and an understood length policy. Print representative examples selected by rule, including longest, shortest and multi-turn cases.

The automatic audit can verify syntax and alignment; a human or task verifier still needs to check whether the answer obeys the instruction. A well-formed wrong answer is a powerful wrong lesson.

The next chapter assumes these contracts are fixed. It derives the objective and follows the answer gradient through the whole decoder, then compares full tuning with a constrained low-rank update. Its actual CPU experiment teaches mechanics; its separately specified Spark run tests a real pretrained base model.

## 8.11 Exercises

1. Draw the serialized sequence for a system-user-assistant conversation.
2. Identify which logit predicts the first assistant body token.
3. Explain why zero prompt loss does not imply zero prompt gradient.
4. Give a case where masking by pad token value removes a valid ending.
5. Specify a safe overlength-example policy and its measurable cost.
6. Construct an attention mask isolating two packed conversations.
7. Explain why EOS alone does not isolate packed context.
8. Calculate token exposure for equally sampled ten-token and ninety-token answers.
9. Design grouping rules for related synthetic examples.
10. Write a data-card limitation that prevents a four-copy-task fixture from becoming a claim about general assistance.

The [worked solutions](../solutions/08-instruction-data-as-an-interface.md) include executable checks and an interpretation of each failure.
