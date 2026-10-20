# Image Upload Order

Cover (upload separately, 5:2): `assets/cover-post-training.png`.

1. `assets/01-same-prompt-two-models.gif` — The same request, two models. The base model's likeliest next word starts another assignment; after post-training it starts an answer. Probabilities and replies are illustrative, modeled on OpenAI's published GPT-3 vs InstructGPT example, not measured.
2. `assets/02-two-stages.gif` — One model, two stages. Pretraining pours in a huge stream of raw text; post-training feeds the same model three small, curated streams: examples, comparisons and checked answers. Volumes are not to scale.
3. `assets/03-pretraining-vs-post-training.png` — Pretraining and post-training side by side: goal, data, signal, cost and how much the model changes.
4. `assets/03-instruction-tuning.gif` — Instruction tuning. The instruction is context only; the model is graded on each word of the response. Words are revealed one at a time for readability; training grades them all in one pass.
5. `assets/04-preference-alignment.gif` — Preference alignment. The preferred answer becomes more likely and the other less likely, while a leash keeps the model near the original. Simplified: the judge route (RLHF) and DPO share one picture.
6. `assets/05-rlvr.gif` — RLVR. The model tries several answers; a checker marks each right or wrong, and the right answer becomes more likely round by round. This checker reads only the final answer.
