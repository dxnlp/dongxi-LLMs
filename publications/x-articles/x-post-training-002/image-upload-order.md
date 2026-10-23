# Image Upload Order

Cover (upload separately, 5:2): `assets/cover-post-training-02.png`. All figures are English and shared by both versions.

1. `assets/01-alpaca-recipe.gif` — Stanford's Alpaca (2023): an AI teacher wrote 52,000 instruction-response examples, and LLaMA 7B became an instruction-following model. Numbers as published by the Stanford team.
2. `assets/02-lima.gif` — LIMA (2023): 1,000 curated examples and no RL on top of LLaMA 65B; 43% of answers rated equal to or better than GPT-4's. Published result.
3. `assets/03-flan.gif` — FLAN (2021): a 137B model tuned on 60+ tasks phrased as instructions beat zero-shot GPT-3 175B on 20 of 25 unseen tasks. Published result; which tiles stay grey is illustrative.
4. `assets/04-recipe.gif` — The five steps of instruction tuning: collect, format, mask, train, evaluate.
5. `assets/05-three-forms.gif` — One example in three forms: a chat, one token sequence with template markers, and the loss mask. Markers simplified.
6. `assets/06-three-forms-summary.png` — The three forms side by side, with Qwen's real template markers.
7. `assets/07-what-gets-loss.gif` — In a multi-turn conversation, only the assistant responses and their end markers get a loss.
8. `assets/08-masked-not-untouched.gif` — The prompt gets no loss, but the response's gradient still flows back to it through attention. Schematic.
9. `assets/09-one-position-shift.gif` — Each position predicts the next token. Without the shift, each prediction is paired with its own token, and the model learns to copy. Schematic.
10. `assets/10-token-share.gif` — The loss is averaged over loss tokens, so a 500-token response weighs 250 times as much as a 2-token one. Illustrative counts.
