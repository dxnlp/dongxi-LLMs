# Image Upload Order

Cover (upload separately, 5:2): `assets/cover-token-birth.png`. All figures are English and shared by both versions.

1. `assets/01-overview.gif` — The whole loop: the prompt is tokenized once; each forward pass gives logits, one token is picked and appended, and its text joins the reply. Nine passes write the reply; the end token stops it.
2. `assets/02-chat-template.gif` — The question is wrapped in the chat template (amber). The answer starts on the empty line after assistant.
3. `assets/03-tokenize.gif` — The text is cut into vocabulary entries, and each becomes its token ID. Real Qwen3 tokens and IDs.
4. `assets/04-prefill.gif` — Each ID selects a row of the embedding table. All 13 positions pass through the layers together and leave their keys and values in the KV cache. Only the last position reaches the LM head: 151,936 logits. Three layers stand for 28.
5. `assets/05-pick-token.gif` — Softmax, temperature, top-p 0.9; greedy decoding always takes Part, sampling gives Part, Open, Part, No, Part. Illustrative numbers.
6. `assets/06-decode-loop.gif` — Each pass feeds only the newest token. It reads the cache in every layer (green arcs) and adds its own keys and values; the picked token drops back in as the next input.
7. `assets/07-detokenize.gif` — The IDs are looked up in reverse; the end token is dropped, spaces return, and the pieces join into the reply.
