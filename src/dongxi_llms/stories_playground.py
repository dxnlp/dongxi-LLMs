"""Bounded single-request inference for one explicitly configured local checkpoint."""
import json
import math
from pathlib import Path
import threading
import time


def validate_request(body):
    if not isinstance(body, dict):
        raise ValueError('Expected a JSON object')
    prompt = body.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 8192:
        raise ValueError('Enter a nonempty story opening (at most 8 KiB).')
    count, seed, temperature = body.get('max_new_tokens', 128), body.get('seed', 909), body.get('temperature', .8)
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError('Generation length must be 1–256 tokens.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer between 0 and 4294967295.')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 1.5:
        raise ValueError('Temperature must be between 0 and 1.5; zero means greedy.')
    return prompt, count, seed, temperature


class Playground:
    def __init__(self, checkpoint, tokenizer_path, run_name):
        import torch
        from tokenizers import Tokenizer
        from dongxi_llms.decoder_lab import DecoderConfig
        from dongxi_llms.stories_training import StoriesDecoder, available_gib
        self.torch, self.memory = torch, available_gib
        self.lock = threading.Lock()
        self.run_name, self.checkpoint = run_name, Path(checkpoint).name
        if available_gib() < 25:
            raise RuntimeError('Host memory reserve is too low to load the model.')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Playground requires the verified Spark CUDA/BF16 environment.')
        # mmap avoids eagerly materializing unused optimizer tensors on CPU.
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        self.step = state['step']
        cfg = DecoderConfig(**state['contract']['model'])
        with torch.random.fork_rng(devices=[]):
            self.model = StoriesDecoder(cfg)
        self.model.load_state_dict(state['model'], strict=True)
        del state  # Never move optimizer/checkpoint RNG state to the GPU.
        self.model = self.model.to('cuda').eval()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.context = cfg.max_length
        if self.tokenizer.get_vocab_size() != cfg.vocab:
            raise ValueError('Tokenizer/model vocabulary mismatch')

    def info(self):
        return dict(enabled=True, loaded=True, run=self.run_name, checkpoint=self.checkpoint,
                    update=self.step, context=self.context, max_new_tokens=256,
                    device='Spark CUDA / BF16 autocast', busy=self.lock.locked())

    def generate(self, body):
        prompt, count, seed, temperature = validate_request(body)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('The model is busy. Try again after the current generation finishes.')
        try:
            torch = self.torch
            from dongxi_llms.stories_data import EOS
            prefix = [EOS, *self.tokenizer.encode(prompt, add_special_tokens=False).ids]
            if len(prefix) >= self.context or len(prefix) > 768:
                raise ValueError('Story opening is too long. Use at most 767 prompt tokens.')
            if self.memory() < 25:
                raise RuntimeError('Generation refused: host memory reserve below 25 GiB.')
            start = time.monotonic()
            limit = min(count, self.context-len(prefix))
            generator = torch.Generator(device='cuda').manual_seed(seed)
            ids = torch.tensor([prefix], device='cuda')
            generated, reason = [], 'max_new_tokens' if limit == count else 'context_limit'
            with torch.inference_mode():
                for _ in range(limit):
                    if time.monotonic()-start > 60:
                        reason = 'time_limit'
                        break
                    if self.memory() < 25:
                        reason = 'memory_reserve'
                        break
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = self.model.lm_head(self.model.features(ids)[:, -1]).float()
                    if not torch.isfinite(logits).all():
                        raise RuntimeError('Nonfinite logits; generation stopped.')
                    token = int(logits.argmax(-1).item()) if temperature == 0 else int(
                        torch.multinomial((logits/temperature).softmax(-1), 1, generator=generator).item())
                    generated.append(token)
                    if token == EOS:
                        reason = 'eos'
                        break
                    ids = torch.cat((ids, torch.tensor([[token]], device='cuda')), 1)
            return dict(prompt=prompt, continuation=self.tokenizer.decode(generated, skip_special_tokens=True),
                        generated_tokens=len(generated), prompt_tokens=len(prefix)-1, seed=seed,
                        temperature=temperature, stop_reason=reason, seconds=time.monotonic()-start,
                        checkpoint=self.checkpoint, run=self.run_name, update=self.step)
        finally:
            self.lock.release()
