# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str):
        print(prompt)
        return torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.long)

    def generate(
        self,
        prompt: torch.Tensor,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        prompt_len = prompt.size(1) - (prompt == self.tokenizer.pad_id).view(-1).sum().item()
        input_text_mask = prompt != self.tokenizer.pad_id
        start_pos = prompt_len
        prev_pos = 0
        for cur_pos in range(start_pos, max_gen_len):
            logits = self.model.forward(prompt[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], prompt[:, cur_pos], next_token
            )
            prompt[:, cur_pos] = next_token
            prev_pos = cur_pos
        # Decoding
        decoded = []
        t = prompt[0, :start_pos + max_gen_len].tolist()
        # cut to eos tok if any
        try:
            t = t[:t.index(self.tokenizer.eos_id)]
        except ValueError:
            pass
        # Don't know what is -1 and how it's generated
        t = [s for s in t if s != -1]
        try:
            decoded.append(self.tokenizer.decode(t))
            return decoded
        except Exception as e:
            print(f"Tokenization error: {e}")
            print("Trying per symbol tokenization.")
        # print(f"Tokens before decoding: {t}")
        for st in t:
            try:
                rt = self.tokenizer.decode(st)
                decoded.append(rt)
            except IndexError:
                print(f"Tokenizer error on token {st}")
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

