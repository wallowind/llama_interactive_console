# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from functools import lru_cache
from typing import Optional, Tuple

"""
NOTES
    Article: Llama 2: Open Foundation and Fine-Tuned Chat Models
        (link: https://arxiv.org/pdf/2307.09288.pdf)
    1) Removed all fairscale related stuff from model and rewrite it to pure torch
    2) Removed `batch_dim` from all processing logic, because for inference it always sets to 1
    3) Removed `reshape_for_broadcast` for freqs_cis ← correct shape sets at createion (1, seq_len, 1, head_dim)
    4) Removed repeat_kv functionality, because according to article (Table 1.) only 34b and 70b models used GQA
    5) Changed `apply_rotary_emb` to support cache for seq_len > 1 in `q`
    6) Changed `mask` initialisation to create in Attention (where it is used) using `lru_cache` to avoid recalculation
"""


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000  # defined right now
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32  # not used
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qp, ep = xq.size(1), xk.size(1)  # seq_len (q/k)
    sp = 0 if qp == ep else ep - qp  # if k is cached set offset for q
    # print(f"{xq.size()} < ({sp}, {ep})")
    # print(f"{xk.size()} < (0, {ep})")
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:1, sp:ep]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:1, :ep]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@lru_cache(maxsize=1)
def get_mask(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype):
    if q_len == 1:
        return None
    mask = torch.full((1, 1, q_len, k_len), float("-inf"), device=device, dtype=dtype)
    return mask.triu(diagonal=k_len - q_len + 1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = torch.nn.Linear(in_features=args.dim, out_features=args.dim, bias=False)
        self.wk = torch.nn.Linear(in_features=args.dim, out_features=args.dim, bias=False)
        self.wv = torch.nn.Linear(in_features=args.dim, out_features=args.dim, bias=False)
        self.wo = torch.nn.Linear(in_features=args.dim, out_features=args.dim, bias=False)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # xk and xv are reshaped in hooks
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # possible assymetric rotary
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = xk.permute(0, 2, 3, 1)  # equivalent to key.transpose(1, 2).transpose(2, 3)
        values = xv.transpose(1, 2)

        # Convert scores to float32 here, because mask: `triu` is not implemented for bfloat16??
        scores = torch.matmul(xq, keys).float() / self.scale
        # scores shape: [batch, heads, query, keys]
        mask = get_mask(scores.size(2), scores.size(3), scores.device, scores.dtype)
        # print(f"LRU: {get_mask.cache_info()}")
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores, dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # st = time.time()
        h = x + self.attention.forward(
            self.attention_norm(x), freqs_cis
        )
        # print(f"Attention in: {(time.time() - st):.8f}")
        # st = time.time()
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        # print(f"MLP in: {(time.time() - st):.8f}")
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.layers = torch.nn.ModuleList()
        # layer_id is never used in original code...
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.tok_embeddings = torch.nn.Embedding(self.params.vocab_size, self.params.dim)
        self.output = torch.nn.Linear(self.params.dim, self.params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )[None, :, None, :]  # (1, seq_len, 1, head_dim)

    # @torch.inference_mode()  # not sure what's the difference, but I like `no_grad` more
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor):
        # u_st = time.time()
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # mask = None  # NOTE: mask is created in Attention layer
        for i, layer in enumerate(self.layers):
            # st = time.time()
            h = layer(h, self.freqs_cis)
            # print(f"Calc in {(time.time() - st):.8f}")
        h = self.norm(h).type_as(self.output.weight)
        output = self.output(h).float()
        # print(f"All done in: {(time.time() - u_st):.6f}")
        # input("---")
        return output
