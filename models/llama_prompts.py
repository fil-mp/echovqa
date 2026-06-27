# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    w_bias: bool = False # use bias tuning
    w_lora: bool = False # use lora tuning
    lora_rank: int = 16
    w_new_gate: bool = False # for compatibility

class LLaMAAdapter(nn.Module):
    def __init__(self, hidden_dim, ds_factor, ln_after=False, ln_before=False, dropout=0.1):
        super().__init__()
        assert not hidden_dim % ds_factor, f"Hidden dim {hidden_dim} must be divisible by ds_factor {ds_factor}"
        
        self.down = nn.Linear(hidden_dim, hidden_dim // ds_factor)
        self.up = nn.Linear(hidden_dim // ds_factor, hidden_dim)
        self.act = nn.GELU()  # Using GELU to match LLaMA's activation
        
        # Layer norm flags and layers
        self.ln_after = ln_after
        self.ln_before = ln_before
        if ln_after or ln_before:
            self.ln = nn.LayerNorm(hidden_dim)
            
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module, std=1e-3):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            # Clamp weights for stability
            module.weight.data = torch.clamp(module.weight.data, min=-2 * std, max=2 * std)
            if module.bias is not None:
                module.bias.data = torch.clamp(module.bias.data, min=-2 * std, max=2 * std)
                
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states):
        # Store original input for residual
        residual = hidden_states
        
        # Apply layer norm before if configured
        if self.ln_before:
            hidden_states = self.ln(hidden_states)
            
        # Down project
        hidden_states = self.down(hidden_states)
        hidden_states = self.act(hidden_states)
        
        # Apply dropout if configured
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
            
        # Up project
        hidden_states = self.up(hidden_states)
        
        # Apply layer norm after if configured
        if self.ln_after:
            hidden_states = self.ln(hidden_states)
            
        # Add residual connection
        return residual + hidden_states

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


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.wq.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)

        self.w_lora = args.w_lora
        if args.w_lora:
           self.lora_wq_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wq_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wk_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wk_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wv_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wv_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wo_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wo_l2 = Linear(args.lora_rank, args.dim, bias=False)
           nn.init.constant_(self.lora_wq_l2.weight.data, 0)
           nn.init.constant_(self.lora_wk_l2.weight.data, 0)
           nn.init.constant_(self.lora_wv_l2.weight.data, 0)
           nn.init.constant_(self.lora_wo_l2.weight.data, 0)

        self.cache_k = None
        self.cache_v = None

        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        
        self.w_new_gate = args.w_new_gate
        if args.w_new_gate:
            self.new_gate = torch.nn.Parameter(torch.ones(1, 1, 1, 1))


    def train(self, mode: bool = True):
        if mode:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
        return super().train(mode)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, k=None, v=None):
        bsz, seqlen, _ = x.shape
        if k is not None and v is not None:
            # print(x.dtype)
            # if not self.training:
            #     x = x.to(self.wq.weight.dtype)
            xq = self.wq(x)
            # if self.w_lora:
            #     xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
            # xk, xv = self.wk(k), self.wv(v)  
        
            # if self.w_lora:
            #     xk = xk + self.lora_wk_l2(self.lora_wk_l1(k))
            #     xv = xv + self.lora_wv_l2(self.lora_wv_l1(v))
            
            if self.w_lora:
                # if not self.training:
                #     x_lora = x.to(self.lora_wq_l1.weight.dtype)
                #     xq = xq + self.lora_wq_l2(self.lora_wq_l1(x_lora))
                # else:
                    xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))

            # Project k and v
            # if not self.training:
            #     k = k.to(self.wk.weight.dtype)
            #     v = v.to(self.wv.weight.dtype)

            xk = self.wk(k)
            xv = self.wv(v)

            if self.w_lora:
                # if not self.training:
                #     k_lora = k.to(self.lora_wk_l1.weight.dtype)
                #     v_lora = v.to(self.lora_wv_l1.weight.dtype)
                #     xk = xk + self.lora_wk_l2(self.lora_wk_l1(k_lora))
                #     xv = xv + self.lora_wv_l2(self.lora_wv_l1(v_lora))
                # else:
                    xk = xk + self.lora_wk_l2(self.lora_wk_l1(k))
                    xv = xv + self.lora_wv_l2(self.lora_wv_l1(v))


            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, k.size(1), self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, v.size(1), self.n_local_heads, self.head_dim)

            # For query, use original freqs_cis
            xq, _ = apply_rotary_emb(xq, xq, freqs_cis=freqs_cis)
            
            # For key, extend freqs_cis if needed
            if k.size(1) > seqlen:  # If we have prompts
                num_prompts = k.size(1) - seqlen
                # Pad freqs_cis with zeros for prompt tokens
                prompt_freqs = precompute_freqs_cis(
                    self.head_dim, 
                    num_prompts,
                    theta=10000.0
                ).to(freqs_cis.device)

                # prompt_freqs = torch.ones_like(freqs_cis[:num_prompts])
                extended_freqs = torch.cat([prompt_freqs, freqs_cis], dim=0)
                _, xk = apply_rotary_emb(xk, xk, freqs_cis=extended_freqs)
            else:
                _, xk = apply_rotary_emb(xk, xk, freqs_cis=freqs_cis)    

            ###no prefix -end###

            if not self.training:
                if k is None:  # Only cache when not using prompts
                    self.cache_k = self.cache_k.to(xq)
                    self.cache_v = self.cache_v.to(xq)

                    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

                    keys = self.cache_k[:bsz, : start_pos + seqlen]
                    values = self.cache_v[:bsz, : start_pos + seqlen]
                else:
                    keys = xk
                    values = xv    
            else:
                assert start_pos==0
                keys = xk
                values = xv
        else:
            if not self.training:
                x_main = x.to(self.wq.weight.dtype)
                xq = self.wq(x_main)
                xk = self.wk(x_main)
                xv = self.wv(x_main)

                if self.w_lora:
                    x_lora = x_main.to(self.lora_wq_l1.weight.dtype)
                    
                    lora_q = self.lora_wq_l2(self.lora_wq_l1(x_lora)).to(xq.dtype)
                    lora_k = self.lora_wk_l2(self.lora_wk_l1(x_lora)).to(xk.dtype)
                    lora_v = self.lora_wv_l2(self.lora_wv_l1(x_lora)).to(xv.dtype)

                    xq = xq + lora_q
                    xk = xk + lora_k
                    xv = xv + lora_v
            else:        
                xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
                if self.w_lora:
                    xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
                    xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
                    xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

            if not self.training:
                self.cache_k = self.cache_k.to(xq)
                self.cache_v = self.cache_v.to(xq)

                self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

                keys = self.cache_k[:bsz, : start_pos + seqlen]
                values = self.cache_v[:bsz, : start_pos + seqlen]
            else:
                assert start_pos==0
                keys = xk
                values = xv        

        if adapter is not None:
            # if not self.training:
            #     adapter_len = adapter.shape[1]
            #     adapter_casted = adapter.to(self.wv.weight.dtype)
            #     adapter_v = self.wv(adapter_casted).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
            #     adapter_v = adapter_v.transpose(1, 2)

            #     if adapter_len > 1:
            #         adapter_k = self.wk(adapter_casted).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
            #         adapter_k = adapter_k.transpose(1, 2)
            # else:        
                adapter_len = adapter.shape[1]
                adapter_v = self.wv(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
                adapter_v = adapter_v.transpose(1, 2)

                if adapter_len > 1:
                    adapter_k = self.wk(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
                    adapter_k = adapter_k.transpose(1, 2)


        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)

        if adapter is not None:
            if adapter_len > 1:
                adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
                adapter_scores = self.gate.tanh() * F.softmax(adapter_scores.float(), dim=-1).type_as(xq)
                if self.w_new_gate:
                    adapter_scores = self.new_gate * adapter_scores
                # print(output.dtype, adapter_scores.dtype, adapter_v.dtype) 
                # if not self.training:
                #     adapter_scores = adapter_scores.to(adapter_v.dtype)
   
                output = output + torch.matmul(adapter_scores, adapter_v)
            else:
                output = output + self.gate.tanh() * adapter_v

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        # if self.w_lora:
        #    return self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        # else:
        #    return self.wo(output)

        if self.w_lora:
            # if not self.training:
            #     # Step 1: Cast for LoRA layers
            #     out_lora = output.to(self.lora_wo_l1.weight.dtype)
            #     lora_out = self.lora_wo_l2(self.lora_wo_l1(out_lora))
            #     # Step 2: Cast LoRA output back to match main output dtype
            #     output_casted = output.to(self.wo.weight.dtype)
            #     main_out = self.wo(output_casted)                
            #     if main_out.dtype != lora_out.dtype:
            #         lora_out = lora_out.to(main_out.dtype)
            #     return main_out + lora_out
            # else:
                return self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        else:
            return self.wo(output)
   


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        args: ModelArgs
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=args.w_bias
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)

        self.w_lora = args.w_lora
        if args.w_lora:
           self.lora_w1_l1 = Linear(dim, args.lora_rank, bias=False)
           self.lora_w1_l2 = Linear(args.lora_rank, hidden_dim, bias=False)
           self.lora_w2_l1 = Linear(hidden_dim, args.lora_rank, bias=False)
           self.lora_w2_l2 = Linear(args.lora_rank, dim, bias=False)
           self.lora_w3_l1 = Linear(dim, args.lora_rank, bias=False)
           self.lora_w3_l2 = Linear(args.lora_rank, hidden_dim, bias=False)
           nn.init.constant_(self.lora_w1_l2.weight.data, 0)
           nn.init.constant_(self.lora_w2_l2.weight.data, 0)
           nn.init.constant_(self.lora_w3_l2.weight.data, 0)

    def forward(self, x):
        if self.w_lora:
        #    out = F.silu(self.w1(x) + self.lora_w1_l2(self.lora_w1_l1(x))) * (self.w3(x) + self.lora_w3_l2(self.lora_w3_l1(x)))
        #    return self.w2(out) + self.lora_w2_l2(self.lora_w2_l1(out))
            # if not self.training:
            #     # Cast x for LoRA path
            #     x_main = x.to(self.w1.weight.dtype)

            #     # LoRA 1 and 3
            #     lora1 = self.lora_w1_l2(self.lora_w1_l1(x_main.to(self.lora_w1_l1.weight.dtype))).to(x_main.dtype)
            #     lora3 = self.lora_w3_l2(self.lora_w3_l1(x_main.to(self.lora_w3_l1.weight.dtype))).to(x_main.dtype)

            #     out = F.silu(self.w1(x_main) + lora1) * (self.w3(x_main) + lora3)

            #     # LoRA 2
            #     out_lora2 = self.lora_w2_l2(self.lora_w2_l1(out.to(self.lora_w2_l1.weight.dtype)))
            #     out_lora2 = out_lora2.to(out.dtype)

            #     # Final output
            #     main_out = self.w2(out.to(self.w2.weight.dtype))
            #     return main_out + out_lora2
            # else:
                out = F.silu(self.w1(x) + self.lora_w1_l2(self.lora_w1_l1(x))) * (self.w3(x) + self.lora_w3_l2(self.lora_w3_l1(x)))
                return self.w2(out) + self.lora_w2_l2(self.lora_w2_l1(out))

        else:
           return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        # Original components
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # # Add adapters
        # self.attn_adapter = LLaMAAdapter(
        #     hidden_dim=args.dim,
        #     ds_factor=8,  # Configurable downsampling factor
        #     ln_after=True,
        #     dropout=0.1
        # )
        # self.ffn_adapter = LLaMAAdapter(
        #     hidden_dim=args.dim,
        #     ds_factor=8,  # Configurable downsampling factor
        #     ln_after=True,
        #     dropout=0.1
        # )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, k=None, v=None):
        
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, k, v)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()