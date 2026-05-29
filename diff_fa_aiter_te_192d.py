import os
from contextlib import nullcontext

import torch
from torch.profiler import profile, ProfilerActivity
from flash_attn import flash_attn_varlen_func
from aiter import flash_attn_varlen_func as flash_attn_varlen_func_aiter

os.environ["NVTE_CK_USES_BWD_V3"] = "1"
from transformer_engine.pytorch.attention.dot_product_attention.backends import FusedAttention

device = torch.device("cuda:0")
dtype = torch.bfloat16

total_seqlen = 1024 * 32  # 32768
num_heads = 16
head_dim = 192
softmax_scale = 0.08838834764831843
causal = True

fused_attention = FusedAttention(
    softmax_scale,
    attention_type="self",
    layer_number=1,
    deterministic=False,
    attention_dropout=0.0,
    attention_dropout_ctx=nullcontext,
)
attention_mask = torch.triu(
    torch.ones(total_seqlen, total_seqlen, device=device), diagonal=1
).bool().unsqueeze(0).unsqueeze(0)


def run_te(cu, max_seqlen):
    return fused_attention(
        q, k, v,
        qkv_layout="thd_thd_thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        attn_mask_type="padding_causal",
        attention_mask=attention_mask,
        window_size=(-1, 0),
        fused_attention_backend=1,  # tex.NVTE_Fused_Attn_Backend.NVTE_CK
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
        fast_zero_fill=True,
        cp_group=None,
        cp_global_ranks=None,
        cp_stream=None,
        cp_comm_type="p2p",
        fp8=False,
        fp8_meta={},
        quantizers={},
        pad_between_seqs=False,
        inference_params=None,
    )

q = torch.randn((total_seqlen, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((total_seqlen, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((total_seqlen, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)

cu_single = torch.tensor([0, total_seqlen], dtype=torch.int32, device=device)
max_seqlen_single = total_seqlen

cu_varlen_list = [0, 3602, 5672, 9207, 11229, 14814, 15286, 18924, 19417, 20504, 24106, 26087, 29675, 32768]
cu_varlen = torch.tensor(cu_varlen_list, dtype=torch.int32, device=device)
max_seqlen_varlen = max(b - a for a, b in zip(cu_varlen_list[:-1], cu_varlen_list[1:]))  # 3638

configs = [
    ("single_long_seq", cu_single, max_seqlen_single),
    ("varlen_batch", cu_varlen, max_seqlen_varlen),
]


def clear_grads():
    q.grad = None
    k.grad = None
    v.grad = None


for name, cu, max_seqlen in configs:
    print(f"\n========== config: {name} (max_seqlen={max_seqlen}, num_seqs={cu.numel() - 1}) ==========")

    for _ in range(5):  # warmup
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale, causal=causal,
        )
        out.sum().backward()
        clear_grads()

        out2, _ = flash_attn_varlen_func_aiter(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale, causal=causal, return_lse=True,
        )
        out2.sum().backward()
        clear_grads()

        out3 = run_te(cu, max_seqlen)
        out3.sum().backward()
        clear_grads()

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale, causal=causal,
        )
        out.sum().backward()
        q_grad = q.grad.clone()
        clear_grads()
    print(f"--- flash_attn ({name}) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        out2, _ = flash_attn_varlen_func_aiter(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale, causal=causal, return_lse=True,
        )
        out2.sum().backward()
        q_grad2 = q.grad.clone()
        clear_grads()
    print(f"--- aiter ({name}) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        out3 = run_te(cu, max_seqlen)
        out3.sum().backward()
        q_grad3 = q.grad.clone()
        clear_grads()
    print(f"--- transformer_engine ({name}) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(
        f"[{name}] q_grad nan (fa/aiter/te):",
        q_grad.isnan().any().item(), q_grad2.isnan().any().item(), q_grad3.isnan().any().item(),
    )
    print(f"[{name}] diff output aiter-fa, mean/max:", (out2 - out).abs().mean().item(), (out2 - out).abs().max().item())
    print(f"[{name}] diff grad   aiter-fa, mean/max:", (q_grad2 - q_grad).abs().mean().item(), (q_grad2 - q_grad).abs().max().item())
