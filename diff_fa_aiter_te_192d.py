import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from flash_attn import flash_attn_varlen_func

device = torch.device("cuda:0")
dtype = torch.bfloat16

total_seqlen = 1024 * 32  # 32768
num_heads = 16
head_dim = 192
softmax_scale = head_dim ** -0.5
causal = True

is_rocm = torch.version.hip is not None
is_cuda = torch.version.cuda is not None and not is_rocm
backend = "rocm" if is_rocm else "cuda" if is_cuda else "unknown"

aiter_flash_attn_varlen_func = None
fused_attention = None
dot_product_attention = None


def get_aiter_flash_attn_varlen_func():
    global aiter_flash_attn_varlen_func
    if aiter_flash_attn_varlen_func is None:
        from aiter import flash_attn_varlen_func as func

        aiter_flash_attn_varlen_func = func
    return aiter_flash_attn_varlen_func


def get_fused_attention():
    global fused_attention
    if fused_attention is None:
        os.environ["NVTE_CK_USES_BWD_V3"] = "1"
        from transformer_engine.pytorch.attention.dot_product_attention.backends import FusedAttention

        fused_attention = FusedAttention(
            softmax_scale,
            attention_type="self",
            layer_number=1,
            deterministic=False,
            attention_dropout=0.0,
            attention_dropout_ctx=nullcontext,
        )
    return fused_attention


def get_dot_product_attention():
    global dot_product_attention
    if dot_product_attention is None:
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_UNFUSED_ATTN"] = "0"

        from transformer_engine.pytorch import DotProductAttention

        dot_product_attention = DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            attention_dropout=0.0,
            qkv_format="thd",
            attn_mask_type="padding_causal",
            softmax_scale=softmax_scale,
        ).to(device)
        dot_product_attention.train()
    return dot_product_attention


def run_te_rocm(cu, max_seqlen):
    attention = get_fused_attention()
    out = attention(
        q, k, v,
        qkv_layout="thd_thd_thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        attn_mask_type="padding_causal",
        attention_mask=None,
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
    if out.ndim == 2:
        out = out.view_as(q)
    return out


def run_te_cuda(cu, max_seqlen):
    attention = get_dot_product_attention()
    out = attention(
        q,
        k,
        v,
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        attn_mask_type="padding_causal",
    )
    if out.ndim == 2:
        out = out.view_as(q)
    return out


def run_te(cu, max_seqlen):
    if is_rocm:
        return run_te_rocm(cu, max_seqlen)
    return run_te_cuda(cu, max_seqlen)


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


def run_flash_attn(cu, max_seqlen):
    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale, causal=causal,
    )


def run_aiter(cu, max_seqlen):
    aiter_flash_attn = get_aiter_flash_attn_varlen_func()
    out, _ = aiter_flash_attn(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale, causal=causal, return_lse=True,
    )
    return out


def run_pytorch(cu, max_seqlen):
    outputs = []
    for i in range(cu.numel() - 1):
        start, end = cu[i].item(), cu[i + 1].item()
        q_i = q[start:end].transpose(0, 1).unsqueeze(0)
        k_i = k[start:end].transpose(0, 1).unsqueeze(0)
        v_i = v[start:end].transpose(0, 1).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def make_implementations():
    implementations = [("flash_attn", run_flash_attn)]
    if is_rocm:
        implementations.extend(
            [
                ("aiter", run_aiter),
                ("transformer_engine", run_te),
                ("pytorch", run_pytorch),
            ]
        )
    elif is_cuda:
        implementations.extend(
            [
                ("transformer_engine", run_te),
                ("pytorch", run_pytorch),
            ]
        )
    else:
        raise RuntimeError("This benchmark expects a CUDA or ROCm PyTorch build.")
    return implementations


def run_profile(label, fn, cu, max_seqlen, config_name):
    clear_grads()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        out = fn(cu, max_seqlen)
        out.sum().backward()
        q_grad = q.grad.clone()
    clear_grads()
    print(f"--- {label} ({config_name}) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return out, q_grad


implementations = make_implementations()
print(f"backend: {backend}")

for name, cu, max_seqlen in configs:
    print(f"\n========== config: {name} (max_seqlen={max_seqlen}, num_seqs={cu.numel() - 1}) ==========")

    for _ in range(5):  # warmup
        for _, fn in implementations:
            out = fn(cu, max_seqlen)
            out.sum().backward()
            clear_grads()

    profiled = []
    for label, fn in implementations:
        out, q_grad = run_profile(label, fn, cu, max_seqlen, name)
        profiled.append((label, out, q_grad))

    reference_label, reference_out, reference_q_grad = profiled[0]
    print(f"[{name}] q_grad nan:")
    for label, _, q_grad in profiled:
        print(f"  {label}: {q_grad.isnan().any().item()}")

    for label, out, q_grad in profiled[1:]:
        output_diff = (out.detach() - reference_out.detach()).abs()
        grad_diff = (q_grad.detach() - reference_q_grad.detach()).abs()
        print(f"[{name}] diff output {label}-{reference_label}, mean/max:", output_diff.mean().item(), output_diff.max().item())
        print(f"[{name}] diff grad   {label}-{reference_label}, mean/max:", grad_diff.mean().item(), grad_diff.max().item())
