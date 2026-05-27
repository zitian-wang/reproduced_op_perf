import os

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from flash_attn import flash_attn_varlen_func

# Make the Transformer Engine comparison use TE/cuDNN fused attention, not its
# flash-attn dispatch path, so it is a real replacement for the missing aiter.
os.environ["NVTE_FLASH_ATTN"] = "0"
os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_UNFUSED_ATTN"] = "0"

device = torch.device("cuda:0")
q = torch.randn((68496, 16, 72), dtype = torch.bfloat16, device = device, requires_grad = True)
k = torch.randn((68496, 16, 72), dtype = torch.bfloat16, device = device, requires_grad = True)
v = torch.randn((68496, 16, 72), dtype = torch.bfloat16, device = device, requires_grad = True)

cu_q = torch.tensor([0,  3928,  6111,  9866, 11515, 14854, 17020, 19084, 22679, 25888,
        29416, 32193, 35665, 39053, 42166, 45280, 49252, 53121, 56984, 60892, 63234, 66823, 68496], dtype=torch.int32).to(device)
cu_k = torch.tensor([0,  3928,  6111,  9866, 11515, 14854, 17020, 19084, 22679, 25888,
        29416, 32193, 35665, 39053, 42166, 45280, 49252, 53121, 56984, 60892, 63234, 66823, 68496], dtype=torch.int32).to(device)


causal = True
softmax_scale = 0.08838834764831845


def make_te_attention():
    from transformer_engine.pytorch import DotProductAttention

    return DotProductAttention(
        num_attention_heads=q.shape[1],
        kv_channels=q.shape[2],
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
        softmax_scale=softmax_scale,
    ).to(device)


te_attention = make_te_attention()
te_attention.train()


def clear_grads():
    q.grad = None
    k.grad = None
    v.grad = None


def run_flash_attn():
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def run_transformer_engine():
    out = te_attention(
        q,
        k,
        v,
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=4096,
        max_seqlen_kv=4096,
        attn_mask_type="padding_causal",
    )
    if out.ndim == 2:
        out = out.view_as(q)
    return out


def run_sdpa():
    outputs = []
    for i in range(cu_q.numel() - 1):
        q_start, q_end = cu_q[i].item(), cu_q[i + 1].item()
        k_start, k_end = cu_k[i].item(), cu_k[i + 1].item()
        q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
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


def run_profile(label, fn):
    clear_grads()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        with record_function(label):
            out = fn()
            out.sum().backward()
            q_grad = q.grad.clone()
    clear_grads()
    print(f"\n{label}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return out, q_grad


for i in range(5): #warmup
    for fn in (run_flash_attn, run_transformer_engine, run_sdpa):
        out = fn()
        out.sum().backward()
        clear_grads()


out_flash, q_grad_flash = run_profile("flash-attn", run_flash_attn)
out_te, q_grad_te = run_profile("transformer-engine", run_transformer_engine)
out_sdpa, q_grad_sdpa = run_profile("pytorch-sdpa", run_sdpa)


def print_diff(label, actual, expected):
    diff = (actual.detach() - expected.detach()).abs()
    print(f'{label}, mean/max: ', diff.mean().item(), diff.max().item())


print('q_grad flash/te/sdpa is nan: ', q_grad_flash.isnan().any(), q_grad_te.isnan().any(), q_grad_sdpa.isnan().any())
print_diff('diff output flash/te', out_te, out_flash)
print_diff('diff grad flash/te ', q_grad_te, q_grad_flash)
print_diff('diff output flash/sdpa', out_sdpa, out_flash)
print_diff('diff grad flash/sdpa ', q_grad_sdpa, q_grad_flash)
