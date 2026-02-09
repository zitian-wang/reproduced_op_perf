import os
import torch
from contextlib import nullcontext
from transformer_engine.pytorch.attention.dot_product_attention.backends import FusedAttention
from torch.profiler import profile, record_function, ProfilerActivity

os.environ['NVTE_CK_USES_BWD_V3'] = '1'

softmax_scale=0.08838834764831843
attention_type='self'
layer_number=1
deterministic=False
attn_kwargs = {'attention_dropout': 0.0,
              'attention_dropout_ctx': nullcontext}
seqlen = 1024*32
device = torch.device("cuda")
query_layer = torch.randn((seqlen,16,192), dtype = torch.bfloat16, device = device, requires_grad = True)
key_layer = torch.randn((seqlen,16,192), dtype = torch.bfloat16, device = device, requires_grad = True)
value_layer = torch.randn((seqlen,16,128), dtype = torch.bfloat16, device = device, requires_grad = True)
#cu_seqlens_q = cu_seqlens_kv = torch.tensor([0,  3602,  5672,  9207, 11229, 14814, 15286, 18924, 19417, 20504, 24106, 26087, 29675, 32768], dtype=torch.int32).to(device)
cu_seqlens_q = cu_seqlens_kv = torch.tensor([0,  seqlen], dtype=torch.int32).to(device)
attention_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool().unsqueeze(0).unsqueeze(0).to(device)
# attention_mask = torch.load('attention_mask.pt').to(device)
max_seqlen = seqlen

fused_attention = FusedAttention(
    softmax_scale,
    attention_type=attention_type,
    layer_number=layer_number,
    deterministic=deterministic,
    **attn_kwargs,
)

profiler0 = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=0,
        warmup=5,
        active=10,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
    record_shapes=True,
    #profile_memory=True,
    with_stack=True
)

n_query_layer = torch.randn((seqlen,16,128), dtype = torch.bfloat16, device = device, requires_grad = True)
n_key_layer = torch.randn((seqlen,16,128), dtype = torch.bfloat16, device = device, requires_grad = True)

with profiler0:
    for i in range(15):
        out = fused_attention(
            n_query_layer,
            n_key_layer,
            value_layer,
            qkv_layout='thd_thd_thd',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            attn_mask_type='padding_causal',
            attention_mask=attention_mask,
            window_size=(-1,0),
            fused_attention_backend=1, #tex.NVTE_Fused_Attn_Backend.NVTE_CK
            core_attention_bias_type='no_bias',
            core_attention_bias=None,
            fast_zero_fill=True,
            cp_group=None,
            cp_global_ranks=None,
            cp_stream=None,
            cp_comm_type='p2p',
            fp8=False,
            fp8_meta={},
            quantizers={},
            pad_between_seqs=False,
            inference_params=None,
        )
        out.sum().backward()
        q_grad = n_query_layer.grad.clone()
        n_query_layer.grad = None
        profiler0.step()
print(profiler0.key_averages().table(sort_by="cuda_time_total",row_limit=10))


profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=0,
        active=10,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
    record_shapes=True,
    #profile_memory=True,
    with_stack=True
)



with profiler:
    for i in range(15):
        out = fused_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_layout='thd_thd_thd',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            attn_mask_type='padding_causal',
            attention_mask=attention_mask,
            window_size=(-1,0),
            fused_attention_backend=1, #tex.NVTE_Fused_Attn_Backend.NVTE_CK
            core_attention_bias_type='no_bias',
            core_attention_bias=None,
            fast_zero_fill=True,
            cp_group=None,
            cp_global_ranks=None,
            cp_stream=None,
            cp_comm_type='p2p',
            fp8=False,
            fp8_meta={},
            quantizers={},
            pad_between_seqs=False,
            inference_params=None,
        )
        out.sum().backward()
        q_grad = query_layer.grad.clone()
        query_layer.grad = None
        profiler.step()
print(profiler.key_averages().table(sort_by="cuda_time_total",row_limit=10))


# modify value

profiler2 = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=0,
        active=10,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
    record_shapes=True,
    #profile_memory=True,
    with_stack=True
)



with profiler2:
    for i in range(15):
        # Pad value_layer from 128 to 192 dimensions
        original_head_dim = value_layer.shape[-1]
        padded_head_dim = 192
        pad_part = torch.zeros((value_layer.shape[0],value_layer.shape[1], padded_head_dim - original_head_dim)).to(value_layer.dtype).to(value_layer.device)
        pad_part = pad_part.detach()   # ~X~N确梯度~M~A~O~G~Y~G~L
        value_layer_padded = torch.cat([value_layer, pad_part], dim=-1)


        # Compute with padded tensors
        out_padded = fused_attention(
            query_layer,
            key_layer,
            value_layer_padded,
            qkv_layout='thd_thd_thd',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            attn_mask_type='padding_causal',
            attention_mask=attention_mask,
            window_size=(-1,0),
            fused_attention_backend=1, #tex.NVTE_Fused_Attn_Backend.NVTE_CK
            core_attention_bias_type='no_bias',
            core_attention_bias=None,
            fast_zero_fill=True,
            cp_group=None,
            cp_global_ranks=None,
            cp_stream=None,
            cp_comm_type='p2p',
            fp8=False,
            fp8_meta={},
            quantizers={},
            pad_between_seqs=False,
            inference_params=None,
        )
        # Reshape from (32768, 384) to (32768, 2, 192), take first 128 dims, reshape back

        out_padded = out_padded.reshape(value_layer.shape[0],value_layer.shape[1], -1)
        out_padded = out_padded[:,:, :original_head_dim]
        out_padded = out_padded.reshape(out_padded.shape[0],-1)
        out_padded.sum().backward()
        q_grad_padded = query_layer.grad.clone()
        query_layer.grad = None
        profiler2.step()
print(profiler2.key_averages().table(sort_by="cuda_time_total",row_limit=10))





# Calculate differences
abs_diff = torch.abs(out - out_padded)
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()
relative_diff = (abs_diff / (torch.abs(out) + 1e-8)).mean().item()

print("=" * 50)
print("Comparison between out and out_padded:")
print(f"  Max absolute difference: {max_diff}")
print(f"  Mean absolute difference: {mean_diff}")
print(f"  Mean relative difference: {relative_diff}")
print(f"  out shape: {out.shape}")
print(f"  out_padded shape: {out_padded.shape}")
print("=" * 50)


# Calculate q_grad differences
abs_diff = torch.abs(q_grad - q_grad_padded)
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()
relative_diff = (abs_diff / (torch.abs(q_grad) + 1e-8)).mean().item()

print("=" * 50)
print("Comparison between q_grad and q_grad_padded:")
print(f"  Max absolute difference: {max_diff}")
print(f"  Mean absolute difference: {mean_diff}")
print(f"  Mean relative difference: {relative_diff}")
print(f"  q_grad shape: {q_grad.shape}")
print(f"  q_grad_padded shape: {q_grad_padded.shape}")
print("=" * 50)
