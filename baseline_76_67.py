import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    ],
    key=["seqlen", "head_dim", "num_slices"],
)
@triton.jit
def custom_fwd_kernel(
    Q, K, V, Out,
    q_ranges_ptr, k_ranges_ptr, attn_type_map_ptr, sink_ptr,
    softmax_scale,
    seqlen, num_q_heads, num_kv_heads, head_dim, num_slices, num_sink,
    stride_qz, stride_qh, stride_qd,
    stride_kz, stride_kh, stride_kd,
    stride_vz, stride_vh, stride_vd,
    stride_oz, stride_oh, stride_od,
    stride_sink_s, stride_sink_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_SLICES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    group_size = num_q_heads // num_kv_heads
    pid_kv = pid_h // group_size

    block_q_start = pid_m * BLOCK_M
    block_q_end = block_q_start + BLOCK_M
    offs_m = block_q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < seqlen

    q_ptrs = Q + offs_m[:, None] * stride_qz + pid_h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for s in range(0, MAX_SLICES):
        if s < num_slices:
            qs = tl.load(q_ranges_ptr + s * 2)
            qe = tl.load(q_ranges_ptr + s * 2 + 1)
            ks = tl.load(k_ranges_ptr + s * 2)
            ke = tl.load(k_ranges_ptr + s * 2 + 1)
            attn_type = tl.load(attn_type_map_ptr + s)

            has_q_overlap = (block_q_end > qs) & (block_q_start < qe)
            Lq = qe - qs
            Lk = ke - ks
            in_q_slice = (offs_m >= qs) & (offs_m < qe) & mask_m

            num_k_blocks = tl.cdiv(Lk, BLOCK_N)
            r_max = tl.minimum(block_q_end - 1, qe - 1) - qs
            causal_limit = r_max + (Lk - Lq)
            causal_blocks = tl.cdiv(causal_limit + 1, BLOCK_N)
            causal_blocks = tl.maximum(causal_blocks, 0)

            actual_k_blocks = tl.where(attn_type == 1, tl.minimum(num_k_blocks, causal_blocks), num_k_blocks)
            actual_k_blocks = tl.where(has_q_overlap, actual_k_blocks, 0)

            for b in range(0, actual_k_blocks):
                offs_n = ks + b * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < ke

                k_ptrs = K + offs_n[:, None] * stride_kz + pid_kv * stride_kh + offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

                qk = tl.dot(q, tl.trans(k)) * softmax_scale

                r = offs_m - qs
                u = offs_n - ks

                if attn_type == 0:
                    vis = in_q_slice[:, None] & mask_n[None, :]
                elif attn_type == 1:
                    vis = in_q_slice[:, None] & mask_n[None, :] & (u[None, :] <= (r[:, None] + (Lk - Lq)))
                elif attn_type == 2:
                    vis = in_q_slice[:, None] & mask_n[None, :] & (u[None, :] >= r[:, None])
                else:
                    vis = in_q_slice[:, None] & mask_n[None, :] & (u[None, :] >= r[:, None]) & (u[None, :] <= (r[:, None] + (Lk - Lq)))

                qk = tl.where(vis, qk, -float("inf"))

                m_ij = tl.max(qk, 1)
                m_new = tl.maximum(m_i, m_ij)

                m_i_clamped = tl.maximum(m_i, -1e9)
                m_new_clamped = tl.maximum(m_new, -1e9)
                alpha = tl.where(m_i > -float("inf"), tl.math.exp(m_i_clamped - m_new_clamped), 0.0)

                qk_clamped = tl.where(vis, qk, -1e9)
                delta_qk = qk_clamped - m_new_clamped[:, None]
                p = tl.where(vis, tl.math.exp(delta_qk), 0.0)
                p_bf16 = p.to(tl.bfloat16)

                v_ptrs = V + offs_n[:, None] * stride_vz + pid_kv * stride_vh + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

                acc = acc * alpha[:, None]
                acc = tl.dot(p_bf16, v, acc)
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_new

    has_keys = l_i > 0.0
    acc = acc / tl.where(has_keys[:, None], l_i[:, None], 1.0)
    acc = tl.where(has_keys[:, None], acc, 0.0)

    if num_sink > 0:
        offs_sink = tl.arange(0, 16)
        mask_sink = offs_sink < num_sink
        sink_vals = tl.load(sink_ptr + offs_sink * stride_sink_s + pid_h * stride_sink_h, mask=mask_sink, other=-float("inf"))
        sink_max = tl.max(sink_vals, 0)
        sink_sum = tl.sum(tl.math.exp(sink_vals - sink_max), 0)
        sink_lse = sink_max + tl.math.log(sink_sum)

        token_lse = m_i + tl.math.log(tl.maximum(l_i, 1e-20))
        m_final = tl.maximum(token_lse, sink_lse)
        num = tl.math.exp(token_lse - m_final)
        den = num + tl.math.exp(sink_lse - m_final)
        sink_scale = num / den
        acc = acc * tl.where(has_keys[:, None], sink_scale[:, None], 0.0)

    out_ptrs = Out + offs_m[:, None] * stride_oz + pid_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


def run_kernel(
    q,
    k,
    v,
    q_ranges,
    k_ranges,
    attn_type_map,
    sink,
    output,
    softmax_scale,
    seqlen,
    num_q_heads,
    num_kv_heads,
    head_dim,
    num_slices,
    num_sink,
):
    S = int(seqlen)
    Hq = int(num_q_heads)
    Hkv = int(num_kv_heads)
    D = int(head_dim)
    Ns = int(num_sink)
    N = int(num_slices)
    scale = float(softmax_scale)

    MAX_SLICES = 1 if N == 1 else 16
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), Hq)

    custom_fwd_kernel[grid](
        q, k, v, output,
        q_ranges, k_ranges, attn_type_map, sink,
        scale,
        S, Hq, Hkv, D, N, Ns,
        Hq * D, D, 1,
        Hkv * D, D, 1,
        Hkv * D, D, 1,
        Hq * D, D, 1,
        Hq, 1,
        BLOCK_D=D,
        MAX_SLICES=MAX_SLICES,
    )
