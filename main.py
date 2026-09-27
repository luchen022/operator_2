# XPUOJ_MAGIATTENTION_V3_DIRECT_KERNEL
# Sandbox-minimal correctness/performance baseline. No host-side metadata preprocessing.

import torch
import triton
import triton.language as tl


@triton.jit
def _ffa_fwd_kernel(
    Q,
    K,
    V,
    Q_RANGES,
    K_RANGES,
    ATTN_TYPE,
    SINK,
    OUT,
    softmax_scale,
    S: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    NSLICES: tl.constexpr,
    NSINK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_valid = offs_m < S

    q_ptrs = Q + (offs_m[:, None] * HQ + h) * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    kv_h = h // GROUP

    # Online-softmax state. First fold in Attention Sink logits.
    # Sink contributes to the denominator only, so acc remains zero.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    for sn in tl.static_range(0, NSINK):
        x = tl.load(SINK + sn * HQ + h).to(tl.float32)
        m_new = tl.maximum(m_i, x)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        beta = tl.exp2((x - m_new) * 1.4426950408889634)
        l_i = l_i * alpha + beta
        m_i = m_new

    # Merge all attention slices directly into the same online-softmax state.
    # The problem guarantees that, for a fixed Q position, visible K sets from
    # overlapping slices are disjoint.
    for si in tl.static_range(0, NSLICES):
        qs = tl.load(Q_RANGES + si * 2 + 0).to(tl.int32)
        qe = tl.load(Q_RANGES + si * 2 + 1).to(tl.int32)
        ks = tl.load(K_RANGES + si * 2 + 0).to(tl.int32)
        ke = tl.load(K_RANGES + si * 2 + 1).to(tl.int32)
        typ = tl.load(ATTN_TYPE + si).to(tl.int32)

        q_in_slice = q_valid & (offs_m >= qs) & (offs_m < qe)
        k_len = ke - ks

        for rel_n in tl.range(0, k_len, BLOCK_N, num_stages=2):
            offs_n = ks + rel_n + tl.arange(0, BLOCK_N)
            k_valid = offs_n < ke

            k_ptrs = K + (offs_n[:, None] * HKV + kv_h) * D + offs_d[None, :]
            k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * softmax_scale

            p = offs_m[:, None]
            j = offs_n[None, :]

            # Equivalent global-coordinate forms of the four local masks.
            causal_ok = j <= (p + ke - qe)
            inv_ok = j >= (p + ks - qs)

            visible = (
                (typ == 0)
                | ((typ == 1) & causal_ok)
                | ((typ == 2) & inv_ok)
                | ((typ == 3) & causal_ok & inv_ok)
            )
            visible = visible & q_in_slice[:, None] & k_valid[None, :]
            qk = tl.where(visible, qk, -float("inf"))

            block_max = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p_ij = tl.exp2((qk - m_new[:, None]) * 1.4426950408889634)

            l_i = l_i * alpha + tl.sum(p_ij, axis=1)
            acc = acc * alpha[:, None]

            v_ptrs = V + (offs_n[:, None] * HKV + kv_h) * D + offs_d[None, :]
            vv = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

            # Match the reference precision path: probabilities are rounded to
            # BF16 before P@V; the dot product accumulates in FP32.
            acc += tl.dot(p_ij.to(tl.bfloat16), vv)
            m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = OUT + (offs_m[:, None] * HQ + h) * D + offs_d[None, :]
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_valid[:, None])


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
    s = int(seqlen)
    hq = int(num_q_heads)
    hkv = int(num_kv_heads)
    d = int(head_dim)
    nslices = int(num_slices)
    nsink = int(num_sink)

    if d == 128:
        block_m = 64
        block_n = 64
        num_warps = 8
    else:
        block_m = 128
        block_n = 64
        num_warps = 4

    grid_m = (s + block_m - 1) // block_m

    _ffa_fwd_kernel[(grid_m, hq)](
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        attn_type_map,
        sink,
        output,
        softmax_scale,
        S=s,
        HQ=hq,
        HKV=hkv,
        D=d,
        GROUP=hq // hkv,
        NSLICES=nslices,
        NSINK=nsink,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
