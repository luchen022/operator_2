import tilelang
import tilelang.language as T

# TileLang FlashAttention experiment v1.
# Generic correctness path for all scored shapes; Python-level JIT specializes
# S/Hq/Hkv/D/N/Ns/scale for each testcase.
_KERNEL_CACHE = {}
_PRINTED_BUILD = False

_LOG2E = 1.4426950408889634


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def _build_ffa_kernel(
    S,
    Hq,
    Hkv,
    D,
    N,
    Ns,
    softmax_scale,
    block_M=64,
    block_N=64,
    num_stages=2,
    threads=128,
):
    G = Hq // Hkv
    scale_log2 = softmax_scale * _LOG2E
    # Finite masking sentinel avoids (-inf)-(-inf) in online softmax when a
    # Q row has no visible keys in an early K tile (e.g. shifted/full slices).
    neg_large = -1.0e30

    q_shape = (S, Hq, D)
    kv_shape = (S, Hkv, D)
    qrange_shape = (N, 2)
    krange_shape = (N, 2)
    type_shape = (N,)
    sink_shape = (Ns, Hq)

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype="bfloat16"),
        K: T.Tensor(kv_shape, dtype="bfloat16"),
        V: T.Tensor(kv_shape, dtype="bfloat16"),
        QRanges: T.Tensor(qrange_shape, dtype="int32"),
        KRanges: T.Tensor(krange_shape, dtype="int32"),
        Types: T.Tensor(type_shape, dtype="int32"),
        Sink: T.Tensor(sink_shape, dtype="float32"),
        Out: T.Tensor(q_shape, dtype="bfloat16"),
    ):
        with T.Kernel(T.ceildiv(S, block_M), Hq, threads=threads) as (bx, by):
            kvh = by // G

            Q_shared = T.alloc_shared((block_M, D), dtype="bfloat16")
            K_shared = T.alloc_shared((block_N, D), dtype="bfloat16")
            V_shared = T.alloc_shared((block_N, D), dtype="bfloat16")
            O_shared = T.alloc_shared((block_M, D), dtype="bfloat16")

            # FP32 QK tile and online-softmax state.
            acc_s = T.alloc_fragment((block_M, block_N), dtype="float32")
            mult = T.alloc_fragment((block_M, block_N), dtype="float32")
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype="bfloat16")
            acc_o = T.alloc_fragment((block_M, D), dtype="float32")

            scores_max = T.alloc_fragment((block_M,), dtype="float32")
            scores_max_prev = T.alloc_fragment((block_M,), dtype="float32")
            scores_scale = T.alloc_fragment((block_M,), dtype="float32")
            scores_sum = T.alloc_fragment((block_M,), dtype="float32")
            logsum = T.alloc_fragment((block_M,), dtype="float32")

            # Metadata is tiny (N <= 10 in scored cases), cache it in shared.
            q_ranges_s = T.alloc_shared((N, 2), dtype="int32")
            k_ranges_s = T.alloc_shared((N, 2), dtype="int32")
            types_s = T.alloc_shared((N,), dtype="int32")

            T.copy(QRanges, q_ranges_s)
            T.copy(KRanges, k_ranges_s)
            T.copy(Types, types_s)

            T.copy(
                Q[bx * block_M : (bx + 1) * block_M, by, :],
                Q_shared,
            )

            T.fill(acc_o, 0.0)
            T.fill(logsum, 0.0)
            T.fill(scores_max, neg_large)

            for kb in T.Pipelined(T.ceildiv(S, block_N), num_stages=num_stages):
                T.copy(
                    K[kb * block_N : (kb + 1) * block_N, kvh, :],
                    K_shared,
                )

                # Build exact visibility multiplicity for this Q/K tile.
                # A key can appear through multiple overlapping active slices;
                # multiplicity preserves the repeated-softmax-entry semantics.
                for i, j in T.Parallel(block_M, block_N):
                    qidx = bx * block_M + i
                    kidx = kb * block_N + j
                    mult[i, j] = 0.0

                    for s in T.serial(N):
                        qs = q_ranges_s[s, 0]
                        qe = q_ranges_s[s, 1]
                        ks = k_ranges_s[s, 0]
                        ke = k_ranges_s[s, 1]
                        typ = types_s[s]

                        r = qidx - qs
                        u = kidx - ks
                        delta = (ke - ks) - (qe - qs)

                        q_active = (qidx >= qs) & (qidx < qe)
                        k_active = (kidx >= ks) & (kidx < ke)

                        mask_ok = (
                            (typ == 0)
                            | ((typ == 1) & (u <= r + delta))
                            | ((typ == 2) & (u >= r))
                            | ((typ == 3) & (u >= r) & (u <= r + delta))
                        )

                        if q_active & k_active & mask_ok:
                            mult[i, j] = mult[i, j] + 1.0

                    acc_s[i, j] = T.if_then_else(
                        mult[i, j] > 0.0,
                        0.0,
                        neg_large,
                    )

                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # Online softmax.
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(acc_s.dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp2(
                        (scores_max_prev[i] - scores_max[i]) * scale_log2
                    )

                # Keep the raw exp in FP32; multiplicity weights the softmax
                # mass, but BF16 rounding happens before the multiplicity is
                # applied to P@V (matching duplicate identical slice entries).
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(
                        (acc_s[i, j] - scores_max[i]) * scale_log2
                    )

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = acc_s[i, j] * mult[i, j]

                T.reduce_sum(acc_s, scores_sum, dim=1)

                for i in T.Parallel(block_M):
                    logsum[i] = (
                        logsum[i] * scores_scale[i] + scores_sum[i]
                    )

                # For P@V, convert the weighted unnormalized probabilities to
                # BF16 exactly where the reference/Triton path loses precision.
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, D):
                    acc_o[i, j] = acc_o[i, j] * scores_scale[i]

                T.copy(
                    V[kb * block_N : (kb + 1) * block_N, kvh, :],
                    V_shared,
                )
                T.gemm(
                    acc_s_cast,
                    V_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            # Attention Sink contributes denominator mass only.
            # scores_max is an unscaled QK maximum, so convert it to the same
            # log2 domain used by the exponentials.
            for i in T.Parallel(block_M):
                for s in T.serial(Ns):
                    logsum[i] = logsum[i] + T.exp2(
                        Sink[s, by] * _LOG2E
                        - scores_max[i] * scale_log2
                    )

            for i, j in T.Parallel(block_M, D):
                acc_o[i, j] = acc_o[i, j] / logsum[i]

            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Out[bx * block_M : (bx + 1) * block_M, by, :],
            )

    return kernel


def _get_kernel(S, Hq, Hkv, D, N, Ns, scale):
    # Keep scale in the specialization key: the competition uses a fixed scale
    # per testcase, and baking it in removes scalar work from the timed path.
    key = (S, Hq, Hkv, D, N, Ns, float(scale))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        # Conservative first config based directly on TileLang's official GQA
        # FlashAttention example. Later versions will specialize per testcase.
        kernel = _build_ffa_kernel(
            S,
            Hq,
            Hkv,
            D,
            N,
            Ns,
            float(scale),
            block_M=64,
            block_N=64,
            num_stages=2,
            threads=128,
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


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
    N = int(num_slices)
    Ns = int(num_sink)
    scale = float(softmax_scale)

    global _PRINTED_BUILD
    if not _PRINTED_BUILD:
        print("BUILD TILELANG_GENERIC_FA_V2_FINITE_MASK")
        _PRINTED_BUILD = True

    kernel = _get_kernel(S, Hq, Hkv, D, N, Ns, scale)
    kernel(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        attn_type_map,
        sink,
        output,
    )
