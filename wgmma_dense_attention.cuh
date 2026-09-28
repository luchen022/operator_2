#pragma once

#include "wgmma_sm90_core.cuh"
#include <math_constants.h>

namespace wgmma_attention {

using namespace wgmma_sm90;

constexpr int HD = 128;
constexpr int M = 64;
constexpr int N = 64;
constexpr int K16 = 16;
constexpr int K_SLICES_QK = HD / K16;   // 8
constexpr int K_SLICES_PV = N / K16;    // 4
constexpr int BLOCK_ELEMS = M * K16;    // 1024 BF16
constexpr int BLOCK_VECS = BLOCK_ELEMS / 8; // 128 uint4

__device__ __forceinline__ void zero32(float (&x)[32]) {
#pragma unroll
    for (int i = 0; i < 32; ++i) x[i] = 0.0f;
}

__device__ __forceinline__ void rescale_fragment(
    float (&x)[32], float alpha0, float alpha1
) {
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        x[4 * g + 0] *= alpha0;
        x[4 * g + 1] *= alpha0;
        x[4 * g + 2] *= alpha1;
        x[4 * g + 3] *= alpha1;
    }
}

__device__ __forceinline__ void stage_q64_d128(
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ s,
    int token0,
    int kvh,
    int Hq,
    int G
) {
    const int tid = threadIdx.x; // 0..127

#pragma unroll
    for (int ds = 0; ds < K_SLICES_QK; ++ds) {
        int prow, kvec;
        canonical_vec_coord(tid, prow, kvec);

        const int tok = token0 + prow / G;
        const int qh = kvh * G + (prow % G);
        const int d0 = ds * 16 + kvec * 8;

        const __nv_bfloat16* src =
            q + (int64_t(tok) * Hq + qh) * HD + d0;
        __nv_bfloat16* dst =
            s + ds * BLOCK_ELEMS + tid * 8;

        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
}

__device__ __forceinline__ void stage_k64_d128(
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ s,
    int key0,
    int kvh,
    int Hkv
) {
    const int tid = threadIdx.x;

#pragma unroll
    for (int ds = 0; ds < K_SLICES_QK; ++ds) {
        int krow, kvec;
        canonical_vec_coord(tid, krow, kvec);

        const int d0 = ds * 16 + kvec * 8;
        const __nv_bfloat16* src =
            k + (int64_t(key0 + krow) * Hkv + kvh) * HD + d0;
        __nv_bfloat16* dst =
            s + ds * BLOCK_ELEMS + tid * 8;

        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
}

// Stage V as B = V^T for WGMMA P[M,K] @ B[N,K]^T.
// Eight canonical 64x16 blocks:
//   half 0/1 (D 0..63 / 64..127) x key-slice 0..3.
__device__ __forceinline__ void stage_vt64_d128(
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ s,
    int key0,
    int kvh,
    int Hkv
) {
    const int tid = threadIdx.x;

#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int ks = 0; ks < K_SLICES_PV; ++ks) {
            int drow, kvec;
            canonical_vec_coord(tid, drow, kvec);

            __nv_bfloat16* dst =
                s + (half * K_SLICES_PV + ks) * BLOCK_ELEMS + tid * 8;

#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = key0 + ks * 16 + kvec * 8 + e;
                const int d = half * 64 + drow;
                dst[e] = v[(int64_t(key) * Hkv + kvh) * HD + d];
            }
        }
    }
}

__device__ __forceinline__ bool dense_visible(
    int causal, int qtok, int key
) {
    return !causal || key <= qtok;
}

// Dense FULL / CAUSAL D=128 GQA kernel.
// One CTA = exactly one warpgroup = 128 threads.
// One CTA computes 64 packed-Q rows sharing a single KV head.
__global__ __launch_bounds__(128, 1)
void dense_wgmma_fwd(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int G,
    int Ns,
    int causal
) {
    __shared__ __align__(128) __nv_bfloat16 q_s[
        K_SLICES_QK * BLOCK_ELEMS
    ]; // 16 KiB

    // Reused as K (QK phase) then V^T (PV phase), both 16 KiB.
    __shared__ __align__(128) __nv_bfloat16 kv_s[
        2 * K_SLICES_PV * BLOCK_ELEMS
    ];

    // 64x64 BF16 probability tile in canonical 4x(64x16) blocks: 8 KiB.
    __shared__ __align__(128) __nv_bfloat16 p_s[
        K_SLICES_PV * BLOCK_ELEMS
    ];

    const int token_M = M / G;
    const int tile = blockIdx.x;
    const int kvh = blockIdx.y;
    const int token0 = tile * token_M;

    stage_q64_d128(q, q_s, token0, kvh, Hq, G);
    __syncthreads();

    float out0[32];
    float out1[32];
    zero32(out0);
    zero32(out1);

    const FragCoord fc = frag_coord();

    const int prow0 = fc.row0;
    const int prow1 = fc.row1;

    const int qtok0 = token0 + prow0 / G;
    const int qtok1 = token0 + prow1 / G;
    const int qh0 = kvh * G + (prow0 % G);
    const int qh1 = kvh * G + (prow1 % G);

    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0f;
    float l1 = 0.0f;

    int k_blocks = S / N;
    if (causal) {
        const int max_q = token0 + token_M - 1;
        k_blocks = (max_q + 1 + N - 1) / N;
    }

    for (int kb = 0; kb < k_blocks; ++kb) {
        const int key0 = kb * N;

        stage_k64_d128(k, kv_s, key0, kvh, Hkv);
        __syncthreads();

        float score[32];
        zero32(score);

        fence();
#pragma unroll
        for (int ds = 0; ds < K_SLICES_QK; ++ds) {
            const uint64_t q_desc =
                make_kmajor_64x16_desc(q_s + ds * BLOCK_ELEMS);
            const uint64_t k_desc =
                make_kmajor_64x16_desc(kv_s + ds * BLOCK_ELEMS);
            mma_m64n64k16_bf16(score, q_desc, k_desc);
        }
        commit_group();
        wait_group<0>();

        float local_max0 = -CUDART_INF_F;
        float local_max1 = -CUDART_INF_F;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            if (dense_visible(causal, qtok0, key0 + c0)) {
                local_max0 = fmaxf(local_max0, score[4 * g + 0] * softmax_scale);
            }
            if (dense_visible(causal, qtok0, key0 + c1)) {
                local_max0 = fmaxf(local_max0, score[4 * g + 1] * softmax_scale);
            }

            if (dense_visible(causal, qtok1, key0 + c0)) {
                local_max1 = fmaxf(local_max1, score[4 * g + 2] * softmax_scale);
            }
            if (dense_visible(causal, qtok1, key0 + c1)) {
                local_max1 = fmaxf(local_max1, score[4 * g + 3] * softmax_scale);
            }
        }

        const float tile_m0 = row4_max(local_max0);
        const float tile_m1 = row4_max(local_max1);
        const float new_m0 = fmaxf(m0, tile_m0);
        const float new_m1 = fmaxf(m1, tile_m1);

        const float alpha0 =
            (m0 == -CUDART_INF_F) ? 0.0f : __expf(m0 - new_m0);
        const float alpha1 =
            (m1 == -CUDART_INF_F) ? 0.0f : __expf(m1 - new_m1);

        float local_sum0 = 0.0f;
        float local_sum1 = 0.0f;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            float p00 = 0.0f;
            float p01 = 0.0f;
            float p10 = 0.0f;
            float p11 = 0.0f;

            if (dense_visible(causal, qtok0, key0 + c0)) {
                p00 = __expf(score[4 * g + 0] * softmax_scale - new_m0);
                local_sum0 += p00;
            }
            if (dense_visible(causal, qtok0, key0 + c1)) {
                p01 = __expf(score[4 * g + 1] * softmax_scale - new_m0);
                local_sum0 += p01;
            }

            if (dense_visible(causal, qtok1, key0 + c0)) {
                p10 = __expf(score[4 * g + 2] * softmax_scale - new_m1);
                local_sum1 += p10;
            }
            if (dense_visible(causal, qtok1, key0 + c1)) {
                p11 = __expf(score[4 * g + 3] * softmax_scale - new_m1);
                local_sum1 += p11;
            }

            // Reference semantics: BF16 rounding before P@V.
            const int ks0 = c0 >> 4;
            const int kc0 = c0 & 15;
            const int ks1 = c1 >> 4;
            const int kc1 = c1 & 15;

            p_s[ks0 * BLOCK_ELEMS + canonical_kmajor_offset(prow0, kc0)] =
                __float2bfloat16_rn(p00);
            p_s[ks1 * BLOCK_ELEMS + canonical_kmajor_offset(prow0, kc1)] =
                __float2bfloat16_rn(p01);
            p_s[ks0 * BLOCK_ELEMS + canonical_kmajor_offset(prow1, kc0)] =
                __float2bfloat16_rn(p10);
            p_s[ks1 * BLOCK_ELEMS + canonical_kmajor_offset(prow1, kc1)] =
                __float2bfloat16_rn(p11);
        }

        const float tile_l0 = row4_sum(local_sum0);
        const float tile_l1 = row4_sum(local_sum1);

        l0 = l0 * alpha0 + tile_l0;
        l1 = l1 * alpha1 + tile_l1;
        m0 = new_m0;
        m1 = new_m1;

        rescale_fragment(out0, alpha0, alpha1);
        rescale_fragment(out1, alpha0, alpha1);

        // P stores must be visible before WGMMA reads.
        __syncthreads();

        stage_vt64_d128(v, kv_s, key0, kvh, Hkv);
        __syncthreads();

        // P @ V[:, 0:64]
        fence();
#pragma unroll
        for (int ks = 0; ks < K_SLICES_PV; ++ks) {
            const uint64_t p_desc =
                make_kmajor_64x16_desc(p_s + ks * BLOCK_ELEMS);
            const uint64_t v_desc =
                make_kmajor_64x16_desc(
                    kv_s + ks * BLOCK_ELEMS
                );
            mma_m64n64k16_bf16(out0, p_desc, v_desc);
        }
        commit_group();
        wait_group<0>();

        // P @ V[:, 64:128]
        fence();
#pragma unroll
        for (int ks = 0; ks < K_SLICES_PV; ++ks) {
            const uint64_t p_desc =
                make_kmajor_64x16_desc(p_s + ks * BLOCK_ELEMS);
            const uint64_t v_desc =
                make_kmajor_64x16_desc(
                    kv_s + (K_SLICES_PV + ks) * BLOCK_ELEMS
                );
            mma_m64n64k16_bf16(out1, p_desc, v_desc);
        }
        commit_group();
        wait_group<0>();

        __syncthreads();
    }

    float denom0 = l0;
    float denom1 = l1;

#pragma unroll 1
    for (int s = 0; s < Ns; ++s) {
        denom0 += __expf(sink[s * Hq + qh0] - m0);
        denom1 += __expf(sink[s * Hq + qh1] - m1);
    }

    const float inv0 = 1.0f / denom0;
    const float inv1 = 1.0f / denom1;

    const int tok0 = token0 + prow0 / G;
    const int tok1 = token0 + prow1 / G;

#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int c0 = frag_col(g, 0);
        const int c1 = frag_col(g, 1);

        out[(int64_t(tok0) * Hq + qh0) * HD + c0] =
            __float2bfloat16_rn(out0[4 * g + 0] * inv0);
        out[(int64_t(tok0) * Hq + qh0) * HD + c1] =
            __float2bfloat16_rn(out0[4 * g + 1] * inv0);
        out[(int64_t(tok1) * Hq + qh1) * HD + c0] =
            __float2bfloat16_rn(out0[4 * g + 2] * inv1);
        out[(int64_t(tok1) * Hq + qh1) * HD + c1] =
            __float2bfloat16_rn(out0[4 * g + 3] * inv1);

        out[(int64_t(tok0) * Hq + qh0) * HD + 64 + c0] =
            __float2bfloat16_rn(out1[4 * g + 0] * inv0);
        out[(int64_t(tok0) * Hq + qh0) * HD + 64 + c1] =
            __float2bfloat16_rn(out1[4 * g + 1] * inv0);
        out[(int64_t(tok1) * Hq + qh1) * HD + 64 + c0] =
            __float2bfloat16_rn(out1[4 * g + 2] * inv1);
        out[(int64_t(tok1) * Hq + qh1) * HD + 64 + c1] =
            __float2bfloat16_rn(out1[4 * g + 3] * inv1);
    }
}

inline void launch_dense_wgmma(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* sink,
    __nv_bfloat16* out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int Ns,
    int causal
) {
    const int G = Hq / Hkv;
    const int token_M = M / G;
    dim3 grid(S / token_M, Hkv, 1);
    dense_wgmma_fwd<<<grid, 128>>>(
        q, k, v, sink, out, softmax_scale,
        S, Hq, Hkv, G, Ns, causal
    );
}

}  // namespace wgmma_attention
