#pragma once

#include "wgmma_sm90_core.cuh"
#include <math_constants.h>

namespace wgmma_g128 {

using namespace wgmma_sm90;

constexpr int S12 = 8192;
constexpr int HQ12 = 128;
constexpr int Hkv12 = 1;
constexpr int HD12 = 128;
constexpr int NS12 = 4;

constexpr int M = 64;
constexpr int N = 64;
constexpr int BLOCK_ELEMS = 64 * 16; // 1024 BF16
constexpr int QK_SLICES = 8;
constexpr int PV_SLICES = 4;

__device__ __forceinline__ void zero32(float (&x)[32]) {
#pragma unroll
    for (int i = 0; i < 32; ++i) x[i] = 0.0f;
}

__device__ __forceinline__ void rescale32(
    float (&x)[32], float a0, float a1
) {
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        x[4*g+0] *= a0;
        x[4*g+1] *= a0;
        x[4*g+2] *= a1;
        x[4*g+3] *= a1;
    }
}

__global__ __launch_bounds__(256, 1)
void g128_full_wgmma_fwd(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale
) {
    // Two independent Q/P tiles, one per warpgroup.
    __shared__ __align__(128) __nv_bfloat16 q_s[2 * QK_SLICES * BLOCK_ELEMS];
    __shared__ __align__(128) __nv_bfloat16 p_s[2 * PV_SLICES * BLOCK_ELEMS];

    // Shared once by both warpgroups: K then reused as V^T.
    __shared__ __align__(128) __nv_bfloat16 kv_s[QK_SLICES * BLOCK_ELEMS];

    const int wg = threadIdx.x >> 7;      // 0 / 1
    const int wtid = threadIdx.x & 127;   // 0..127 within warpgroup
    const int token = blockIdx.x;
    const int qh_base = wg * 64;

    __nv_bfloat16* my_q = q_s + wg * QK_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* my_p = p_s + wg * PV_SLICES * BLOCK_ELEMS;

    // Each warpgroup stages its own 64 Q heads for this one token.
#pragma unroll
    for (int ds = 0; ds < QK_SLICES; ++ds) {
        int qrow, kvec;
        canonical_vec_coord(wtid, qrow, kvec);
        const int qh = qh_base + qrow;
        const int d0 = ds * 16 + kvec * 8;

        const __nv_bfloat16* src =
            q + (int64_t(token) * HQ12 + qh) * HD12 + d0;
        __nv_bfloat16* dst =
            my_q + ds * BLOCK_ELEMS + wtid * 8;

        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
    __syncthreads();

    float out0[32];
    float out1[32];
    zero32(out0);
    zero32(out1);

    const FragCoord fc = frag_coord();
    const int row0 = fc.row0;
    const int row1 = fc.row1;
    const int qh0 = qh_base + row0;
    const int qh1 = qh_base + row1;

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < S12 / N; ++kb) {
        const int key0 = kb * N;

        // Load K once per CTA, not once per warpgroup.
        if (wg == 0) {
#pragma unroll
            for (int ds = 0; ds < QK_SLICES; ++ds) {
                int krow, kvec;
                canonical_vec_coord(wtid, krow, kvec);
                const int d0 = ds * 16 + kvec * 8;

                const __nv_bfloat16* src =
                    k + int64_t(key0 + krow) * HD12 + d0;
                __nv_bfloat16* dst =
                    kv_s + ds * BLOCK_ELEMS + wtid * 8;

                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(src);
            }
        }
        __syncthreads();

        float score[32];
        zero32(score);

        fence();
#pragma unroll
        for (int ds = 0; ds < QK_SLICES; ++ds) {
            mma_m64n64k16_bf16(
                score,
                make_kmajor_64x16_desc(my_q + ds * BLOCK_ELEMS),
                make_kmajor_64x16_desc(kv_s + ds * BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        float local_max0 = -CUDART_INF_F;
        float local_max1 = -CUDART_INF_F;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            local_max0 = fmaxf(local_max0, score[4*g+0] * softmax_scale);
            local_max0 = fmaxf(local_max0, score[4*g+1] * softmax_scale);
            local_max1 = fmaxf(local_max1, score[4*g+2] * softmax_scale);
            local_max1 = fmaxf(local_max1, score[4*g+3] * softmax_scale);
        }

        const float tm0 = row4_max(local_max0);
        const float tm1 = row4_max(local_max1);
        const float nm0 = fmaxf(m0, tm0);
        const float nm1 = fmaxf(m1, tm1);
        const float a0 = (m0 == -CUDART_INF_F) ? 0.0f : __expf(m0 - nm0);
        const float a1 = (m1 == -CUDART_INF_F) ? 0.0f : __expf(m1 - nm1);

        float sum0 = 0.0f, sum1 = 0.0f;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            const float p00 = __expf(score[4*g+0] * softmax_scale - nm0);
            const float p01 = __expf(score[4*g+1] * softmax_scale - nm0);
            const float p10 = __expf(score[4*g+2] * softmax_scale - nm1);
            const float p11 = __expf(score[4*g+3] * softmax_scale - nm1);

            sum0 += p00 + p01;
            sum1 += p10 + p11;

            const int s0 = c0 >> 4, kc0 = c0 & 15;
            const int s1 = c1 >> 4, kc1 = c1 & 15;

            my_p[s0 * BLOCK_ELEMS + canonical_kmajor_offset(row0, kc0)] =
                __float2bfloat16_rn(p00);
            my_p[s1 * BLOCK_ELEMS + canonical_kmajor_offset(row0, kc1)] =
                __float2bfloat16_rn(p01);
            my_p[s0 * BLOCK_ELEMS + canonical_kmajor_offset(row1, kc0)] =
                __float2bfloat16_rn(p10);
            my_p[s1 * BLOCK_ELEMS + canonical_kmajor_offset(row1, kc1)] =
                __float2bfloat16_rn(p11);
        }

        l0 = l0 * a0 + row4_sum(sum0);
        l1 = l1 * a1 + row4_sum(sum1);
        m0 = nm0;
        m1 = nm1;

        rescale32(out0, a0, a1);
        rescale32(out1, a0, a1);

        __syncthreads();

        // Transpose V once into [D, key] canonical B tiles.
        if (wg == 0) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int ks = 0; ks < PV_SLICES; ++ks) {
                    int drow, kvec;
                    canonical_vec_coord(wtid, drow, kvec);
                    __nv_bfloat16* dst =
                        kv_s + (half * PV_SLICES + ks) * BLOCK_ELEMS
                             + wtid * 8;

#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int key = key0 + ks * 16 + kvec * 8 + e;
                        const int d = half * 64 + drow;
                        dst[e] = v[int64_t(key) * HD12 + d];
                    }
                }
            }
        }
        __syncthreads();

        fence();
#pragma unroll
        for (int ks = 0; ks < PV_SLICES; ++ks) {
            mma_m64n64k16_bf16(
                out0,
                make_kmajor_64x16_desc(my_p + ks * BLOCK_ELEMS),
                make_kmajor_64x16_desc(kv_s + ks * BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        fence();
#pragma unroll
        for (int ks = 0; ks < PV_SLICES; ++ks) {
            mma_m64n64k16_bf16(
                out1,
                make_kmajor_64x16_desc(my_p + ks * BLOCK_ELEMS),
                make_kmajor_64x16_desc(kv_s + (PV_SLICES + ks) * BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        __syncthreads();
    }

    float denom0 = l0;
    float denom1 = l1;
#pragma unroll
    for (int s = 0; s < NS12; ++s) {
        denom0 += __expf(sink[s * HQ12 + qh0] - m0);
        denom1 += __expf(sink[s * HQ12 + qh1] - m1);
    }
    const float inv0 = 1.0f / denom0;
    const float inv1 = 1.0f / denom1;

#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int c0 = frag_col(g, 0);
        const int c1 = frag_col(g, 1);

        out[(int64_t(token) * HQ12 + qh0) * HD12 + c0] =
            __float2bfloat16_rn(out0[4*g+0] * inv0);
        out[(int64_t(token) * HQ12 + qh0) * HD12 + c1] =
            __float2bfloat16_rn(out0[4*g+1] * inv0);
        out[(int64_t(token) * HQ12 + qh1) * HD12 + c0] =
            __float2bfloat16_rn(out0[4*g+2] * inv1);
        out[(int64_t(token) * HQ12 + qh1) * HD12 + c1] =
            __float2bfloat16_rn(out0[4*g+3] * inv1);

        out[(int64_t(token) * HQ12 + qh0) * HD12 + 64+c0] =
            __float2bfloat16_rn(out1[4*g+0] * inv0);
        out[(int64_t(token) * HQ12 + qh0) * HD12 + 64+c1] =
            __float2bfloat16_rn(out1[4*g+1] * inv0);
        out[(int64_t(token) * HQ12 + qh1) * HD12 + 64+c0] =
            __float2bfloat16_rn(out1[4*g+2] * inv1);
        out[(int64_t(token) * HQ12 + qh1) * HD12 + 64+c1] =
            __float2bfloat16_rn(out1[4*g+3] * inv1);
    }
}

inline void launch_g128_wgmma(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* sink,
    __nv_bfloat16* out,
    float softmax_scale
) {
    g128_full_wgmma_fwd<<<S12, 256>>>(
        q, k, v, sink, out, softmax_scale
    );
}

} // namespace wgmma_g128
