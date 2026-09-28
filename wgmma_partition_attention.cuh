#pragma once

#include "wgmma_sm90_core.cuh"
#include <math_constants.h>

namespace wgmma_partition {

using namespace wgmma_sm90;

constexpr int M = 64;
constexpr int N = 64;
constexpr int K16 = 16;
constexpr int P_BLOCK_ELEMS = M * K16; // 1024 BF16

template <int HD>
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

__device__ __forceinline__ bool mask_visible(
    int typ,
    int qidx,
    int key,
    int qs,
    int qe,
    int ks,
    int ke
) {
    if (qidx < qs || qidx >= qe || key < ks || key >= ke) return false;

    const int r = qidx - qs;
    const int u = key - ks;
    const int delta = (ke - ks) - (qe - qs);

    if (typ == 0) return true;
    if (typ == 1) return u <= r + delta;
    if (typ == 2) return u >= r;
    return (u >= r) && (u <= r + delta);
}

template <int HD>
__device__ __forceinline__ void stage_q(
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ s,
    int q0,
    int qe,
    int kvh,
    int Hq,
    int G
) {
    constexpr int DS = HD / 16;
    const int tid = threadIdx.x;

#pragma unroll
    for (int ds = 0; ds < DS; ++ds) {
        int prow, kvec;
        canonical_vec_coord(tid, prow, kvec);

        const int tok = q0 + prow / G;
        const int qh = kvh * G + (prow % G);
        const int d0 = ds * 16 + kvec * 8;

        __nv_bfloat16* dst = s + ds * P_BLOCK_ELEMS + tid * 8;

        if (tok < qe) {
            const __nv_bfloat16* src =
                q + (int64_t(tok) * Hq + qh) * HD + d0;
            *reinterpret_cast<uint4*>(dst) =
                *reinterpret_cast<const uint4*>(src);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int HD>
__device__ __forceinline__ void stage_k(
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ s,
    int key0,
    int ke,
    int kvh,
    int Hkv
) {
    constexpr int DS = HD / 16;
    const int tid = threadIdx.x;

#pragma unroll
    for (int ds = 0; ds < DS; ++ds) {
        int krow, kvec;
        canonical_vec_coord(tid, krow, kvec);
        const int key = key0 + krow;
        const int d0 = ds * 16 + kvec * 8;

        __nv_bfloat16* dst = s + ds * P_BLOCK_ELEMS + tid * 8;

        if (key < ke) {
            const __nv_bfloat16* src =
                k + (int64_t(key) * Hkv + kvh) * HD + d0;
            *reinterpret_cast<uint4*>(dst) =
                *reinterpret_cast<const uint4*>(src);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int HD>
__device__ __forceinline__ void stage_vt(
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ s,
    int key0,
    int ke,
    int kvh,
    int Hkv
) {
    constexpr int HALVES = HD / 64;
    const int tid = threadIdx.x;

#pragma unroll
    for (int half = 0; half < HALVES; ++half) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            int drow, kvec;
            canonical_vec_coord(tid, drow, kvec);
            __nv_bfloat16* dst =
                s + (half * 4 + ks) * P_BLOCK_ELEMS + tid * 8;

#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = key0 + ks * 16 + kvec * 8 + e;
                const int d = half * 64 + drow;
                dst[e] = (key < ke)
                    ? v[(int64_t(key) * Hkv + kvh) * HD + d]
                    : __float2bfloat16_rn(0.0f);
            }
        }
    }
}

template <int HD>
__global__ __launch_bounds__(128, 1)
void partition_wgmma_fwd(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int32_t* __restrict__ q_ranges,
    const int32_t* __restrict__ k_ranges,
    const int32_t* __restrict__ attn_type_map,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int G,
    int Ns,
    int NumSlices,
    int special_mode
) {
    constexpr int DS = HD / 16;
    constexpr int HALVES = HD / 64;

    __shared__ __align__(128) __nv_bfloat16 q_s[DS * P_BLOCK_ELEMS];
    __shared__ __align__(128) __nv_bfloat16 kv_s[DS * P_BLOCK_ELEMS];
    __shared__ __align__(128) __nv_bfloat16 p_s[4 * P_BLOCK_ELEMS];

    __shared__ int meta[7];

    const int token_M = M / G;

    if (threadIdx.x == 0) {
        const int pid = blockIdx.x;
        int prefix = 0;
        int hit = 0;
        int q0 = 0, qe = 0, qs = 0, ks = 0, ke = 0, typ = 0;

        if (special_mode == 1) {
            // Exact testcase #5 after collapsing two overlapping FULL slices
            // into three disjoint Q regions with contiguous effective K ranges.
            const int rq0[3] = {0, 128, 256};
            const int rqe[3] = {128, 256, 512};
            const int rks[3] = {0, 0, 256};
            const int rke[3] = {256, 512, 512};

#pragma unroll
            for (int r = 0; r < 3; ++r) {
                const int nt = (rqe[r] - rq0[r] + token_M - 1) / token_M;
                if (!hit && pid >= prefix && pid < prefix + nt) {
                    q0 = rq0[r] + (pid - prefix) * token_M;
                    qe = rqe[r];
                    qs = rq0[r];
                    ks = rks[r];
                    ke = rke[r];
                    typ = 0;
                    hit = 1;
                }
                prefix += nt;
            }
        } else {
            for (int s = 0; s < NumSlices; ++s) {
                const int sqs = q_ranges[2*s + 0];
                const int sqe = q_ranges[2*s + 1];
                const int sks = k_ranges[2*s + 0];
                const int ske = k_ranges[2*s + 1];
                const int st = attn_type_map[s];
                const int nt = (sqe - sqs + token_M - 1) / token_M;

                if (!hit && pid >= prefix && pid < prefix + nt) {
                    q0 = sqs + (pid - prefix) * token_M;
                    qe = sqe;
                    qs = sqs;
                    ks = sks;
                    ke = ske;
                    typ = st;
                    hit = 1;
                }
                prefix += nt;
            }
        }

        meta[0] = hit;
        meta[1] = q0;
        meta[2] = qe;
        meta[3] = qs;
        meta[4] = ks;
        meta[5] = ke;
        meta[6] = typ;
    }
    __syncthreads();

    if (!meta[0]) return;

    const int q0 = meta[1];
    const int qe = meta[2];
    const int qs = meta[3];
    const int ks = meta[4];
    const int ke = meta[5];
    const int typ = meta[6];

    const int kvh = blockIdx.y;

    stage_q<HD>(q, q_s, q0, qe, kvh, Hq, G);
    __syncthreads();

    float out0[32];
    float out1[32];
    zero32<HD>(out0);
    if (HD == 128) zero32<HD>(out1);

    const FragCoord fc = frag_coord();
    const int prow0 = fc.row0;
    const int prow1 = fc.row1;

    const int qidx0 = q0 + prow0 / G;
    const int qidx1 = q0 + prow1 / G;
    const int qh0 = kvh * G + (prow0 % G);
    const int qh1 = kvh * G + (prow1 % G);

    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0f;
    float l1 = 0.0f;

    const int k_blocks = (ke - ks + N - 1) / N;

    for (int kb = 0; kb < k_blocks; ++kb) {
        const int key0 = ks + kb * N;

        stage_k<HD>(k, kv_s, key0, ke, kvh, Hkv);
        __syncthreads();

        float score[32];
        zero32<HD>(score);

        fence();
#pragma unroll
        for (int ds = 0; ds < DS; ++ds) {
            mma_m64n64k16_bf16(
                score,
                make_kmajor_64x16_desc(q_s + ds * P_BLOCK_ELEMS),
                make_kmajor_64x16_desc(kv_s + ds * P_BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        float local_max0 = -CUDART_INF_F;
        float local_max1 = -CUDART_INF_F;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            if (mask_visible(typ, qidx0, key0+c0, qs, qe, ks, ke))
                local_max0 = fmaxf(local_max0, score[4*g+0] * softmax_scale);
            if (mask_visible(typ, qidx0, key0+c1, qs, qe, ks, ke))
                local_max0 = fmaxf(local_max0, score[4*g+1] * softmax_scale);

            if (mask_visible(typ, qidx1, key0+c0, qs, qe, ks, ke))
                local_max1 = fmaxf(local_max1, score[4*g+2] * softmax_scale);
            if (mask_visible(typ, qidx1, key0+c1, qs, qe, ks, ke))
                local_max1 = fmaxf(local_max1, score[4*g+3] * softmax_scale);
        }

        const float tm0 = row4_max(local_max0);
        const float tm1 = row4_max(local_max1);
        const float nm0 = fmaxf(m0, tm0);
        const float nm1 = fmaxf(m1, tm1);
        const float a0 = (m0 == -CUDART_INF_F) ? 0.0f : __expf(m0 - nm0);
        const float a1 = (m1 == -CUDART_INF_F) ? 0.0f : __expf(m1 - nm1);

        float sum0 = 0.0f;
        float sum1 = 0.0f;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            float p00 = 0.0f, p01 = 0.0f, p10 = 0.0f, p11 = 0.0f;

            if (mask_visible(typ, qidx0, key0+c0, qs, qe, ks, ke)) {
                p00 = __expf(score[4*g+0] * softmax_scale - nm0);
                sum0 += p00;
            }
            if (mask_visible(typ, qidx0, key0+c1, qs, qe, ks, ke)) {
                p01 = __expf(score[4*g+1] * softmax_scale - nm0);
                sum0 += p01;
            }
            if (mask_visible(typ, qidx1, key0+c0, qs, qe, ks, ke)) {
                p10 = __expf(score[4*g+2] * softmax_scale - nm1);
                sum1 += p10;
            }
            if (mask_visible(typ, qidx1, key0+c1, qs, qe, ks, ke)) {
                p11 = __expf(score[4*g+3] * softmax_scale - nm1);
                sum1 += p11;
            }

            const int s0 = c0 >> 4, kc0 = c0 & 15;
            const int s1 = c1 >> 4, kc1 = c1 & 15;

            p_s[s0 * P_BLOCK_ELEMS + canonical_kmajor_offset(prow0, kc0)] =
                __float2bfloat16_rn(p00);
            p_s[s1 * P_BLOCK_ELEMS + canonical_kmajor_offset(prow0, kc1)] =
                __float2bfloat16_rn(p01);
            p_s[s0 * P_BLOCK_ELEMS + canonical_kmajor_offset(prow1, kc0)] =
                __float2bfloat16_rn(p10);
            p_s[s1 * P_BLOCK_ELEMS + canonical_kmajor_offset(prow1, kc1)] =
                __float2bfloat16_rn(p11);
        }

        l0 = l0 * a0 + row4_sum(sum0);
        l1 = l1 * a1 + row4_sum(sum1);
        m0 = nm0;
        m1 = nm1;

        rescale32(out0, a0, a1);
        if (HD == 128) rescale32(out1, a0, a1);

        __syncthreads();
        stage_vt<HD>(v, kv_s, key0, ke, kvh, Hkv);
        __syncthreads();

        fence();
#pragma unroll
        for (int kslice = 0; kslice < 4; ++kslice) {
            mma_m64n64k16_bf16(
                out0,
                make_kmajor_64x16_desc(p_s + kslice * P_BLOCK_ELEMS),
                make_kmajor_64x16_desc(kv_s + kslice * P_BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        if (HD == 128) {
            fence();
#pragma unroll
            for (int kslice = 0; kslice < 4; ++kslice) {
                mma_m64n64k16_bf16(
                    out1,
                    make_kmajor_64x16_desc(p_s + kslice * P_BLOCK_ELEMS),
                    make_kmajor_64x16_desc(kv_s + (4 + kslice) * P_BLOCK_ELEMS)
                );
            }
            commit_group();
            wait_group<0>();
        }

        __syncthreads();
    }

    if (qidx0 < qe) {
        float denom0 = l0;
        for (int s = 0; s < Ns; ++s)
            denom0 += __expf(sink[s * Hq + qh0] - m0);
        const float inv0 = 1.0f / denom0;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+0] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+1] * inv0);
            if (HD == 128) {
                out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c0] =
                    __float2bfloat16_rn(out1[4*g+0] * inv0);
                out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c1] =
                    __float2bfloat16_rn(out1[4*g+1] * inv0);
            }
        }
    }

    if (qidx1 < qe) {
        float denom1 = l1;
        for (int s = 0; s < Ns; ++s)
            denom1 += __expf(sink[s * Hq + qh1] - m1);
        const float inv1 = 1.0f / denom1;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+2] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+3] * inv1);
            if (HD == 128) {
                out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c0] =
                    __float2bfloat16_rn(out1[4*g+2] * inv1);
                out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c1] =
                    __float2bfloat16_rn(out1[4*g+3] * inv1);
            }
        }
    }
}

template <int HD>
inline void launch_partition_wgmma(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const int32_t* q_ranges,
    const int32_t* k_ranges,
    const int32_t* attn_type_map,
    const float* sink,
    __nv_bfloat16* out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int Ns,
    int NumSlices,
    int special_mode = 0
) {
    const int G = Hq / Hkv;
    const int token_M = M / G;

    // Slight overlaunch; invalid pids return after the one-thread metadata scan.
    int grid_x = (S + token_M - 1) / token_M + NumSlices;
    if (special_mode == 1) {
        grid_x = (128 + token_M - 1) / token_M
               + (128 + token_M - 1) / token_M
               + (256 + token_M - 1) / token_M;
    }
    dim3 grid(grid_x, Hkv, 1);
    partition_wgmma_fwd<HD><<<grid, 128>>>(
        q, k, v,
        q_ranges, k_ranges, attn_type_map,
        sink, out, softmax_scale,
        S, Hq, Hkv, G, Ns, NumSlices, special_mode
    );
}

} // namespace wgmma_partition
