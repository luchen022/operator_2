#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math_constants.h>

using namespace nvcuda;

namespace {

constexpr int WARPS = 4;
constexpr int PACKED_M = 64;   // packed Q rows per CTA
constexpr int K_TILE = 32;
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
constexpr int MAX_SLICES = 16;

template <typename Frag, int N>
__device__ __forceinline__ void zero_frags(Frag (&x)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        wmma::fill_fragment(x[i], 0.0f);
    }
}

template <int HDIM>
__device__ __forceinline__ void load_q_packed(
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ sq,
    int packed0,
    int kv_head,
    int Hq,
    int G
) {
    constexpr int chunks_per_row = HDIM / 8;
    constexpr int total_chunks = PACKED_M * chunks_per_row;

    for (int chunk = threadIdx.x; chunk < total_chunks; chunk += blockDim.x) {
        int row = chunk / chunks_per_row;
        int c = chunk - row * chunks_per_row;
        int d0 = c * 8;

        int packed = packed0 + row;
        int tok = packed / G;
        int gh = packed - tok * G;
        int qh = kv_head * G + gh;

        const __nv_bfloat16* src =
            q + (int64_t(tok) * Hq + qh) * HDIM + d0;
        __nv_bfloat16* dst = sq + row * HDIM + d0;

        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
}

template <int HDIM>
__device__ __forceinline__ void load_kv_tile(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ skv,
    int n0,
    int kv_head,
    int Hkv
) {
    constexpr int chunks_per_row = HDIM / 8;
    constexpr int total_chunks = K_TILE * chunks_per_row;

    for (int chunk = threadIdx.x; chunk < total_chunks; chunk += blockDim.x) {
        int row = chunk / chunks_per_row;
        int c = chunk - row * chunks_per_row;
        int d0 = c * 8;

        const __nv_bfloat16* g =
            src + (int64_t(n0 + row) * Hkv + kv_head) * HDIM + d0;
        __nv_bfloat16* s = skv + row * HDIM + d0;

        *reinterpret_cast<uint4*>(s) =
            *reinterpret_cast<const uint4*>(g);
    }
}

__device__ __forceinline__ int visible_count(
    int qtok,
    int key,
    uint32_t active_mask,
    int N,
    const int* __restrict__ q0s,
    const int* __restrict__ q1s,
    const int* __restrict__ k0s,
    const int* __restrict__ k1s,
    const int* __restrict__ types
) {
    int count = 0;

#pragma unroll 1
    for (int s = 0; s < N; ++s) {
        if ((active_mask & (1u << s)) == 0) continue;

        const int ks = k0s[s];
        const int ke = k1s[s];
        if (key < ks || key >= ke) continue;

        const int qs = q0s[s];
        const int qe = q1s[s];
        const int r = qtok - qs;
        const int u = key - ks;
        const int qlen = qe - qs;
        const int klen = ke - ks;
        const int delta = klen - qlen;
        const int typ = types[s];

        bool vis = false;
        if (typ == 0) {
            vis = true;                              // FULL
        } else if (typ == 1) {
            vis = (u <= r + delta);                 // CAUSAL
        } else if (typ == 2) {
            vis = (u >= r);                         // INVCAUSAL
        } else {
            vis = (u >= r) && (u <= r + delta);    // BICAUSAL
        }

        count += vis ? 1 : 0;
    }

    return count;
}

template <int HDIM>
__global__ __launch_bounds__(128, 1)
void generic_ffa_wmma(
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
    int N,
    int Ns
) {
    const int kv_head = blockIdx.y;
    const int G = Hq / Hkv;
    const int packed0 = blockIdx.x * PACKED_M;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_row0 = warp * WM;

    __shared__ __align__(16) __nv_bfloat16 sq[PACKED_M * HDIM];
    __shared__ __align__(16) __nv_bfloat16 skv[K_TILE * HDIM];

    // Each warp owns one independent 16x16 score scratch.
    __shared__ __align__(16) float score[WARPS * WM * WN];

    // Each warp keeps 16x32 normalized probabilities for the current K tile.
    __shared__ __align__(16) __nv_bfloat16 prob[WARPS * WM * K_TILE];

    __shared__ int s_q0[MAX_SLICES];
    __shared__ int s_q1[MAX_SLICES];
    __shared__ int s_k0[MAX_SLICES];
    __shared__ int s_k1[MAX_SLICES];
    __shared__ int s_typ[MAX_SLICES];

    if (threadIdx.x < N) {
        const int s = threadIdx.x;
        s_q0[s] = q_ranges[s * 2 + 0];
        s_q1[s] = q_ranges[s * 2 + 1];
        s_k0[s] = k_ranges[s * 2 + 0];
        s_k1[s] = k_ranges[s * 2 + 1];
        s_typ[s] = attn_type_map[s];
    }

    load_q_packed<HDIM>(q, sq, packed0, kv_head, Hq, G);
    __syncthreads();

    using AFrag = wmma::fragment<
        wmma::matrix_a, WM, WN, WK,
        __nv_bfloat16, wmma::row_major>;
    using BColFrag = wmma::fragment<
        wmma::matrix_b, WM, WN, WK,
        __nv_bfloat16, wmma::col_major>;
    using BRowFrag = wmma::fragment<
        wmma::matrix_b, WM, WN, WK,
        __nv_bfloat16, wmma::row_major>;
    using CFrag = wmma::fragment<
        wmma::accumulator, WM, WN, WK, float>;

    float* ws = score + warp * WM * WN;
    __nv_bfloat16* wp = prob + warp * WM * K_TILE;

    // Row identity for the 16 lanes that own softmax rows.
    int row_tok = 0;
    int row_qh = 0;
    uint32_t active_mask = 0;

    if (lane < WM) {
        const int packed = packed0 + warp_row0 + lane;
        row_tok = packed / G;
        const int gh = packed - row_tok * G;
        row_qh = kv_head * G + gh;

#pragma unroll 1
        for (int s = 0; s < N; ++s) {
            if (row_tok >= s_q0[s] && row_tok < s_q1[s]) {
                active_mask |= (1u << s);
            }
        }
    }

    float row_m = -CUDART_INF_F;
    float row_l = 0.0f;

    // ------------------------------------------------------------------
    // Pass 1: QK -> exact slice/mask interpretation -> max + sumexp.
    // We scan the global K axis once; overlapping slices are represented
    // by multiplicity in the softmax mass.
    // ------------------------------------------------------------------
    for (int n0 = 0; n0 < S; n0 += K_TILE) {
        load_kv_tile<HDIM>(k, skv, n0, kv_head, Hkv);
        __syncthreads();

#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            CFrag c;
            wmma::fill_fragment(c, 0.0f);

#pragma unroll
            for (int kk0 = 0; kk0 < HDIM; kk0 += WK) {
                AFrag af;
                BColFrag bf;

                wmma::load_matrix_sync(
                    af,
                    sq + warp_row0 * HDIM + kk0,
                    HDIM
                );
                wmma::load_matrix_sync(
                    bf,
                    skv + nt * WN * HDIM + kk0,
                    HDIM
                );
                wmma::mma_sync(c, af, bf, c);
            }

            wmma::store_matrix_sync(ws, c, WN, wmma::mem_row_major);
            __syncwarp();

            if (lane < WM) {
                const int key0 = n0 + nt * WN;
                float local_max = -CUDART_INF_F;

#pragma unroll
                for (int j = 0; j < WN; ++j) {
                    const int key = key0 + j;
                    const int mult = visible_count(
                        row_tok, key, active_mask, N,
                        s_q0, s_q1, s_k0, s_k1, s_typ
                    );

                    if (mult > 0) {
                        local_max = fmaxf(
                            local_max,
                            ws[lane * WN + j] * softmax_scale
                        );
                    }
                }

                const float m_new = fmaxf(row_m, local_max);
                float tile_sum = 0.0f;

#pragma unroll
                for (int j = 0; j < WN; ++j) {
                    const int key = key0 + j;
                    const int mult = visible_count(
                        row_tok, key, active_mask, N,
                        s_q0, s_q1, s_k0, s_k1, s_typ
                    );

                    if (mult > 0) {
                        tile_sum += float(mult) * __expf(
                            ws[lane * WN + j] * softmax_scale - m_new
                        );
                    }
                }

                const float alpha =
                    (row_m == -CUDART_INF_F) ? 0.0f : __expf(row_m - m_new);

                row_l = row_l * alpha + tile_sum;
                row_m = m_new;
            }

            __syncwarp();
        }

        __syncthreads();
    }

    // Attention Sink contributes to the denominator only.
    float norm_m = 0.0f;
    float inv_denom = 0.0f;

    if (lane < WM) {
        float sink_max = -CUDART_INF_F;

#pragma unroll 1
        for (int s = 0; s < Ns; ++s) {
            sink_max = fmaxf(sink_max, sink[s * Hq + row_qh]);
        }

        norm_m = fmaxf(row_m, sink_max);
        float denom = row_l * __expf(row_m - norm_m);

#pragma unroll 1
        for (int s = 0; s < Ns; ++s) {
            denom += __expf(sink[s * Hq + row_qh] - norm_m);
        }

        inv_denom = 1.0f / denom;
    }

    // ------------------------------------------------------------------
    // Pass 2: recompute QK, normalize, round P to BF16, then P@V on WMMA.
    // ------------------------------------------------------------------
    constexpr int OUT_TILES = HDIM / WN;
    CFrag out_frag[OUT_TILES];
    zero_frags(out_frag);

    for (int n0 = 0; n0 < S; n0 += K_TILE) {
        load_kv_tile<HDIM>(k, skv, n0, kv_head, Hkv);
        __syncthreads();

#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            CFrag c;
            wmma::fill_fragment(c, 0.0f);

#pragma unroll
            for (int kk0 = 0; kk0 < HDIM; kk0 += WK) {
                AFrag af;
                BColFrag bf;

                wmma::load_matrix_sync(
                    af,
                    sq + warp_row0 * HDIM + kk0,
                    HDIM
                );
                wmma::load_matrix_sync(
                    bf,
                    skv + nt * WN * HDIM + kk0,
                    HDIM
                );
                wmma::mma_sync(c, af, bf, c);
            }

            wmma::store_matrix_sync(ws, c, WN, wmma::mem_row_major);
            __syncwarp();

            if (lane < WM) {
                const int key0 = n0 + nt * WN;
                __nv_bfloat16* prow = wp + lane * K_TILE + nt * WN;

#pragma unroll
                for (int j = 0; j < WN; ++j) {
                    const int key = key0 + j;
                    const int mult = visible_count(
                        row_tok, key, active_mask, N,
                        s_q0, s_q1, s_k0, s_k1, s_typ
                    );

                    if (mult == 0) {
                        prow[j] = __float2bfloat16_rn(0.0f);
                    } else {
                        const float p = __expf(
                            ws[lane * WN + j] * softmax_scale - norm_m
                        ) * inv_denom;

                        // Reference semantics round each probability to BF16
                        // before P@V. If a key appears through two overlapping
                        // slices, duplicate BF16 probabilities sum linearly.
                        const __nv_bfloat16 pb = __float2bfloat16_rn(p);
                        const float agg =
                            float(mult) * __bfloat162float(pb);
                        prow[j] = __float2bfloat16_rn(agg);
                    }
                }
            }

            __syncwarp();
        }

        __syncthreads();

        // Reuse K shared storage for V after all probabilities are materialized.
        load_kv_tile<HDIM>(v, skv, n0, kv_head, Hkv);
        __syncthreads();

#pragma unroll
        for (int pk = 0; pk < K_TILE; pk += WK) {
            AFrag pf;
            wmma::load_matrix_sync(pf, wp + pk, K_TILE);

#pragma unroll
            for (int dt = 0; dt < OUT_TILES; ++dt) {
                BRowFrag vf;
                wmma::load_matrix_sync(
                    vf,
                    skv + pk * HDIM + dt * WN,
                    HDIM
                );
                wmma::mma_sync(
                    out_frag[dt],
                    pf,
                    vf,
                    out_frag[dt]
                );
            }
        }

        __syncthreads();
    }

    // Scatter packed rows back to output [S,Hq,D].
#pragma unroll
    for (int dt = 0; dt < OUT_TILES; ++dt) {
        wmma::store_matrix_sync(ws, out_frag[dt], WN, wmma::mem_row_major);
        __syncwarp();

        for (int t = lane; t < WM * WN; t += 32) {
            const int r = t >> 4;
            const int c = t & 15;
            const int packed = packed0 + warp_row0 + r;
            const int tok = packed / G;
            const int gh = packed - tok * G;
            const int qh = kv_head * G + gh;
            const int d = dt * WN + c;

            out[(int64_t(tok) * Hq + qh) * HDIM + d] =
                __float2bfloat16_rn(ws[t]);
        }

        __syncwarp();
    }
}


template <int SEQ, int HQ, int HKV, int G, int NSINK, bool CAUSAL, int SEGMENT_MODE = 0>
__global__ __launch_bounds__(256, 1)
void dense_packgqa_online_wmma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale
) {
    constexpr int PM = 128;
    constexpr int KN = 64;
    constexpr int W = 8;
    constexpr int HD = 128;
    constexpr int TOKEN_M = PM / G;
    constexpr int SCORE_ELEMS_PER_WARP = 16 * KN;
    constexpr int PROB_ELEMS_PER_WARP = 16 * KN;
    constexpr int OUT_ELEMS_PER_WARP = 16 * HD;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* kv_s = q_s + PM * HD;
    float* score_s = reinterpret_cast<float*>(kv_s + KN * HD);
    __nv_bfloat16* prob_s =
        reinterpret_cast<__nv_bfloat16*>(score_s + W * SCORE_ELEMS_PER_WARP);
    float* out_s =
        reinterpret_cast<float*>(prob_s + W * PROB_ELEMS_PER_WARP);
    float* alpha_s = out_s + W * OUT_ELEMS_PER_WARP;

    const int kvh = blockIdx.y;
    const int tile = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_row0 = warp * 16;
    const int token0 = tile * TOKEN_M;

    float* ws = score_s + warp * SCORE_ELEMS_PER_WARP;
    __nv_bfloat16* wp = prob_s + warp * PROB_ELEMS_PER_WARP;
    float* wo = out_s + warp * OUT_ELEMS_PER_WARP;

    // Pack G query heads that share one KV head into a 128-row matrix.
    for (int chunk = threadIdx.x; chunk < (PM * HD) / 8; chunk += blockDim.x) {
        const int p = chunk / (HD / 8);
        const int c = chunk - p * (HD / 8);
        const int d0 = c * 8;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        const int tok = token0 + tok_local;
        const int qh = kvh * G + gh;

        const __nv_bfloat16* gptr =
            q + (int64_t(tok) * HQ + qh) * HD + d0;
        __nv_bfloat16* sptr = q_s + p * HD + d0;
        *reinterpret_cast<uint4*>(sptr) =
            *reinterpret_cast<const uint4*>(gptr);
    }

    for (int i = threadIdx.x; i < W * OUT_ELEMS_PER_WARP; i += blockDim.x) {
        out_s[i] = 0.0f;
    }
    __syncthreads();

    using AFrag = wmma::fragment<
        wmma::matrix_a, 16, 16, 16,
        __nv_bfloat16, wmma::row_major>;
    using BColFrag = wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        __nv_bfloat16, wmma::col_major>;
    using BRowFrag = wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        __nv_bfloat16, wmma::row_major>;
    using CFrag = wmma::fragment<
        wmma::accumulator, 16, 16, 16, float>;

    float row_m = -CUDART_INF_F;
    float row_l = 0.0f;
    int row_tok = 0;
    int row_qh = 0;

    if (lane < 16) {
        const int p = warp_row0 + lane;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        row_tok = token0 + tok_local;
        row_qh = kvh * G + gh;
    }

    int k_start = 0;
    int k_len = SEQ;

    // Exact testcase #5: two overlapping FULL slices collapse into three
    // fixed Q regions with one contiguous effective K interval each.
    if (SEGMENT_MODE == 1) {
        if (token0 < 128) {
            k_start = 0;
            k_len = 256;
        } else if (token0 < 256) {
            k_start = 0;
            k_len = 512;
        } else {
            k_start = 256;
            k_len = 256;
        }
    }

    int k_tiles = (k_len + KN - 1) / KN;
    if (CAUSAL) {
        const int max_q = token0 + TOKEN_M - 1;
        k_tiles = (max_q + 1 + KN - 1) / KN;
    }

    for (int kb = 0; kb < k_tiles; ++kb) {
        const int n0 = k_start + kb * KN;

        // Cooperative 16B K load.
        for (int chunk = threadIdx.x; chunk < (KN * HD) / 8; chunk += blockDim.x) {
            const int r = chunk / (HD / 8);
            const int c = chunk - r * (HD / 8);
            const int d0 = c * 8;
            const __nv_bfloat16* gptr =
                k + (int64_t(n0 + r) * HKV + kvh) * HD + d0;
            __nv_bfloat16* sptr = kv_s + r * HD + d0;
            *reinterpret_cast<uint4*>(sptr) =
                *reinterpret_cast<const uint4*>(gptr);
        }
        __syncthreads();

        // 16 packed Q rows x 64 K rows per warp.
        CFrag qk_frag[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            wmma::fill_fragment(qk_frag[nt], 0.0f);
        }

#pragma unroll
        for (int kk0 = 0; kk0 < HD; kk0 += 16) {
            AFrag af;
            wmma::load_matrix_sync(
                af,
                q_s + warp_row0 * HD + kk0,
                HD
            );

#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                BColFrag bf;
                wmma::load_matrix_sync(
                    bf,
                    kv_s + nt * 16 * HD + kk0,
                    HD
                );
                wmma::mma_sync(qk_frag[nt], af, bf, qk_frag[nt]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            wmma::store_matrix_sync(
                ws + nt * 16,
                qk_frag[nt],
                KN,
                wmma::mem_row_major
            );
        }
        __syncwarp();

        if (lane < 16) {
            const bool frontier = CAUSAL && (n0 + KN - 1 > token0);
            float tile_m = -CUDART_INF_F;

#pragma unroll
            for (int j = 0; j < KN; ++j) {
                const int key = n0 + j;
                if (!frontier || key <= row_tok) {
                    tile_m = fmaxf(tile_m, ws[lane * KN + j] * softmax_scale);
                }
            }

            const float m_new = fmaxf(row_m, tile_m);
            const float alpha =
                (row_m == -CUDART_INF_F) ? 0.0f : __expf(row_m - m_new);
            float tile_l = 0.0f;

#pragma unroll
            for (int j = 0; j < KN; ++j) {
                const int key = n0 + j;
                float p = 0.0f;
                if (!frontier || key <= row_tok) {
                    p = __expf(ws[lane * KN + j] * softmax_scale - m_new);
                    tile_l += p;
                }
                wp[lane * KN + j] = __float2bfloat16_rn(p);
            }

            alpha_s[warp * 16 + lane] = alpha;
            row_l = row_l * alpha + tile_l;
            row_m = m_new;
        }
        __syncwarp();

        // Rescale the accumulated numerator to the new online-softmax max.
        for (int t = lane; t < OUT_ELEMS_PER_WARP; t += 32) {
            const int r = t / HD;
            wo[t] *= alpha_s[warp * 16 + r];
        }
        __syncwarp();

        // K is dead; reuse the same shared tile for V.
        __syncthreads();
        for (int chunk = threadIdx.x; chunk < (KN * HD) / 8; chunk += blockDim.x) {
            const int r = chunk / (HD / 8);
            const int c = chunk - r * (HD / 8);
            const int d0 = c * 8;
            const __nv_bfloat16* gptr =
                v + (int64_t(n0 + r) * HKV + kvh) * HD + d0;
            __nv_bfloat16* sptr = kv_s + r * HD + d0;
            *reinterpret_cast<uint4*>(sptr) =
                *reinterpret_cast<const uint4*>(gptr);
        }
        __syncthreads();

        CFrag out_frag[8];
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            wmma::load_matrix_sync(
                out_frag[dt],
                wo + dt * 16,
                HD,
                wmma::mem_row_major
            );
        }

#pragma unroll
        for (int pk = 0; pk < KN; pk += 16) {
            AFrag pf;
            wmma::load_matrix_sync(pf, wp + pk, KN);

#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                BRowFrag vf;
                wmma::load_matrix_sync(
                    vf,
                    kv_s + pk * HD + dt * 16,
                    HD
                );
                wmma::mma_sync(
                    out_frag[dt],
                    pf,
                    vf,
                    out_frag[dt]
                );
            }
        }

#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            wmma::store_matrix_sync(
                wo + dt * 16,
                out_frag[dt],
                HD,
                wmma::mem_row_major
            );
        }
        __syncthreads();
    }

    // Fold Attention Sink into the denominator and emit BF16 output.
    if (lane < 16) {
        float denom = row_l;
#pragma unroll
        for (int s = 0; s < NSINK; ++s) {
            denom += __expf(sink[s * HQ + row_qh] - row_m);
        }
        alpha_s[warp * 16 + lane] = 1.0f / denom;
    }
    __syncwarp();

    for (int t = lane; t < OUT_ELEMS_PER_WARP; t += 32) {
        const int r = t / HD;
        const int d = t - r * HD;
        const int p = warp_row0 + r;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        const int tok = token0 + tok_local;
        const int qh = kvh * G + gh;

        out[(int64_t(tok) * HQ + qh) * HD + d] =
            __float2bfloat16_rn(wo[t] * alpha_s[warp * 16 + r]);
    }
}


template <int SEQ, int HQ, int HKV, int G, int NSINK, bool CAUSAL, int SEGMENT_MODE = 0>
void launch_dense_packgqa_online(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* sink,
    __nv_bfloat16* out,
    float softmax_scale
) {
    constexpr int PM = 128;
    constexpr int KN = 64;
    constexpr int HD = 128;
    constexpr int W = 8;
    constexpr int TOKEN_M = PM / G;

    constexpr size_t smem_bytes =
        PM * HD * sizeof(__nv_bfloat16) +
        KN * HD * sizeof(__nv_bfloat16) +
        W * 16 * KN * sizeof(float) +
        W * 16 * KN * sizeof(__nv_bfloat16) +
        W * 16 * HD * sizeof(float) +
        W * 16 * sizeof(float);

    cudaFuncSetAttribute(
        dense_packgqa_online_wmma<SEQ, HQ, HKV, G, NSINK, CAUSAL, SEGMENT_MODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes)
    );

    dim3 grid((SEQ + TOKEN_M - 1) / TOKEN_M, HKV, 1);
    dim3 block(256, 1, 1);
    dense_packgqa_online_wmma<SEQ, HQ, HKV, G, NSINK, CAUSAL, SEGMENT_MODE>
        <<<grid, block, smem_bytes>>>(
            q, k, v, sink, out, softmax_scale
        );
}


template <int SEQ, int HQ, int HKV, int G, int HD, int NSLICES, int NSINK>
__global__ __launch_bounds__(256, 1)
void partition_packgqa_online_wmma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int32_t* __restrict__ q_ranges,
    const int32_t* __restrict__ k_ranges,
    const int32_t* __restrict__ attn_type_map,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale
) {
    constexpr int PM = 128;
    constexpr int KN = 64;
    constexpr int W = 8;
    constexpr int TOKEN_M = PM / G;
    constexpr int SCORE_ELEMS_PER_WARP = 16 * KN;
    constexpr int PROB_ELEMS_PER_WARP = 16 * KN;
    constexpr int OUT_ELEMS_PER_WARP = 16 * HD;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* kv_s = q_s + PM * HD;
    float* score_s = reinterpret_cast<float*>(kv_s + KN * HD);
    __nv_bfloat16* prob_s =
        reinterpret_cast<__nv_bfloat16*>(score_s + W * SCORE_ELEMS_PER_WARP);
    float* out_s =
        reinterpret_cast<float*>(prob_s + W * PROB_ELEMS_PER_WARP);
    float* alpha_s = out_s + W * OUT_ELEMS_PER_WARP;

    __shared__ int meta[7];
    // meta: valid, q0, qe, qs, ks, ke, type
    if (threadIdx.x == 0) {
        const int pid = blockIdx.x;
        int prefix = 0;
        int hit = 0;
        int q0_sel = 0, qe_sel = 0, qs_sel = 0;
        int ks_sel = 0, ke_sel = 0, typ_sel = 0;

#pragma unroll
        for (int s = 0; s < NSLICES; ++s) {
            const int qs = q_ranges[s * 2 + 0];
            const int qe = q_ranges[s * 2 + 1];
            const int ks = k_ranges[s * 2 + 0];
            const int ke = k_ranges[s * 2 + 1];
            const int typ = attn_type_map[s];
            const int qlen = qe - qs;
            const int nt = (qlen + TOKEN_M - 1) / TOKEN_M;

            if (!hit && pid >= prefix && pid < prefix + nt) {
                const int local = pid - prefix;
                q0_sel = qs + local * TOKEN_M;
                qe_sel = qe;
                qs_sel = qs;
                ks_sel = ks;
                ke_sel = ke;
                typ_sel = typ;
                hit = 1;
            }
            prefix += nt;
        }

        meta[0] = hit;
        meta[1] = q0_sel;
        meta[2] = qe_sel;
        meta[3] = qs_sel;
        meta[4] = ks_sel;
        meta[5] = ke_sel;
        meta[6] = typ_sel;
    }
    __syncthreads();

    if (!meta[0]) return;

    const int q0 = meta[1];
    const int qe = meta[2];
    const int qs = meta[3];
    const int ks = meta[4];
    const int ke = meta[5];
    const int typ = meta[6];
    const int qlen = qe - qs;
    const int klen = ke - ks;
    const int delta = klen - qlen;

    const int kvh = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_row0 = warp * 16;

    float* ws = score_s + warp * SCORE_ELEMS_PER_WARP;
    __nv_bfloat16* wp = prob_s + warp * PROB_ELEMS_PER_WARP;
    float* wo = out_s + warp * OUT_ELEMS_PER_WARP;

    constexpr int q_chunks_per_row = HD / 8;
    for (int chunk = threadIdx.x; chunk < PM * q_chunks_per_row; chunk += blockDim.x) {
        const int p = chunk / q_chunks_per_row;
        const int c = chunk - p * q_chunks_per_row;
        const int d0 = c * 8;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        const int qidx = q0 + tok_local;
        const int qh = kvh * G + gh;
        __nv_bfloat16* sptr = q_s + p * HD + d0;

        if (qidx < qe) {
            const __nv_bfloat16* gptr =
                q + (int64_t(qidx) * HQ + qh) * HD + d0;
            *reinterpret_cast<uint4*>(sptr) =
                *reinterpret_cast<const uint4*>(gptr);
        } else {
            *reinterpret_cast<uint4*>(sptr) = make_uint4(0, 0, 0, 0);
        }
    }

    for (int i = threadIdx.x; i < W * OUT_ELEMS_PER_WARP; i += blockDim.x) {
        out_s[i] = 0.0f;
    }
    __syncthreads();

    using AFrag = wmma::fragment<
        wmma::matrix_a, 16, 16, 16,
        __nv_bfloat16, wmma::row_major>;
    using BColFrag = wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        __nv_bfloat16, wmma::col_major>;
    using BRowFrag = wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        __nv_bfloat16, wmma::row_major>;
    using CFrag = wmma::fragment<
        wmma::accumulator, 16, 16, 16, float>;

    float row_m = -CUDART_INF_F;
    float row_l = 0.0f;
    int row_qidx = 0;
    int row_qh = 0;
    int row_r = 0;

    if (lane < 16) {
        const int p = warp_row0 + lane;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        row_qidx = q0 + tok_local;
        row_qh = kvh * G + gh;
        row_r = row_qidx - qs;
    }

    const int k_tiles = (klen + KN - 1) / KN;

    for (int kb = 0; kb < k_tiles; ++kb) {
        const int n0 = ks + kb * KN;

        constexpr int kv_chunks_per_row = HD / 8;
        for (int chunk = threadIdx.x; chunk < KN * kv_chunks_per_row; chunk += blockDim.x) {
            const int r = chunk / kv_chunks_per_row;
            const int c = chunk - r * kv_chunks_per_row;
            const int d0 = c * 8;
            const int kidx = n0 + r;
            __nv_bfloat16* sptr = kv_s + r * HD + d0;

            if (kidx < ke) {
                const __nv_bfloat16* gptr =
                    k + (int64_t(kidx) * HKV + kvh) * HD + d0;
                *reinterpret_cast<uint4*>(sptr) =
                    *reinterpret_cast<const uint4*>(gptr);
            } else {
                *reinterpret_cast<uint4*>(sptr) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        CFrag qk_frag[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            wmma::fill_fragment(qk_frag[nt], 0.0f);
        }

#pragma unroll
        for (int kk0 = 0; kk0 < HD; kk0 += 16) {
            AFrag af;
            wmma::load_matrix_sync(
                af,
                q_s + warp_row0 * HD + kk0,
                HD
            );

#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                BColFrag bf;
                wmma::load_matrix_sync(
                    bf,
                    kv_s + nt * 16 * HD + kk0,
                    HD
                );
                wmma::mma_sync(qk_frag[nt], af, bf, qk_frag[nt]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            wmma::store_matrix_sync(
                ws + nt * 16,
                qk_frag[nt],
                KN,
                wmma::mem_row_major
            );
        }
        __syncwarp();

        if (lane < 16) {
            float tile_m = -CUDART_INF_F;

#pragma unroll
            for (int j = 0; j < KN; ++j) {
                const int kidx = n0 + j;
                const int u = kidx - ks;
                bool visible = (row_qidx < qe) && (kidx < ke);

                if (typ == 1) {
                    visible = visible && (u <= row_r + delta);
                } else if (typ == 2) {
                    visible = visible && (u >= row_r);
                } else if (typ == 3) {
                    visible = visible && (u >= row_r) && (u <= row_r + delta);
                }

                if (visible) {
                    tile_m = fmaxf(tile_m, ws[lane * KN + j] * softmax_scale);
                }
            }

            const float m_new = fmaxf(row_m, tile_m);
            const float alpha =
                (row_m == -CUDART_INF_F) ? 0.0f : __expf(row_m - m_new);
            float tile_l = 0.0f;

#pragma unroll
            for (int j = 0; j < KN; ++j) {
                const int kidx = n0 + j;
                const int u = kidx - ks;
                bool visible = (row_qidx < qe) && (kidx < ke);

                if (typ == 1) {
                    visible = visible && (u <= row_r + delta);
                } else if (typ == 2) {
                    visible = visible && (u >= row_r);
                } else if (typ == 3) {
                    visible = visible && (u >= row_r) && (u <= row_r + delta);
                }

                float p = 0.0f;
                if (visible) {
                    p = __expf(ws[lane * KN + j] * softmax_scale - m_new);
                    tile_l += p;
                }
                wp[lane * KN + j] = __float2bfloat16_rn(p);
            }

            alpha_s[warp * 16 + lane] = alpha;
            row_l = row_l * alpha + tile_l;
            row_m = m_new;
        }
        __syncwarp();

        for (int t = lane; t < OUT_ELEMS_PER_WARP; t += 32) {
            const int r = t / HD;
            wo[t] *= alpha_s[warp * 16 + r];
        }
        __syncwarp();

        __syncthreads();
        for (int chunk = threadIdx.x; chunk < KN * kv_chunks_per_row; chunk += blockDim.x) {
            const int r = chunk / kv_chunks_per_row;
            const int c = chunk - r * kv_chunks_per_row;
            const int d0 = c * 8;
            const int kidx = n0 + r;
            __nv_bfloat16* sptr = kv_s + r * HD + d0;

            if (kidx < ke) {
                const __nv_bfloat16* gptr =
                    v + (int64_t(kidx) * HKV + kvh) * HD + d0;
                *reinterpret_cast<uint4*>(sptr) =
                    *reinterpret_cast<const uint4*>(gptr);
            } else {
                *reinterpret_cast<uint4*>(sptr) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        constexpr int OUT_TILES = HD / 16;
        CFrag out_frag[OUT_TILES];
#pragma unroll
        for (int dt = 0; dt < OUT_TILES; ++dt) {
            wmma::load_matrix_sync(
                out_frag[dt],
                wo + dt * 16,
                HD,
                wmma::mem_row_major
            );
        }

#pragma unroll
        for (int pk = 0; pk < KN; pk += 16) {
            AFrag pf;
            wmma::load_matrix_sync(pf, wp + pk, KN);

#pragma unroll
            for (int dt = 0; dt < OUT_TILES; ++dt) {
                BRowFrag vf;
                wmma::load_matrix_sync(
                    vf,
                    kv_s + pk * HD + dt * 16,
                    HD
                );
                wmma::mma_sync(
                    out_frag[dt],
                    pf,
                    vf,
                    out_frag[dt]
                );
            }
        }

#pragma unroll
        for (int dt = 0; dt < OUT_TILES; ++dt) {
            wmma::store_matrix_sync(
                wo + dt * 16,
                out_frag[dt],
                HD,
                wmma::mem_row_major
            );
        }
        __syncthreads();
    }

    if (lane < 16) {
        float denom = row_l;
#pragma unroll
        for (int s = 0; s < NSINK; ++s) {
            denom += __expf(sink[s * HQ + row_qh] - row_m);
        }
        alpha_s[warp * 16 + lane] = 1.0f / denom;
    }
    __syncwarp();

    for (int t = lane; t < OUT_ELEMS_PER_WARP; t += 32) {
        const int r = t / HD;
        const int d = t - r * HD;
        const int p = warp_row0 + r;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        const int qidx = q0 + tok_local;
        const int qh = kvh * G + gh;

        if (qidx < qe) {
            out[(int64_t(qidx) * HQ + qh) * HD + d] =
                __float2bfloat16_rn(wo[t] * alpha_s[warp * 16 + r]);
        }
    }
}


template <int SEQ, int HQ, int HKV, int G, int HD, int NSLICES, int NSINK>
void launch_partition_packgqa_online(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const int32_t* q_ranges,
    const int32_t* k_ranges,
    const int32_t* attn_type_map,
    const float* sink,
    __nv_bfloat16* out,
    float softmax_scale
) {
    constexpr int PM = 128;
    constexpr int KN = 64;
    constexpr int W = 8;
    constexpr int TOKEN_M = PM / G;

    constexpr size_t smem_bytes =
        PM * HD * sizeof(__nv_bfloat16) +
        KN * HD * sizeof(__nv_bfloat16) +
        W * 16 * KN * sizeof(float) +
        W * 16 * KN * sizeof(__nv_bfloat16) +
        W * 16 * HD * sizeof(float) +
        W * 16 * sizeof(float);

    cudaFuncSetAttribute(
        partition_packgqa_online_wmma<
            SEQ, HQ, HKV, G, HD, NSLICES, NSINK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes)
    );

    dim3 grid((SEQ + TOKEN_M - 1) / TOKEN_M + NSLICES, HKV, 1);
    dim3 block(256, 1, 1);

    partition_packgqa_online_wmma<
        SEQ, HQ, HKV, G, HD, NSLICES, NSINK>
        <<<grid, block, smem_bytes>>>(
            q, k, v,
            q_ranges, k_ranges, attn_type_map,
            sink, out, softmax_scale
        );
}


__global__ __launch_bounds__(256, 1)
void overlap8_g4_online_wmma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int32_t* __restrict__ q_ranges,
    const int32_t* __restrict__ k_ranges,
    const int32_t* __restrict__ attn_type_map,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale
) {
    constexpr int SEQ = 4096;
    constexpr int HQ = 32;
    constexpr int HKV = 8;
    constexpr int G = 4;
    constexpr int HD = 128;
    constexpr int NSINK = 6;
    constexpr int PM = 128;
    constexpr int KN = 64;
    constexpr int W = 8;
    constexpr int TOKEN_M = PM / G;
    constexpr int SCORE_ELEMS_PER_WARP = 16 * KN;
    constexpr int PROB_ELEMS_PER_WARP = 16 * KN;
    constexpr int OUT_ELEMS_PER_WARP = 16 * HD;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* kv_s = q_s + PM * HD;
    float* score_s = reinterpret_cast<float*>(kv_s + KN * HD);
    __nv_bfloat16* prob_s =
        reinterpret_cast<__nv_bfloat16*>(score_s + W * SCORE_ELEMS_PER_WARP);
    float* out_s =
        reinterpret_cast<float*>(prob_s + W * PROB_ELEMS_PER_WARP);
    float* alpha_s = out_s + W * OUT_ELEMS_PER_WARP;

    __shared__ int meta[12];
    // Base slice: valid,q0,qe,qs,ks,ke,type. Extra: qs,qe,ks,ke,type.
    if (threadIdx.x == 0) {
        const int pid = blockIdx.x;
        int prefix = 0;
        int hit = 0;
        int q0_sel = 0, qe_sel = 0, qs_sel = 0;
        int ks_sel = 0, ke_sel = 0, typ_sel = 0;

#pragma unroll
        for (int s = 0; s < 7; ++s) {
            const int qs = q_ranges[s * 2 + 0];
            const int qe = q_ranges[s * 2 + 1];
            const int ks = k_ranges[s * 2 + 0];
            const int ke = k_ranges[s * 2 + 1];
            const int typ = attn_type_map[s];
            const int nt = (qe - qs + TOKEN_M - 1) / TOKEN_M;

            if (!hit && pid >= prefix && pid < prefix + nt) {
                q0_sel = qs + (pid - prefix) * TOKEN_M;
                qe_sel = qe;
                qs_sel = qs;
                ks_sel = ks;
                ke_sel = ke;
                typ_sel = typ;
                hit = 1;
            }
            prefix += nt;
        }

        meta[0] = hit;
        meta[1] = q0_sel;
        meta[2] = qe_sel;
        meta[3] = qs_sel;
        meta[4] = ks_sel;
        meta[5] = ke_sel;
        meta[6] = typ_sel;

        meta[7] = q_ranges[14];
        meta[8] = q_ranges[15];
        meta[9] = k_ranges[14];
        meta[10] = k_ranges[15];
        meta[11] = attn_type_map[7];
    }
    __syncthreads();

    if (!meta[0]) return;

    const int q0 = meta[1];
    const int base_qe = meta[2];
    const int base_qs = meta[3];
    const int base_ks = meta[4];
    const int base_ke = meta[5];
    const int base_typ = meta[6];

    const int extra_qs = meta[7];
    const int extra_qe = meta[8];
    const int extra_ks = meta[9];
    const int extra_ke = meta[10];
    const int extra_typ = meta[11];

    const int kvh = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_row0 = warp * 16;

    float* ws = score_s + warp * SCORE_ELEMS_PER_WARP;
    __nv_bfloat16* wp = prob_s + warp * PROB_ELEMS_PER_WARP;
    float* wo = out_s + warp * OUT_ELEMS_PER_WARP;

    for (int chunk = threadIdx.x; chunk < (PM * HD) / 8; chunk += blockDim.x) {
        const int p = chunk / (HD / 8);
        const int c = chunk - p * (HD / 8);
        const int d0 = c * 8;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        const int qidx = q0 + tok_local;
        const int qh = kvh * G + gh;
        __nv_bfloat16* sptr = q_s + p * HD + d0;

        if (qidx < base_qe) {
            const __nv_bfloat16* gptr =
                q + (int64_t(qidx) * HQ + qh) * HD + d0;
            *reinterpret_cast<uint4*>(sptr) =
                *reinterpret_cast<const uint4*>(gptr);
        } else {
            *reinterpret_cast<uint4*>(sptr) = make_uint4(0, 0, 0, 0);
        }
    }

    for (int i = threadIdx.x; i < W * OUT_ELEMS_PER_WARP; i += blockDim.x) {
        out_s[i] = 0.0f;
    }
    __syncthreads();

    using AFrag = wmma::fragment<
        wmma::matrix_a, 16, 16, 16,
        __nv_bfloat16, wmma::row_major>;
    using BColFrag = wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        __nv_bfloat16, wmma::col_major>;
    using BRowFrag = wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        __nv_bfloat16, wmma::row_major>;
    using CFrag = wmma::fragment<
        wmma::accumulator, 16, 16, 16, float>;

    float row_m = -CUDART_INF_F;
    float row_l = 0.0f;
    int row_qidx = 0;
    int row_qh = 0;

    if (lane < 16) {
        const int p = warp_row0 + lane;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        row_qidx = q0 + tok_local;
        row_qh = kvh * G + gh;
    }

    // phase 0 = extra overlapping slice, phase 1 = base partition slice.
    for (int phase = 0; phase < 2; ++phase) {
        const int pqs = (phase == 0) ? extra_qs : base_qs;
        const int pqe = (phase == 0) ? extra_qe : base_qe;
        const int pks = (phase == 0) ? extra_ks : base_ks;
        const int pke = (phase == 0) ? extra_ke : base_ke;
        const int ptyp = (phase == 0) ? extra_typ : base_typ;
        const int pqlen = pqe - pqs;
        const int pklen = pke - pks;
        const int pdelta = pklen - pqlen;
        const int k_tiles = (pklen + KN - 1) / KN;

        for (int kb = 0; kb < k_tiles; ++kb) {
            const int n0 = pks + kb * KN;

            for (int chunk = threadIdx.x; chunk < (KN * HD) / 8; chunk += blockDim.x) {
                const int r = chunk / (HD / 8);
                const int c = chunk - r * (HD / 8);
                const int d0 = c * 8;
                const int kidx = n0 + r;
                __nv_bfloat16* sptr = kv_s + r * HD + d0;

                if (kidx < pke) {
                    const __nv_bfloat16* gptr =
                        k + (int64_t(kidx) * HKV + kvh) * HD + d0;
                    *reinterpret_cast<uint4*>(sptr) =
                        *reinterpret_cast<const uint4*>(gptr);
                } else {
                    *reinterpret_cast<uint4*>(sptr) = make_uint4(0, 0, 0, 0);
                }
            }
            __syncthreads();

            CFrag qk_frag[4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                wmma::fill_fragment(qk_frag[nt], 0.0f);
            }

#pragma unroll
            for (int kk0 = 0; kk0 < HD; kk0 += 16) {
                AFrag af;
                wmma::load_matrix_sync(
                    af,
                    q_s + warp_row0 * HD + kk0,
                    HD
                );
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    BColFrag bf;
                    wmma::load_matrix_sync(
                        bf,
                        kv_s + nt * 16 * HD + kk0,
                        HD
                    );
                    wmma::mma_sync(qk_frag[nt], af, bf, qk_frag[nt]);
                }
            }

#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                wmma::store_matrix_sync(
                    ws + nt * 16,
                    qk_frag[nt],
                    KN,
                    wmma::mem_row_major
                );
            }
            __syncwarp();

            if (lane < 16) {
                const int rr = row_qidx - pqs;
                float tile_m = -CUDART_INF_F;

#pragma unroll
                for (int j = 0; j < KN; ++j) {
                    const int kidx = n0 + j;
                    const int u = kidx - pks;
                    bool visible =
                        (row_qidx < base_qe) &&
                        (row_qidx >= pqs) &&
                        (row_qidx < pqe) &&
                        (kidx < pke);

                    if (ptyp == 1) {
                        visible = visible && (u <= rr + pdelta);
                    } else if (ptyp == 2) {
                        visible = visible && (u >= rr);
                    } else if (ptyp == 3) {
                        visible = visible && (u >= rr) && (u <= rr + pdelta);
                    }

                    if (visible) {
                        tile_m = fmaxf(tile_m, ws[lane * KN + j] * softmax_scale);
                    }
                }

                const float m_new = fmaxf(row_m, tile_m);
                const float alpha =
                    (row_m == -CUDART_INF_F) ? 0.0f : __expf(row_m - m_new);
                float tile_l = 0.0f;

#pragma unroll
                for (int j = 0; j < KN; ++j) {
                    const int kidx = n0 + j;
                    const int u = kidx - pks;
                    bool visible =
                        (row_qidx < base_qe) &&
                        (row_qidx >= pqs) &&
                        (row_qidx < pqe) &&
                        (kidx < pke);

                    if (ptyp == 1) {
                        visible = visible && (u <= rr + pdelta);
                    } else if (ptyp == 2) {
                        visible = visible && (u >= rr);
                    } else if (ptyp == 3) {
                        visible = visible && (u >= rr) && (u <= rr + pdelta);
                    }

                    float p = 0.0f;
                    if (visible) {
                        p = __expf(ws[lane * KN + j] * softmax_scale - m_new);
                        tile_l += p;
                    }
                    wp[lane * KN + j] = __float2bfloat16_rn(p);
                }

                alpha_s[warp * 16 + lane] = alpha;
                row_l = row_l * alpha + tile_l;
                row_m = m_new;
            }
            __syncwarp();

            for (int t = lane; t < OUT_ELEMS_PER_WARP; t += 32) {
                const int r = t / HD;
                wo[t] *= alpha_s[warp * 16 + r];
            }
            __syncwarp();

            __syncthreads();
            for (int chunk = threadIdx.x; chunk < (KN * HD) / 8; chunk += blockDim.x) {
                const int r = chunk / (HD / 8);
                const int c = chunk - r * (HD / 8);
                const int d0 = c * 8;
                const int kidx = n0 + r;
                __nv_bfloat16* sptr = kv_s + r * HD + d0;

                if (kidx < pke) {
                    const __nv_bfloat16* gptr =
                        v + (int64_t(kidx) * HKV + kvh) * HD + d0;
                    *reinterpret_cast<uint4*>(sptr) =
                        *reinterpret_cast<const uint4*>(gptr);
                } else {
                    *reinterpret_cast<uint4*>(sptr) = make_uint4(0, 0, 0, 0);
                }
            }
            __syncthreads();

            CFrag out_frag[8];
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                wmma::load_matrix_sync(
                    out_frag[dt],
                    wo + dt * 16,
                    HD,
                    wmma::mem_row_major
                );
            }

#pragma unroll
            for (int pk = 0; pk < KN; pk += 16) {
                AFrag pf;
                wmma::load_matrix_sync(pf, wp + pk, KN);

#pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    BRowFrag vf;
                    wmma::load_matrix_sync(
                        vf,
                        kv_s + pk * HD + dt * 16,
                        HD
                    );
                    wmma::mma_sync(
                        out_frag[dt],
                        pf,
                        vf,
                        out_frag[dt]
                    );
                }
            }

#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                wmma::store_matrix_sync(
                    wo + dt * 16,
                    out_frag[dt],
                    HD,
                    wmma::mem_row_major
                );
            }
            __syncthreads();
        }
    }

    if (lane < 16) {
        float denom = row_l;
#pragma unroll
        for (int s = 0; s < NSINK; ++s) {
            denom += __expf(sink[s * HQ + row_qh] - row_m);
        }
        alpha_s[warp * 16 + lane] = 1.0f / denom;
    }
    __syncwarp();

    for (int t = lane; t < OUT_ELEMS_PER_WARP; t += 32) {
        const int r = t / HD;
        const int d = t - r * HD;
        const int p = warp_row0 + r;
        const int tok_local = p / G;
        const int gh = p - tok_local * G;
        const int qidx = q0 + tok_local;
        const int qh = kvh * G + gh;

        if (qidx < base_qe) {
            out[(int64_t(qidx) * HQ + qh) * HD + d] =
                __float2bfloat16_rn(wo[t] * alpha_s[warp * 16 + r]);
        }
    }
}


void launch_overlap8_g4_online(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const int32_t* q_ranges,
    const int32_t* k_ranges,
    const int32_t* attn_type_map,
    const float* sink,
    __nv_bfloat16* out,
    float softmax_scale
) {
    constexpr int PM = 128;
    constexpr int KN = 64;
    constexpr int W = 8;
    constexpr int HD = 128;
    constexpr int TOKEN_M = PM / 4;

    constexpr size_t smem_bytes =
        PM * HD * sizeof(__nv_bfloat16) +
        KN * HD * sizeof(__nv_bfloat16) +
        W * 16 * KN * sizeof(float) +
        W * 16 * KN * sizeof(__nv_bfloat16) +
        W * 16 * HD * sizeof(float) +
        W * 16 * sizeof(float);

    cudaFuncSetAttribute(
        overlap8_g4_online_wmma,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes)
    );

    dim3 grid((4096 + TOKEN_M - 1) / TOKEN_M + 7, 8, 1);
    dim3 block(256, 1, 1);
    overlap8_g4_online_wmma<<<grid, block, smem_bytes>>>(
        q, k, v,
        q_ranges, k_ranges, attn_type_map,
        sink, out, softmax_scale
    );
}

} // namespace

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const int32_t* q_ranges,
    const int32_t* k_ranges,
    const int32_t* attn_type_map,
    const float* sink,
    __nv_bfloat16* output,
    float softmax_scale,
    int64_t seqlen,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t num_slices,
    int64_t num_sink
) {
    const int S = int(seqlen);
    const int Hq = int(num_q_heads);
    const int Hkv = int(num_kv_heads);
    const int N = int(num_slices);
    const int Ns = int(num_sink);
    const int G = Hq / Hkv;

    // SM90 CUDA fastpaths: same dense structures that are strongest in the
    // Triton solution, now using one-pass online softmax + PackGQA.
    if (
        S == 4096 && Hq == 32 && Hkv == 8 &&
        head_dim == 128 && N == 1 && Ns == 4
    ) {
        launch_dense_packgqa_online<4096, 32, 8, 4, 4, true>(
            q, k, v, sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 16384 && Hq == 32 && Hkv == 8 &&
        head_dim == 128 && N == 1 && Ns == 4
    ) {
        launch_dense_packgqa_online<16384, 32, 8, 4, 4, true>(
            q, k, v, sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 2048 && Hq == 8 && Hkv == 8 &&
        head_dim == 128 && N == 1 && Ns == 4
    ) {
        launch_dense_packgqa_online<2048, 8, 8, 1, 4, false>(
            q, k, v, sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 8192 && Hq == 128 && Hkv == 1 &&
        head_dim == 128 && N == 1 && Ns == 4
    ) {
        launch_dense_packgqa_online<8192, 128, 1, 128, 4, false>(
            q, k, v, sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 512 && Hq == 16 && Hkv == 8 &&
        head_dim == 128 && N == 2 && Ns == 2
    ) {
        launch_dense_packgqa_online<512, 16, 8, 2, 2, false, 1>(
            q, k, v, sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 8192 && Hq == 64 && Hkv == 8 &&
        head_dim == 128 && N == 2 && Ns == 4
    ) {
        launch_partition_packgqa_online<8192, 64, 8, 8, 128, 2, 4>(
            q, k, v, q_ranges, k_ranges, attn_type_map,
            sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 4096 && Hq == 32 && Hkv == 4 &&
        head_dim == 128 && N == 7 && Ns == 6
    ) {
        launch_partition_packgqa_online<4096, 32, 4, 8, 128, 7, 6>(
            q, k, v, q_ranges, k_ranges, attn_type_map,
            sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 8192 && Hq == 64 && Hkv == 8 &&
        head_dim == 128 && N == 10 && Ns == 8
    ) {
        launch_partition_packgqa_online<8192, 64, 8, 8, 128, 10, 8>(
            q, k, v, q_ranges, k_ranges, attn_type_map,
            sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 4096 && Hq == 8 && Hkv == 2 &&
        head_dim == 128 && N == 7 && Ns == 4
    ) {
        launch_partition_packgqa_online<4096, 8, 2, 4, 128, 7, 4>(
            q, k, v, q_ranges, k_ranges, attn_type_map,
            sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 4096 && Hq == 32 && Hkv == 8 &&
        head_dim == 64 && N == 10 && Ns == 4
    ) {
        launch_partition_packgqa_online<4096, 32, 8, 4, 64, 10, 4>(
            q, k, v, q_ranges, k_ranges, attn_type_map,
            sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 4096 && Hq == 64 && Hkv == 8 &&
        head_dim == 128 && N == 3 && Ns == 4
    ) {
        launch_partition_packgqa_online<4096, 64, 8, 8, 128, 3, 4>(
            q, k, v, q_ranges, k_ranges, attn_type_map,
            sink, output, softmax_scale
        );
        return;
    }

    if (
        S == 4096 && Hq == 32 && Hkv == 8 &&
        head_dim == 128 && N == 8 && Ns == 6
    ) {
        launch_overlap8_g4_online(
            q, k, v,
            q_ranges, k_ranges, attn_type_map,
            sink, output, softmax_scale
        );
        return;
    }

    // Proven 12/12-correct fallback while the remaining static CUDA
    // specializations are being migrated.
    dim3 grid((S * G) / PACKED_M, Hkv, 1);
    dim3 block(128, 1, 1);

    if (head_dim == 128) {
        generic_ffa_wmma<128><<<grid, block>>>(
            q, k, v,
            q_ranges, k_ranges, attn_type_map,
            sink, output,
            softmax_scale,
            S, Hq, Hkv, N, Ns
        );
    } else {
        generic_ffa_wmma<64><<<grid, block>>>(
            q, k, v,
            q_ranges, k_ranges, attn_type_map,
            sink, output,
            softmax_scale,
            S, Hq, Hkv, N, Ns
        );
    }

}
