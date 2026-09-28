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

    // All scored shapes have S*G divisible by 64.
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
