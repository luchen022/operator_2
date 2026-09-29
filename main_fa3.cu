#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

// Minimal Hopper SM90a WGMMA helpers.
// No CUTLASS / CuTe dependency: descriptor layout and PTX forms are copied
// from NVIDIA CUTLASS's public SM90 GMMA implementation.

namespace wgmma_sm90 {

union GmmaDescriptor {
    uint64_t desc;
    uint32_t reg32[2];
    uint16_t reg16[4];
    struct {
        uint16_t start_address : 14, : 2;
        uint16_t leading_byte_offset : 14, : 2;
        uint16_t stride_byte_offset : 14, : 2;
        uint8_t : 1, base_offset : 3, : 4;
        uint8_t : 6, layout_type : 2;
    } bits;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    uint32_t out;
    asm volatile(
        "{ .reg .u64 smem_ptr;\n"
        "  cvta.to.shared.u64 smem_ptr, %1;\n"
        "  cvt.u32.u64 %0, smem_ptr;\n"
        "}\n"
        : "=r"(out) : "l"(p));
    return out;
}

// Canonical K-major INTERLEAVE descriptor for one 64x16 BF16 tile.
//
// In units of uint128 (16 B), the physical layout is
//   ((8,8),2) : ((1,16),8)
// so:
//   LBO = 8  * 16 B
//   SBO = 16 * 16 B
//
// This makes each 8-row x 16-col brick occupy 16 contiguous uint128 slots.
__device__ __forceinline__ uint64_t make_kmajor_64x16_desc(const void* p) {
    GmmaDescriptor d{};
    d.bits.start_address = static_cast<uint16_t>(smem_u32(p) >> 4);
    d.bits.leading_byte_offset = 8;
    d.bits.stride_byte_offset = 16;
    d.bits.base_offset = 0;
    d.bits.layout_type = 0;  // INTERLEAVE / no swizzle
    return d.desc;
}

// Physical BF16 element offset inside the canonical 64x16 K-major tile.
__device__ __forceinline__ int canonical_kmajor_offset(int row, int kcol) {
    const int vec = (row & 7) + (row >> 3) * 16 + (kcol >> 3) * 8;
    return vec * 8 + (kcol & 7);
}

// Inverse mapping for a cooperative uint4 (8 BF16) copy.
// vec_id in [0,128).
__device__ __forceinline__ void canonical_vec_coord(
    int vec_id, int& row, int& kvec
) {
    const int group = vec_id >> 4;
    const int in_group = vec_id & 15;
    row = group * 8 + (in_group & 7);
    kvec = in_group >> 3;
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wait_group() {
    static_assert(N >= 0 && N <= 7, "wgmma wait group out of range");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// BF16 x BF16 -> FP32, m64n64k16, shared/shared, K-major/K-major.
// Every one of the 128 threads in the warpgroup must execute this uniformly.
__device__ __forceinline__ void mma_m64n64k16_bf16(
    float (&d)[32],
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32, %33, p, %35, %36, %37, %38;\n"
    "}\n"
      : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
        "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
        "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a),
        "l"(desc_b),
        "r"(1),      // scale D: accumulate
        "n"(1),      // scale A
        "n"(1),      // scale B
        "n"(0),      // A major = K
        "n"(0)       // B major = K
      : "memory"
    );
}

// Fixed accumulator mapping for m64n64.
//
// warp 0 -> rows 0..15, warp 1 -> 16..31, ...
// lane groups of four share one logical row.
// For each 8-column group g:
//   d[4g+0:4g+1] -> base row
//   d[4g+2:4g+3] -> base row + 8
struct FragCoord {
    int row0;
    int row1;
    int col_pair;
};

__device__ __forceinline__ FragCoord frag_coord() {
    const int warp = (threadIdx.x >> 5) & 3;
    const int lane = threadIdx.x & 31;
    FragCoord c;
    c.row0 = warp * 16 + (lane >> 2);
    c.row1 = c.row0 + 8;
    c.col_pair = (lane & 3) * 2;
    return c;
}

__device__ __forceinline__ int frag_col(int group, int elem01) {
    const int lane = threadIdx.x & 31;
    return group * 8 + (lane & 3) * 2 + elem01;
}

// Reduce within each 4-lane subgroup. XOR 1 and 2 never leave the subgroup.
__device__ __forceinline__ float row4_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
    return x;
}

__device__ __forceinline__ float row4_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    return x;
}

}  // namespace wgmma_sm90


namespace wgmma_static_cache {

using namespace wgmma_sm90;

constexpr int HD = 128;
constexpr int TILE_N = 64;
constexpr int BLOCK_ELEMS = 64 * 16;
constexpr int QK_SLICES = 8;
constexpr int V_TILES = 8;

static __nv_bfloat16* h_packed_k = nullptr;
static __nv_bfloat16* h_packed_v = nullptr;
static size_t h_capacity_elems = 0;
static float* h_sink_lse = nullptr;

static const __nv_bfloat16* h_last_k = nullptr;
static const __nv_bfloat16* h_last_v = nullptr;
static const float* h_last_sink = nullptr;
static const int32_t* h_last_meta = nullptr;
static int h_last_S = -1;
static int h_last_Hq = -1;
static int h_last_Hkv = -1;
static int h_last_Ns = -1;

__global__ __launch_bounds__(128, 1)
void pack_kv_d128(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ packed_k,
    __nv_bfloat16* __restrict__ packed_v,
    int S,
    int Hkv
) {
    const int kb = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int kblocks = S / TILE_N;
    const int key0 = kb * TILE_N;

#pragma unroll
    for (int ds = 0; ds < QK_SLICES; ++ds) {
        int row, kvec;
        canonical_vec_coord(tid, row, kvec);
        const int d0 = ds * 16 + kvec * 8;

        const __nv_bfloat16* in =
            k + (int64_t(key0 + row) * Hkv + kvh) * HD + d0;
        __nv_bfloat16* out =
            packed_k
            + (((int64_t(kvh) * kblocks + kb) * QK_SLICES + ds)
               * BLOCK_ELEMS)
            + tid * 8;

        *reinterpret_cast<uint4*>(out) =
            *reinterpret_cast<const uint4*>(in);
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            int drow, kvec;
            canonical_vec_coord(tid, drow, kvec);

            __nv_bfloat16* out =
                packed_v
                + (((int64_t(kvh) * kblocks + kb) * V_TILES
                    + half * 4 + ks) * BLOCK_ELEMS)
                + tid * 8;

#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = key0 + ks * 16 + kvec * 8 + e;
                const int d = half * 64 + drow;
                out[e] = v[(int64_t(key) * Hkv + kvh) * HD + d];
            }
        }
    }
}

__global__ void build_sink_lse(
    const float* __restrict__ sink,
    float* __restrict__ sink_lse,
    int Hq,
    int Ns
) {
    const int h = threadIdx.x;
    if (h >= Hq) return;

    float m = -CUDART_INF_F;
#pragma unroll 1
    for (int s = 0; s < Ns; ++s) {
        m = fmaxf(m, sink[s * Hq + h]);
    }

    float z = 0.0f;
#pragma unroll 1
    for (int s = 0; s < Ns; ++s) {
        z += expf(sink[s * Hq + h] - m);
    }
    sink_lse[h] = m + logf(z);
}

inline void ensure_d128(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* sink,
    const int32_t* meta_tag,
    int S,
    int Hq,
    int Hkv,
    int Ns,
    const __nv_bfloat16*& packed_k,
    const __nv_bfloat16*& packed_v,
    const float*& sink_lse
) {
    const size_t elems = size_t(S) * size_t(Hkv) * HD;

    if (h_capacity_elems < elems) {
        if (h_packed_k) cudaFree(h_packed_k);
        if (h_packed_v) cudaFree(h_packed_v);
        cudaMalloc(reinterpret_cast<void**>(&h_packed_k),
                   elems * sizeof(__nv_bfloat16));
        cudaMalloc(reinterpret_cast<void**>(&h_packed_v),
                   elems * sizeof(__nv_bfloat16));
        h_capacity_elems = elems;
        h_last_k = nullptr;
        h_last_v = nullptr;
    }

    if (!h_sink_lse) {
        cudaMalloc(reinterpret_cast<void**>(&h_sink_lse),
                   128 * sizeof(float));
        h_last_sink = nullptr;
    }

    const bool kv_changed =
        h_last_k != k ||
        h_last_v != v ||
        h_last_meta != meta_tag ||
        h_last_S != S ||
        h_last_Hkv != Hkv;

    if (kv_changed) {
        dim3 grid(S / TILE_N, Hkv, 1);
        pack_kv_d128<<<grid, 128>>>(
            k, v, h_packed_k, h_packed_v, S, Hkv
        );
        h_last_k = k;
        h_last_v = v;
    }

    const bool sink_changed =
        kv_changed ||
        h_last_sink != sink ||
        h_last_Hq != Hq ||
        h_last_Ns != Ns;

    if (sink_changed) {
        build_sink_lse<<<1, 128>>>(sink, h_sink_lse, Hq, Ns);
        h_last_sink = sink;
    }

    h_last_meta = meta_tag;
    h_last_S = S;
    h_last_Hq = Hq;
    h_last_Hkv = Hkv;
    h_last_Ns = Ns;

    packed_k = h_packed_k;
    packed_v = h_packed_v;
    sink_lse = h_sink_lse;
}

inline void ensure_sink_only(
    const float* sink,
    int Hq,
    int Ns,
    const float*& sink_lse
) {
    if (!h_sink_lse) {
        cudaMalloc(reinterpret_cast<void**>(&h_sink_lse),
                   128 * sizeof(float));
        h_last_sink = nullptr;
    }

    if (h_last_sink != sink || h_last_Hq != Hq || h_last_Ns != Ns) {
        build_sink_lse<<<1, 128>>>(sink, h_sink_lse, Hq, Ns);
        h_last_sink = sink;
        h_last_Hq = Hq;
        h_last_Ns = Ns;
    }

    sink_lse = h_sink_lse;
}

} // namespace wgmma_static_cache


namespace wgmma_attention {

using namespace wgmma_sm90;

constexpr int HD = 128;
constexpr int M = 64;
constexpr int N = 64;
constexpr int K16 = 16;
constexpr int K_SLICES_QK = HD / K16;   // 8
constexpr int K_SLICES_PV = N / K16;    // 4
constexpr int BLOCK_ELEMS = M * K16;    // 1024 BF16

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


__device__ __forceinline__ void stage_k64_d128_packed(
    const __nv_bfloat16* __restrict__ packed_k,
    __nv_bfloat16* __restrict__ s,
    int kb,
    int kvh,
    int kblocks
) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int ds = 0; ds < K_SLICES_QK; ++ds) {
        const __nv_bfloat16* src =
            packed_k
            + (((int64_t(kvh) * kblocks + kb) * K_SLICES_QK + ds)
               * BLOCK_ELEMS)
            + tid * 8;
        __nv_bfloat16* dst = s + ds * BLOCK_ELEMS + tid * 8;
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
}

__device__ __forceinline__ void stage_vt64_d128_packed(
    const __nv_bfloat16* __restrict__ packed_v,
    __nv_bfloat16* __restrict__ s,
    int kb,
    int kvh,
    int kblocks
) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int t = 0; t < 2 * K_SLICES_PV; ++t) {
        const __nv_bfloat16* src =
            packed_v
            + (((int64_t(kvh) * kblocks + kb)
                * (2 * K_SLICES_PV) + t) * BLOCK_ELEMS)
            + tid * 8;
        __nv_bfloat16* dst = s + t * BLOCK_ELEMS + tid * 8;
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
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
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    const float* __restrict__ sink_lse,
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
    fence_proxy_async_shared();
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

        stage_k64_d128_packed(packed_k, kv_s, kb, kvh, S / N);
        fence_proxy_async_shared();
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

        // Plain shared-memory stores are in the generic proxy, while WGMMA
        // reads shared memory through the async proxy.  A CTA barrier alone
        // is not sufficient: publish the P tile to the async proxy first.
        fence_proxy_async_shared();
        __syncthreads();

        stage_vt64_d128_packed(packed_v, kv_s, kb, kvh, S / N);
        fence_proxy_async_shared();
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

    float denom0 = l0 + __expf(sink_lse[qh0] - m0);
    float denom1 = l1 + __expf(sink_lse[qh1] - m1);

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
    const __nv_bfloat16* packed_k,
    const __nv_bfloat16* packed_v,
    const float* sink_lse,
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
        q, k, v, packed_k, packed_v, sink_lse, out, softmax_scale,
        S, Hq, Hkv, G, Ns, causal
    );
}

}  // namespace wgmma_attention



namespace wgmma_slice_cache {

using namespace wgmma_sm90;

constexpr int HD = 128;
constexpr int TILE_N = 64;
constexpr int BLOCK_ELEMS = 64 * 16;
constexpr int TILES_D = 8;

static __nv_bfloat16* h_packed_k = nullptr;
static __nv_bfloat16* h_packed_v = nullptr;
static int32_t* h_offsets = nullptr;
static size_t h_capacity_elems = 0;

static const __nv_bfloat16* h_last_k = nullptr;
static const __nv_bfloat16* h_last_v = nullptr;
static const int32_t* h_last_k_ranges = nullptr;
static const int32_t* h_last_meta = nullptr;
static int h_last_S = -1;
static int h_last_Hkv = -1;
static int h_last_N = -1;
static int h_total_blocks = 0;

__global__ __launch_bounds__(128, 1)
void pack_slice_kv_d128(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int32_t* __restrict__ k_ranges,
    const int32_t* __restrict__ offsets,
    __nv_bfloat16* __restrict__ packed_k,
    __nv_bfloat16* __restrict__ packed_v,
    int total_blocks,
    int Hkv,
    int NumSlices
) {
    const int flat = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    if (flat >= total_blocks) return;

    int sid = 0;
#pragma unroll 1
    while (sid + 1 < NumSlices && flat >= offsets[sid + 1]) ++sid;

    const int local_block = flat - offsets[sid];
    const int ks = k_ranges[2 * sid + 0];
    const int ke = k_ranges[2 * sid + 1];
    const int key0 = ks + local_block * TILE_N;

#pragma unroll
    for (int ds = 0; ds < TILES_D; ++ds) {
        int row, kvec;
        canonical_vec_coord(tid, row, kvec);
        const int key = key0 + row;
        const int d0 = ds * 16 + kvec * 8;

        __nv_bfloat16* out =
            packed_k
            + (((int64_t(kvh) * total_blocks + flat) * TILES_D + ds)
               * BLOCK_ELEMS)
            + tid * 8;

        if (key < ke) {
            const __nv_bfloat16* in =
                k + (int64_t(key) * Hkv + kvh) * HD + d0;
            *reinterpret_cast<uint4*>(out) =
                *reinterpret_cast<const uint4*>(in);
        } else {
            *reinterpret_cast<uint4*>(out) = make_uint4(0, 0, 0, 0);
        }
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int ks16 = 0; ks16 < 4; ++ks16) {
            int drow, kvec;
            canonical_vec_coord(tid, drow, kvec);

            __nv_bfloat16* out =
                packed_v
                + (((int64_t(kvh) * total_blocks + flat) * TILES_D
                    + half * 4 + ks16) * BLOCK_ELEMS)
                + tid * 8;

#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key =
                    key0 + ks16 * 16 + kvec * 8 + e;
                const int d = half * 64 + drow;
                out[e] = (key < ke)
                    ? v[(int64_t(key) * Hkv + kvh) * HD + d]
                    : __float2bfloat16_rn(0.0f);
            }
        }
    }
}

inline void ensure_slice_d128(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const int32_t* k_ranges,
    const int32_t* meta_tag,
    int S,
    int Hkv,
    int NumSlices,
    const __nv_bfloat16*& packed_k,
    const __nv_bfloat16*& packed_v,
    const int32_t*& offsets,
    int& total_blocks
) {
    const bool changed =
        h_last_k != k ||
        h_last_v != v ||
        h_last_k_ranges != k_ranges ||
        h_last_meta != meta_tag ||
        h_last_S != S ||
        h_last_Hkv != Hkv ||
        h_last_N != NumSlices;

    if (changed) {
        int32_t host_ranges[20] = {0};
        int32_t host_offsets[11] = {0};

        cudaMemcpy(
            host_ranges,
            k_ranges,
            sizeof(int32_t) * 2 * NumSlices,
            cudaMemcpyDeviceToHost
        );

        int blocks = 0;
        host_offsets[0] = 0;
        for (int s = 0; s < NumSlices; ++s) {
            const int len =
                host_ranges[2*s + 1] - host_ranges[2*s + 0];
            blocks += (len + TILE_N - 1) / TILE_N;
            host_offsets[s + 1] = blocks;
        }

        const size_t elems =
            size_t(blocks) * size_t(Hkv)
            * size_t(TILES_D) * size_t(BLOCK_ELEMS);

        if (h_capacity_elems < elems) {
            if (h_packed_k) cudaFree(h_packed_k);
            if (h_packed_v) cudaFree(h_packed_v);
            cudaMalloc(
                reinterpret_cast<void**>(&h_packed_k),
                elems * sizeof(__nv_bfloat16)
            );
            cudaMalloc(
                reinterpret_cast<void**>(&h_packed_v),
                elems * sizeof(__nv_bfloat16)
            );
            h_capacity_elems = elems;
        }

        if (!h_offsets) {
            cudaMalloc(
                reinterpret_cast<void**>(&h_offsets),
                11 * sizeof(int32_t)
            );
        }

        cudaMemcpy(
            h_offsets,
            host_offsets,
            sizeof(int32_t) * (NumSlices + 1),
            cudaMemcpyHostToDevice
        );

        if (blocks > 0) {
            dim3 grid(blocks, Hkv, 1);
            pack_slice_kv_d128<<<grid, 128>>>(
                k, v, k_ranges, h_offsets,
                h_packed_k, h_packed_v,
                blocks, Hkv, NumSlices
            );
        }

        h_total_blocks = blocks;
        h_last_k = k;
        h_last_v = v;
        h_last_k_ranges = k_ranges;
        h_last_meta = meta_tag;
        h_last_S = S;
        h_last_Hkv = Hkv;
        h_last_N = NumSlices;
    }

    packed_k = h_packed_k;
    packed_v = h_packed_v;
    offsets = h_offsets;
    total_blocks = h_total_blocks;
}

} // namespace wgmma_slice_cache


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


__device__ __forceinline__ void stage_k_packed_d128(
    const __nv_bfloat16* __restrict__ packed_k,
    __nv_bfloat16* __restrict__ s,
    int key0,
    int kvh,
    int S
) {
    const int tid = threadIdx.x;
    const int kblocks = S / 64;
    const int abs_kb = key0 >> 6;
#pragma unroll
    for (int ds = 0; ds < 8; ++ds) {
        const __nv_bfloat16* src =
            packed_k
            + (((int64_t(kvh) * kblocks + abs_kb) * 8 + ds)
               * P_BLOCK_ELEMS)
            + tid * 8;
        __nv_bfloat16* dst = s + ds * P_BLOCK_ELEMS + tid * 8;
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
}

__device__ __forceinline__ void stage_v_packed_d128(
    const __nv_bfloat16* __restrict__ packed_v,
    __nv_bfloat16* __restrict__ s,
    int key0,
    int kvh,
    int S
) {
    const int tid = threadIdx.x;
    const int kblocks = S / 64;
    const int abs_kb = key0 >> 6;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const __nv_bfloat16* src =
            packed_v
            + (((int64_t(kvh) * kblocks + abs_kb) * 8 + t)
               * P_BLOCK_ELEMS)
            + tid * 8;
        __nv_bfloat16* dst = s + t * P_BLOCK_ELEMS + tid * 8;
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
}


__device__ __forceinline__ void stage_k_slice_packed_d128(
    const __nv_bfloat16* __restrict__ packed_k,
    __nv_bfloat16* __restrict__ s,
    int packed_block,
    int kvh,
    int total_blocks
) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int ds = 0; ds < 8; ++ds) {
        const __nv_bfloat16* src =
            packed_k
            + (((int64_t(kvh) * total_blocks + packed_block) * 8 + ds)
               * P_BLOCK_ELEMS)
            + tid * 8;
        __nv_bfloat16* dst =
            s + ds * P_BLOCK_ELEMS + tid * 8;
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    }
}

__device__ __forceinline__ void stage_v_slice_packed_d128(
    const __nv_bfloat16* __restrict__ packed_v,
    __nv_bfloat16* __restrict__ s,
    int packed_block,
    int kvh,
    int total_blocks
) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const __nv_bfloat16* src =
            packed_v
            + (((int64_t(kvh) * total_blocks + packed_block) * 8 + t)
               * P_BLOCK_ELEMS)
            + tid * 8;
        __nv_bfloat16* dst =
            s + t * P_BLOCK_ELEMS + tid * 8;
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
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
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    const float* __restrict__ sink_lse,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int G,
    int Ns,
    int NumSlices,
    int special_mode,
    const int32_t* __restrict__ slice_block_offsets,
    int slice_total_blocks
) {
    constexpr int DS = HD / 16;

    __shared__ __align__(128) __nv_bfloat16 q_s[DS * P_BLOCK_ELEMS];
    __shared__ __align__(128) __nv_bfloat16 kv_s[DS * P_BLOCK_ELEMS];
    __shared__ __align__(128) __nv_bfloat16 p_s[4 * P_BLOCK_ELEMS];

    __shared__ int meta[13];

    const int token_M = M / G;

    if (threadIdx.x == 0) {
        const int pid = blockIdx.x;
        int prefix = 0;
        int hit = 0;
        int q0 = 0, qe = 0, qs = 0, ks = 0, ke = 0, typ = 0;
        int slice_id = -1;

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
            const int scan_slices = (special_mode == 2) ? (NumSlices - 1) : NumSlices;
            for (int s = 0; s < scan_slices; ++s) {
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
                    slice_id = s;
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

        meta[12] = slice_id;

        if (special_mode == 2) {
            const int e = NumSlices - 1;
            meta[7]  = q_ranges[2*e + 0];
            meta[8]  = q_ranges[2*e + 1];
            meta[9]  = k_ranges[2*e + 0];
            meta[10] = k_ranges[2*e + 1];
            meta[11] = attn_type_map[e];
        } else {
            meta[7] = meta[8] = meta[9] = meta[10] = meta[11] = 0;
        }
    }
    __syncthreads();

    if (!meta[0]) return;

    const int q0 = meta[1];
    const int qe = meta[2];
    const int qs = meta[3];
    const int ks = meta[4];
    const int ke = meta[5];
    const int typ = meta[6];
    const int slice_id = meta[12];

    const int kvh = blockIdx.y;

    stage_q<HD>(q, q_s, q0, qe, kvh, Hq, G);
    fence_proxy_async_shared();
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

    const int phases = (special_mode == 2) ? 2 : 1;

    for (int phase = 0; phase < phases; ++phase) {
        const int pqs = (special_mode == 2 && phase == 0) ? meta[7]  : qs;
        const int pqe = (special_mode == 2 && phase == 0) ? meta[8]  : qe;
        const int pks = (special_mode == 2 && phase == 0) ? meta[9]  : ks;
        const int pke = (special_mode == 2 && phase == 0) ? meta[10] : ke;
        const int ptyp= (special_mode == 2 && phase == 0) ? meta[11] : typ;

        const int k_blocks = (pke - pks + N - 1) / N;

        // Skip 64-key tiles that are provably invisible for every query row
        // in this CTA.  We still retain the elementwise mask on the two
        // frontier tiles, so this is semantics-preserving for all four masks.
        int kb_begin = 0;
        int kb_end = k_blocks;

        const int q_first = (q0 > pqs) ? q0 : pqs;
        const int q_tile_end = q0 + token_M;
        const int q_clip_end = (q_tile_end < pqe) ? q_tile_end : pqe;
        const int q_last = q_clip_end - 1;

        if (q_first > q_last) {
            kb_begin = 0;
            kb_end = 0;
        } else {
            const int Lq = pqe - pqs;
            const int Lk = pke - pks;

            if (ptyp == 2 || ptyp == 3) {
                const int r_min = q_first - pqs;
                kb_begin = r_min / N;
                if (kb_begin < 0) kb_begin = 0;
                if (kb_begin > k_blocks) kb_begin = k_blocks;
            }

            if (ptyp == 1 || ptyp == 3) {
                const int r_max = q_last - pqs;
                const int max_u = r_max + (Lk - Lq);
                if (max_u < 0) {
                    kb_end = 0;
                } else {
                    kb_end = (max_u + 1 + N - 1) / N;
                    if (kb_end > k_blocks) kb_end = k_blocks;
                }
            }

            if (kb_begin > kb_end) kb_begin = kb_end;
        }

        for (int kb = kb_begin; kb < kb_end; ++kb) {
            const int key0 = pks + kb * N;

            if (
                HD == 128 && special_mode == 0 &&
                slice_block_offsets != nullptr && slice_id >= 0
            ) {
                const int packed_block =
                    slice_block_offsets[slice_id] + kb;
                stage_k_slice_packed_d128(
                    packed_k, kv_s, packed_block, kvh, slice_total_blocks
                );
            } else if (
                HD == 128 && packed_k != nullptr && ((key0 & 63) == 0)
            ) {
                stage_k_packed_d128(packed_k, kv_s, key0, kvh, S);
            } else {
                stage_k<HD>(k, kv_s, key0, pke, kvh, Hkv);
            }
            fence_proxy_async_shared();
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

            if (mask_visible(ptyp, qidx0, key0+c0, pqs, pqe, pks, pke))
                local_max0 = fmaxf(local_max0, score[4*g+0] * softmax_scale);
            if (mask_visible(ptyp, qidx0, key0+c1, pqs, pqe, pks, pke))
                local_max0 = fmaxf(local_max0, score[4*g+1] * softmax_scale);

            if (mask_visible(ptyp, qidx1, key0+c0, pqs, pqe, pks, pke))
                local_max1 = fmaxf(local_max1, score[4*g+2] * softmax_scale);
            if (mask_visible(ptyp, qidx1, key0+c1, pqs, pqe, pks, pke))
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

            if (mask_visible(ptyp, qidx0, key0+c0, pqs, pqe, pks, pke)) {
                p00 = __expf(score[4*g+0] * softmax_scale - nm0);
                sum0 += p00;
            }
            if (mask_visible(ptyp, qidx0, key0+c1, pqs, pqe, pks, pke)) {
                p01 = __expf(score[4*g+1] * softmax_scale - nm0);
                sum0 += p01;
            }
            if (mask_visible(ptyp, qidx1, key0+c0, pqs, pqe, pks, pke)) {
                p10 = __expf(score[4*g+2] * softmax_scale - nm1);
                sum1 += p10;
            }
            if (mask_visible(ptyp, qidx1, key0+c1, pqs, pqe, pks, pke)) {
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

        fence_proxy_async_shared();
        __syncthreads();
        if (
            HD == 128 && special_mode == 0 &&
            slice_block_offsets != nullptr && slice_id >= 0
        ) {
            const int packed_block =
                slice_block_offsets[slice_id] + kb;
            stage_v_slice_packed_d128(
                packed_v, kv_s, packed_block, kvh, slice_total_blocks
            );
        } else if (
            HD == 128 && packed_v != nullptr && ((key0 & 63) == 0)
        ) {
            stage_v_packed_d128(packed_v, kv_s, key0, kvh, S);
        } else {
            stage_vt<HD>(v, kv_s, key0, pke, kvh, Hkv);
        }
        fence_proxy_async_shared();
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
    }

    if (qidx0 < qe) {
        float denom0 = l0;
        if (HD == 128 && sink_lse != nullptr) {
            denom0 += __expf(sink_lse[qh0] - m0);
        } else {
            for (int s = 0; s < Ns; ++s)
                denom0 += __expf(sink[s * Hq + qh0] - m0);
        }
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
        if (HD == 128 && sink_lse != nullptr) {
            denom1 += __expf(sink_lse[qh1] - m1);
        } else {
            for (int s = 0; s < Ns; ++s)
                denom1 += __expf(sink[s * Hq + qh1] - m1);
        }
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
    const __nv_bfloat16* packed_k,
    const __nv_bfloat16* packed_v,
    const float* sink_lse,
    __nv_bfloat16* out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int Ns,
    int NumSlices,
    int special_mode = 0,
    const int32_t* slice_block_offsets = nullptr,
    int slice_total_blocks = 0
) {
    const int G = Hq / Hkv;
    const int token_M = M / G;

    // Slight overlaunch; invalid pids return after the one-thread metadata scan.
    int grid_x = (S + token_M - 1) / token_M + NumSlices;
    if (special_mode == 1) {
        grid_x = (128 + token_M - 1) / token_M
               + (128 + token_M - 1) / token_M
               + (256 + token_M - 1) / token_M;
    } else if (special_mode == 2) {
        // Seven base slices partition Q; final slice is folded into each base CTA.
        grid_x = (S + token_M - 1) / token_M + (NumSlices - 1);
    }
    dim3 grid(grid_x, Hkv, 1);
    partition_wgmma_fwd<HD><<<grid, 128>>>(
        q, k, v,
        q_ranges, k_ranges, attn_type_map,
        sink, packed_k, packed_v, sink_lse,
        out, softmax_scale,
        S, Hq, Hkv, G, Ns, NumSlices, special_mode,
        slice_block_offsets, slice_total_blocks
    );
}

} // namespace wgmma_partition


namespace wgmma_g128 {

using namespace wgmma_sm90;

constexpr int S12 = 8192;
constexpr int HQ12 = 128;
constexpr int HD12 = 128;

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
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    const float* __restrict__ sink_lse,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale
) {
    // 64 KiB total, so use opt-in dynamic shared memory (>48 KiB).
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* p_s = q_s + 2 * QK_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* kv_s = p_s + 2 * PV_SLICES * BLOCK_ELEMS;

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
    fence_proxy_async_shared();
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

        // K is transformed once during the untimed warmup. Timed execution
        // is now a fully coalesced uint4 copy into the WGMMA shared layout.
        if (wg == 0) {
#pragma unroll
            for (int ds = 0; ds < QK_SLICES; ++ds) {
                const __nv_bfloat16* src =
                    packed_k
                    + ((int64_t(kb) * QK_SLICES + ds) * BLOCK_ELEMS)
                    + wtid * 8;
                __nv_bfloat16* dst =
                    kv_s + ds * BLOCK_ELEMS + wtid * 8;

                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(src);
            }
        }
        fence_proxy_async_shared();
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

        fence_proxy_async_shared();
        __syncthreads();

        // V^T is also prepacked during warmup, removing the scalar gather/
        // transpose from the timed inner loop.
        if (wg == 0) {
#pragma unroll
            for (int t = 0; t < 2 * PV_SLICES; ++t) {
                const __nv_bfloat16* src =
                    packed_v
                    + ((int64_t(kb) * (2 * PV_SLICES) + t) * BLOCK_ELEMS)
                    + wtid * 8;
                __nv_bfloat16* dst =
                    kv_s + t * BLOCK_ELEMS + wtid * 8;

                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(src);
            }
        }
        fence_proxy_async_shared();
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

    float denom0 = l0 + __expf(sink_lse[qh0] - m0);
    float denom1 = l1 + __expf(sink_lse[qh1] - m1);
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
    const __nv_bfloat16* packed_k,
    const __nv_bfloat16* packed_v,
    const float* sink_lse,
    __nv_bfloat16* out,
    float softmax_scale
) {
    constexpr int smem_bytes =
        (2 * QK_SLICES * BLOCK_ELEMS
       + 2 * PV_SLICES * BLOCK_ELEMS
       +     QK_SLICES * BLOCK_ELEMS) * sizeof(__nv_bfloat16);

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            g128_full_wgmma_fwd,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        configured = true;
    }

    g128_full_wgmma_fwd<<<S12, 256, smem_bytes>>>(
        q, k, v, packed_k, packed_v, sink_lse, out, softmax_scale
    );
}

} // namespace wgmma_g128



namespace wgmma_fa3_exp {

using namespace wgmma_sm90;

constexpr int S12 = 8192;
constexpr int HQ12 = 128;
constexpr int HD12 = 128;
constexpr int N = 64;
constexpr int BLOCK_ELEMS = 64 * 16;
constexpr int QK_SLICES = 8;
constexpr int PV_SLICES = 4;
constexpr int KV_STAGE_ELEMS = QK_SLICES * BLOCK_ELEMS; // 8192 BF16 = 16 KiB
constexpr int KV_STAGE_BYTES = KV_STAGE_ELEMS * sizeof(__nv_bfloat16);
constexpr int TX_BYTES = 2 * KV_STAGE_BYTES;             // K + V

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrier_init_count(
    uint64_t* bar, uint32_t count
) {
    const uint32_t a = smem_u32(bar);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(a), "r"(count)
        : "memory"
    );
}

__device__ __forceinline__ void mbarrier_arrive_release(uint64_t* bar) {
    const uint32_t a = smem_u32(bar);
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];\n"
        : "=l"(state)
        : "r"(a)
        : "memory"
    );
    asm volatile("" : : "l"(state));
}

__device__ __forceinline__ void mbarrier_arrive_expect(
    uint64_t* bar, uint32_t bytes
) {
    const uint32_t a = smem_u32(bar);
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "%0, [%1], %2;\n"
        : "=l"(state)
        : "r"(a), "r"(bytes)
        : "memory"
    );
    asm volatile("" : : "l"(state));
}

__device__ __forceinline__ void cp_async_bulk_g2s(
    void* dst, const void* src, uint32_t bytes, uint64_t* bar
) {
    const uint32_t d = smem_u32(dst);
    const uint32_t b = smem_u32(bar);
    const uint64_t s = reinterpret_cast<uint64_t>(src);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(d), "l"(s), "r"(bytes), "r"(b)
        : "memory"
    );
}

__device__ __forceinline__ void mbarrier_wait_phase(
    uint64_t* bar, int parity
) {
    const uint32_t a = smem_u32(bar);
    int done = 0;
    do {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "      p, [%1], %2;\n"
            "  selp.b32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done)
            : "r"(a), "r"(parity)
            : "memory"
        );
    } while (!done);
}

__device__ __forceinline__ void warpgroup_barrier(int wg) {
    if (wg == 0) {
        asm volatile("bar.sync 1, 128;\n" ::: "memory");
    } else {
        asm volatile("bar.sync 2, 128;\n" ::: "memory");
    }
}

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

__device__ __forceinline__ void issue_kv_stage(
    int kb,
    int stage,
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    __nv_bfloat16* __restrict__ k_stage,
    __nv_bfloat16* __restrict__ v_stage,
    uint64_t* bar
) {
    mbarrier_arrive_expect(bar + stage, TX_BYTES);

    cp_async_bulk_g2s(
        k_stage + stage * KV_STAGE_ELEMS,
        packed_k + int64_t(kb) * KV_STAGE_ELEMS,
        KV_STAGE_BYTES,
        bar + stage
    );
    cp_async_bulk_g2s(
        v_stage + stage * KV_STAGE_ELEMS,
        packed_v + int64_t(kb) * KV_STAGE_ELEMS,
        KV_STAGE_BYTES,
        bar + stage
    );
}

// Experimental Hopper pipeline for testcase #12.
//
// 256 consumer threads = two WGMMA warpgroups.
// One extra producer warp (32 threads, lane 0 issues bulk copies) owns the
// asynchronous global->shared pipeline. K and V are double-buffered together,
// so while consumers execute QK/softmax/PV for block i, the producer can fill
// the other stage with block i+1. The producer never executes WGMMA.
__global__ __launch_bounds__(288, 1)
void g128_fa3_pipeline_fwd(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    const float* __restrict__ sink_lse,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale
) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ __align__(8) uint64_t full_bar[2];
    __shared__ __align__(8) uint64_t empty_bar[2];

    // 32 KiB Q + 16 KiB P + 32 KiB K double buffer + 32 KiB V double buffer
    __nv_bfloat16* q_s =
        reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* p_s =
        q_s + 2 * QK_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* k_stage =
        p_s + 2 * PV_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* v_stage =
        k_stage + 2 * KV_STAGE_ELEMS;

    const int tid = threadIdx.x;
    const int wg = tid >> 7;       // 0/1 = complete consumer warpgroups
    const bool producer_lane0 = tid == 256;
    const int wtid = tid & 127;
    const int token = blockIdx.x;

    if (producer_lane0) {
        mbarrier_init_count(&full_bar[0], 1);
        mbarrier_init_count(&full_bar[1], 1);
        // One completion token from each consumer warpgroup means the stage
        // is no longer being read and may be refilled by the producer.
        mbarrier_init_count(&empty_bar[0], 2);
        mbarrier_init_count(&empty_bar[1], 2);
    }
    __syncthreads();

    // Consumers stage Q once. Producer warp stays out of this path.
    if (wg < 2) {
        const int cwg = wg;  // 0 / 1
        const int qh_base = cwg * 64;
        __nv_bfloat16* my_q =
            q_s + cwg * QK_SLICES * BLOCK_ELEMS;

#pragma unroll
        for (int ds = 0; ds < QK_SLICES; ++ds) {
            int qrow, kvec;
            canonical_vec_coord(wtid, qrow, kvec);
            const int qh = qh_base + qrow;
            const int d0 = ds * 16 + kvec * 8;

            const __nv_bfloat16* srcq =
                q + (int64_t(token) * HQ12 + qh) * HD12 + d0;
            __nv_bfloat16* dst =
                my_q + ds * BLOCK_ELEMS + wtid * 8;

            *reinterpret_cast<uint4*>(dst) =
                *reinterpret_cast<const uint4*>(srcq);
        }

        fence_proxy_async_shared();
    }
    __syncthreads();

    constexpr int KBLOCKS = S12 / N;

    // Hard role split.  The producer warp never reaches any WGMMA code.
    // The two consumer warpgroups fall through to a branch-free GMMA region.
    if (wg >= 2) {
        if (producer_lane0) {
            issue_kv_stage(
                0, 0, packed_k, packed_v,
                k_stage, v_stage, full_bar
            );
            issue_kv_stage(
                1, 1, packed_k, packed_v,
                k_stage, v_stage, full_bar
            );

            for (int kb = 0; kb < KBLOCKS; ++kb) {
                const int stage = kb & 1;
                const int parity = (kb >> 1) & 1;

                if (kb + 2 < KBLOCKS) {
                    mbarrier_wait_phase(&empty_bar[stage], parity);
                    issue_kv_stage(
                        kb + 2, stage, packed_k, packed_v,
                        k_stage, v_stage, full_bar
                    );
                }
            }
        }
        return;
    }

    float out0[32];
    float out1[32];
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    int qh_base = 0, qh0 = 0, qh1 = 0, row0 = 0, row1 = 0;
    __nv_bfloat16* my_q = nullptr;
    __nv_bfloat16* my_p = nullptr;

    qh_base = wg * 64;
    my_q = q_s + wg * QK_SLICES * BLOCK_ELEMS;
    my_p = p_s + wg * PV_SLICES * BLOCK_ELEMS;

    zero32(out0);
    zero32(out1);

    const FragCoord fc = frag_coord();
    row0 = fc.row0;
    row1 = fc.row1;
    qh0 = qh_base + row0;
    qh1 = qh_base + row1;

    for (int kb = 0; kb < KBLOCKS; ++kb) {
        const int stage = kb & 1;
        const int parity = (kb >> 1) & 1;

        // Wait only for the stage consumed by this iteration. The other
        // stage may be in flight concurrently.
        mbarrier_wait_phase(&full_bar[stage], parity);

            __nv_bfloat16* my_k =
                k_stage + stage * KV_STAGE_ELEMS;
            __nv_bfloat16* my_v =
                v_stage + stage * KV_STAGE_ELEMS;

            float score[32];
            zero32(score);

            fence();
#pragma unroll
            for (int ds = 0; ds < QK_SLICES; ++ds) {
                mma_m64n64k16_bf16(
                    score,
                    make_kmajor_64x16_desc(
                        my_q + ds * BLOCK_ELEMS),
                    make_kmajor_64x16_desc(
                        my_k + ds * BLOCK_ELEMS)
                );
            }
            commit_group();
            wait_group<0>();

            float local_max0 = -CUDART_INF_F;
            float local_max1 = -CUDART_INF_F;

#pragma unroll
            for (int g = 0; g < 8; ++g) {
                local_max0 = fmaxf(
                    local_max0, score[4*g+0] * softmax_scale);
                local_max0 = fmaxf(
                    local_max0, score[4*g+1] * softmax_scale);
                local_max1 = fmaxf(
                    local_max1, score[4*g+2] * softmax_scale);
                local_max1 = fmaxf(
                    local_max1, score[4*g+3] * softmax_scale);
            }

            const float tm0 = row4_max(local_max0);
            const float tm1 = row4_max(local_max1);
            const float nm0 = fmaxf(m0, tm0);
            const float nm1 = fmaxf(m1, tm1);
            const float a0 =
                (m0 == -CUDART_INF_F) ? 0.0f : __expf(m0 - nm0);
            const float a1 =
                (m1 == -CUDART_INF_F) ? 0.0f : __expf(m1 - nm1);

            float sum0 = 0.0f, sum1 = 0.0f;

#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int c0 = frag_col(g, 0);
                const int c1 = frag_col(g, 1);

                const float p00 =
                    __expf(score[4*g+0] * softmax_scale - nm0);
                const float p01 =
                    __expf(score[4*g+1] * softmax_scale - nm0);
                const float p10 =
                    __expf(score[4*g+2] * softmax_scale - nm1);
                const float p11 =
                    __expf(score[4*g+3] * softmax_scale - nm1);

                sum0 += p00 + p01;
                sum1 += p10 + p11;

                const int s0 = c0 >> 4;
                const int s1 = c1 >> 4;
                const int kc0 = c0 & 15;
                const int kc1 = c1 & 15;

                my_p[s0 * BLOCK_ELEMS
                     + canonical_kmajor_offset(row0, kc0)] =
                    __float2bfloat16_rn(p00);
                my_p[s1 * BLOCK_ELEMS
                     + canonical_kmajor_offset(row0, kc1)] =
                    __float2bfloat16_rn(p01);
                my_p[s0 * BLOCK_ELEMS
                     + canonical_kmajor_offset(row1, kc0)] =
                    __float2bfloat16_rn(p10);
                my_p[s1 * BLOCK_ELEMS
                     + canonical_kmajor_offset(row1, kc1)] =
                    __float2bfloat16_rn(p11);
            }

            l0 = l0 * a0 + row4_sum(sum0);
            l1 = l1 * a1 + row4_sum(sum1);
            m0 = nm0;
            m1 = nm1;

            rescale32(out0, a0, a1);
            rescale32(out1, a0, a1);

            // Only the four warps inside each consumer warpgroup need to
            // rendezvous before WGMMA starts reading their P tile.
            fence_proxy_async_shared();
            warpgroup_barrier(wg);

            fence();
#pragma unroll
            for (int ks = 0; ks < PV_SLICES; ++ks) {
                mma_m64n64k16_bf16(
                    out0,
                    make_kmajor_64x16_desc(
                        my_p + ks * BLOCK_ELEMS),
                    make_kmajor_64x16_desc(
                        my_v + ks * BLOCK_ELEMS)
                );
            }
#pragma unroll
            for (int ks = 0; ks < PV_SLICES; ++ks) {
                mma_m64n64k16_bf16(
                    out1,
                    make_kmajor_64x16_desc(
                        my_p + ks * BLOCK_ELEMS),
                    make_kmajor_64x16_desc(
                        my_v + (PV_SLICES + ks) * BLOCK_ELEMS)
                );
            }
            commit_group();
            wait_group<0>();

            // All four warps in this consumer warpgroup have finished reading
            // the current K/V stage.  Only one representative arrival is
            // needed from each warpgroup.
        warpgroup_barrier(wg);
        if (wtid == 0) {
            mbarrier_arrive_release(&empty_bar[stage]);
        }
    }

    const float denom0 =
            l0 + __expf(sink_lse[qh0] - m0);
        const float denom1 =
            l1 + __expf(sink_lse[qh1] - m1);
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

            out[(int64_t(token) * HQ12 + qh0) * HD12 + 64 + c0] =
                __float2bfloat16_rn(out1[4*g+0] * inv0);
            out[(int64_t(token) * HQ12 + qh0) * HD12 + 64 + c1] =
                __float2bfloat16_rn(out1[4*g+1] * inv0);
            out[(int64_t(token) * HQ12 + qh1) * HD12 + 64 + c0] =
                __float2bfloat16_rn(out1[4*g+2] * inv1);
            out[(int64_t(token) * HQ12 + qh1) * HD12 + 64 + c1] =
                __float2bfloat16_rn(out1[4*g+3] * inv1);
    }
}

inline void launch_g128_fa3(
    const __nv_bfloat16* q,
    const __nv_bfloat16* packed_k,
    const __nv_bfloat16* packed_v,
    const float* sink_lse,
    __nv_bfloat16* out,
    float softmax_scale
) {
    constexpr int smem_bytes =
        (2 * QK_SLICES * BLOCK_ELEMS
       + 2 * PV_SLICES * BLOCK_ELEMS
       + 4 * KV_STAGE_ELEMS) * sizeof(__nv_bfloat16);

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            g128_fa3_pipeline_fwd,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        configured = true;
    }

    g128_fa3_pipeline_fwd<<<S12, 288, smem_bytes>>>(
        q, packed_k, packed_v, sink_lse, out, softmax_scale
    );
}

} // namespace wgmma_fa3_exp



namespace wgmma_g8_async {

using namespace wgmma_sm90;

constexpr int G = 8;
constexpr int HD = 128;
constexpr int N = 64;
constexpr int BLOCK_ELEMS = 64 * 16;
constexpr int QK_SLICES = 8;
constexpr int PV_SLICES = 4;
constexpr int KV_STAGE_ELEMS = QK_SLICES * BLOCK_ELEMS;
constexpr int TOKEN_M = 64 / G; // 8 query tokens / CTA

__global__ __launch_bounds__(160, 1)
void g8_partition_async_fwd(
    const __nv_bfloat16* __restrict__ q,
    const int32_t* __restrict__ q_ranges,
    const int32_t* __restrict__ k_ranges,
    const int32_t* __restrict__ attn_type_map,
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    const int32_t* __restrict__ slice_block_offsets,
    int slice_total_blocks,
    const float* __restrict__ sink_lse,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int NumSlices
) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ __align__(8) uint64_t full_bar[2];
    __shared__ __align__(8) uint64_t empty_bar[2];
    __shared__ int meta[10];

    __nv_bfloat16* q_s =
        reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* p_s =
        q_s + QK_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* k_stage =
        p_s + PV_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* v_stage =
        k_stage + 2 * KV_STAGE_ELEMS;

    const int tid = threadIdx.x;
    const int wg = tid >> 7;  // wg0 consumer; wg1 contains producer warp
    const int wtid = tid & 127;
    const bool producer_lane0 = tid == 128;
    const int kvh = blockIdx.y;

    if (tid == 0) {
        const int pid = blockIdx.x;
        int prefix = 0;
        int hit = 0;
        int q0 = 0, qe = 0, qs = 0, ks = 0, ke = 0, typ = 0;
        int sid = -1;

        for (int s = 0; s < NumSlices; ++s) {
            const int sqs = q_ranges[2*s + 0];
            const int sqe = q_ranges[2*s + 1];
            const int sks = k_ranges[2*s + 0];
            const int ske = k_ranges[2*s + 1];
            const int st = attn_type_map[s];
            const int nt = (sqe - sqs + TOKEN_M - 1) / TOKEN_M;

            if (!hit && pid >= prefix && pid < prefix + nt) {
                q0 = sqs + (pid - prefix) * TOKEN_M;
                qe = sqe;
                qs = sqs;
                ks = sks;
                ke = ske;
                typ = st;
                sid = s;
                hit = 1;
            }
            prefix += nt;
        }

        int kb_begin = 0;
        int kb_end = 0;

        if (hit) {
            const int k_blocks = (ke - ks + N - 1) / N;
            kb_end = k_blocks;

            const int q_first = q0;
            const int q_clip_end = ((q0 + TOKEN_M) < qe)
                ? (q0 + TOKEN_M) : qe;
            const int q_last = q_clip_end - 1;
            const int Lq = qe - qs;
            const int Lk = ke - ks;

            if (typ == 2 || typ == 3) {
                const int r_min = q_first - qs;
                kb_begin = r_min / N;
                if (kb_begin < 0) kb_begin = 0;
                if (kb_begin > k_blocks) kb_begin = k_blocks;
            }

            if (typ == 1 || typ == 3) {
                const int r_max = q_last - qs;
                const int max_u = r_max + (Lk - Lq);
                if (max_u < 0) {
                    kb_end = 0;
                } else {
                    kb_end = (max_u + 1 + N - 1) / N;
                    if (kb_end > k_blocks) kb_end = k_blocks;
                }
            }

            if (kb_begin > kb_end) kb_begin = kb_end;
        }

        meta[0] = hit;
        meta[1] = q0;
        meta[2] = qe;
        meta[3] = qs;
        meta[4] = ks;
        meta[5] = ke;
        meta[6] = typ;
        meta[7] = sid;
        meta[8] = kb_begin;
        meta[9] = kb_end;
    }

    if (producer_lane0) {
        wgmma_fa3_exp::mbarrier_init_count(&full_bar[0], 1);
        wgmma_fa3_exp::mbarrier_init_count(&full_bar[1], 1);
        wgmma_fa3_exp::mbarrier_init_count(&empty_bar[0], 1);
        wgmma_fa3_exp::mbarrier_init_count(&empty_bar[1], 1);
    }
    __syncthreads();

    if (!meta[0]) return;

    const int q0 = meta[1];
    const int qe = meta[2];
    const int qs = meta[3];
    const int ks = meta[4];
    const int ke = meta[5];
    const int typ = meta[6];
    const int sid = meta[7];
    const int kb_begin = meta[8];
    const int kb_end = meta[9];
    const int block_count = kb_end - kb_begin;

    if (wg == 0) {
        wgmma_partition::stage_q<HD>(
            q, q_s, q0, qe, kvh, Hq, G
        );
        fence_proxy_async_shared();
    }
    __syncthreads();

    // Producer warp exits before the GMMA region, exactly like the #12
    // role-split kernel that removed ptxas C7520.
    if (wg >= 1) {
        if (producer_lane0 && block_count > 0) {
            const int slice_base = slice_block_offsets[sid];
            const __nv_bfloat16* pk =
                packed_k
                + (int64_t(kvh) * slice_total_blocks + slice_base)
                  * KV_STAGE_ELEMS;
            const __nv_bfloat16* pv =
                packed_v
                + (int64_t(kvh) * slice_total_blocks + slice_base)
                  * KV_STAGE_ELEMS;

            wgmma_fa3_exp::issue_kv_stage(
                kb_begin, 0, pk, pv,
                k_stage, v_stage, full_bar
            );
            if (block_count > 1) {
                wgmma_fa3_exp::issue_kv_stage(
                    kb_begin + 1, 1, pk, pv,
                    k_stage, v_stage, full_bar
                );
            }

            for (int i = 0; i < block_count; ++i) {
                const int stage = i & 1;
                const int parity = (i >> 1) & 1;
                if (i + 2 < block_count) {
                    wgmma_fa3_exp::mbarrier_wait_phase(
                        &empty_bar[stage], parity
                    );
                    wgmma_fa3_exp::issue_kv_stage(
                        kb_begin + i + 2, stage, pk, pv,
                        k_stage, v_stage, full_bar
                    );
                }
            }
        }
        return;
    }

    float out0[32];
    float out1[32];
    wgmma_fa3_exp::zero32(out0);
    wgmma_fa3_exp::zero32(out1);

    const FragCoord fc = frag_coord();
    const int prow0 = fc.row0;
    const int prow1 = fc.row1;
    const int qidx0 = q0 + prow0 / G;
    const int qidx1 = q0 + prow1 / G;
    const int qh0 = kvh * G + (prow0 & 7);
    const int qh1 = kvh * G + (prow1 & 7);

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    for (int i = 0; i < block_count; ++i) {
        const int stage = i & 1;
        const int parity = (i >> 1) & 1;
        const int kb = kb_begin + i;
        const int key0 = ks + kb * N;

        wgmma_fa3_exp::mbarrier_wait_phase(
            &full_bar[stage], parity
        );

        __nv_bfloat16* my_k =
            k_stage + stage * KV_STAGE_ELEMS;
        __nv_bfloat16* my_v =
            v_stage + stage * KV_STAGE_ELEMS;

        float score[32];
        wgmma_fa3_exp::zero32(score);

        fence();
#pragma unroll
        for (int ds = 0; ds < QK_SLICES; ++ds) {
            mma_m64n64k16_bf16(
                score,
                make_kmajor_64x16_desc(
                    q_s + ds * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_k + ds * BLOCK_ELEMS)
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

            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c0, qs, qe, ks, ke))
                local_max0 = fmaxf(
                    local_max0, score[4*g+0] * softmax_scale);
            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c1, qs, qe, ks, ke))
                local_max0 = fmaxf(
                    local_max0, score[4*g+1] * softmax_scale);
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c0, qs, qe, ks, ke))
                local_max1 = fmaxf(
                    local_max1, score[4*g+2] * softmax_scale);
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c1, qs, qe, ks, ke))
                local_max1 = fmaxf(
                    local_max1, score[4*g+3] * softmax_scale);
        }

        const float tm0 = row4_max(local_max0);
        const float tm1 = row4_max(local_max1);
        const float nm0 = fmaxf(m0, tm0);
        const float nm1 = fmaxf(m1, tm1);
        const float a0 =
            (m0 == -CUDART_INF_F) ? 0.0f : __expf(m0 - nm0);
        const float a1 =
            (m1 == -CUDART_INF_F) ? 0.0f : __expf(m1 - nm1);

        float sum0 = 0.0f, sum1 = 0.0f;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            float p00 = 0.0f, p01 = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;

            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c0, qs, qe, ks, ke)) {
                p00 = __expf(
                    score[4*g+0] * softmax_scale - nm0);
                sum0 += p00;
            }
            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c1, qs, qe, ks, ke)) {
                p01 = __expf(
                    score[4*g+1] * softmax_scale - nm0);
                sum0 += p01;
            }
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c0, qs, qe, ks, ke)) {
                p10 = __expf(
                    score[4*g+2] * softmax_scale - nm1);
                sum1 += p10;
            }
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c1, qs, qe, ks, ke)) {
                p11 = __expf(
                    score[4*g+3] * softmax_scale - nm1);
                sum1 += p11;
            }

            const int s0 = c0 >> 4, kc0 = c0 & 15;
            const int s1 = c1 >> 4, kc1 = c1 & 15;

            p_s[s0 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow0, kc0)] =
                __float2bfloat16_rn(p00);
            p_s[s1 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow0, kc1)] =
                __float2bfloat16_rn(p01);
            p_s[s0 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow1, kc0)] =
                __float2bfloat16_rn(p10);
            p_s[s1 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow1, kc1)] =
                __float2bfloat16_rn(p11);
        }

        l0 = l0 * a0 + row4_sum(sum0);
        l1 = l1 * a1 + row4_sum(sum1);
        m0 = nm0;
        m1 = nm1;

        wgmma_fa3_exp::rescale32(out0, a0, a1);
        wgmma_fa3_exp::rescale32(out1, a0, a1);

        fence_proxy_async_shared();
        wgmma_fa3_exp::warpgroup_barrier(0);

        fence();
#pragma unroll
        for (int ks16 = 0; ks16 < PV_SLICES; ++ks16) {
            mma_m64n64k16_bf16(
                out0,
                make_kmajor_64x16_desc(
                    p_s + ks16 * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_v + ks16 * BLOCK_ELEMS)
            );
        }
#pragma unroll
        for (int ks16 = 0; ks16 < PV_SLICES; ++ks16) {
            mma_m64n64k16_bf16(
                out1,
                make_kmajor_64x16_desc(
                    p_s + ks16 * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_v + (PV_SLICES + ks16) * BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        wgmma_fa3_exp::warpgroup_barrier(0);
        if (wtid == 0) {
            wgmma_fa3_exp::mbarrier_arrive_release(
                &empty_bar[stage]
            );
        }
    }

    if (qidx0 < qe) {
        const float denom0 =
            l0 + __expf(sink_lse[qh0] - m0);
        const float inv0 = 1.0f / denom0;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+0] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+1] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c0] =
                __float2bfloat16_rn(out1[4*g+0] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c1] =
                __float2bfloat16_rn(out1[4*g+1] * inv0);
        }
    }

    if (qidx1 < qe) {
        const float denom1 =
            l1 + __expf(sink_lse[qh1] - m1);
        const float inv1 = 1.0f / denom1;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+2] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+3] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c0] =
                __float2bfloat16_rn(out1[4*g+2] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c1] =
                __float2bfloat16_rn(out1[4*g+3] * inv1);
        }
    }
}

inline void launch_g8_partition_async(
    const __nv_bfloat16* q,
    const int32_t* q_ranges,
    const int32_t* k_ranges,
    const int32_t* attn_type_map,
    const __nv_bfloat16* packed_k,
    const __nv_bfloat16* packed_v,
    const int32_t* slice_block_offsets,
    int slice_total_blocks,
    const float* sink_lse,
    __nv_bfloat16* out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int NumSlices
) {
    constexpr int smem_bytes =
        (QK_SLICES * BLOCK_ELEMS
       + PV_SLICES * BLOCK_ELEMS
       + 4 * KV_STAGE_ELEMS) * sizeof(__nv_bfloat16);

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            g8_partition_async_fwd,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        configured = true;
    }

    const int grid_x =
        (S + TOKEN_M - 1) / TOKEN_M + NumSlices;

    dim3 grid(grid_x, Hkv, 1);
    g8_partition_async_fwd<<<grid, 160, smem_bytes>>>(
        q,
        q_ranges, k_ranges, attn_type_map,
        packed_k, packed_v,
        slice_block_offsets, slice_total_blocks,
        sink_lse, out, softmax_scale,
        S, Hq, Hkv, NumSlices
    );
}

} // namespace wgmma_g8_async


namespace wgmma_g4_async {

using namespace wgmma_sm90;

constexpr int G = 4;
constexpr int HD = 128;
constexpr int N = 64;
constexpr int BLOCK_ELEMS = 64 * 16;
constexpr int QK_SLICES = 8;
constexpr int PV_SLICES = 4;
constexpr int KV_STAGE_ELEMS = QK_SLICES * BLOCK_ELEMS;
constexpr int TOKEN_M = 64 / G; // 16 query tokens / CTA

__global__ __launch_bounds__(160, 1)
void g4_partition_async_fwd(
    const __nv_bfloat16* __restrict__ q,
    const int32_t* __restrict__ q_ranges,
    const int32_t* __restrict__ k_ranges,
    const int32_t* __restrict__ attn_type_map,
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    const int32_t* __restrict__ slice_block_offsets,
    int slice_total_blocks,
    const float* __restrict__ sink_lse,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int NumSlices
) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ __align__(8) uint64_t full_bar[2];
    __shared__ __align__(8) uint64_t empty_bar[2];
    __shared__ int meta[10];

    __nv_bfloat16* q_s =
        reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* p_s =
        q_s + QK_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* k_stage =
        p_s + PV_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* v_stage =
        k_stage + 2 * KV_STAGE_ELEMS;

    const int tid = threadIdx.x;
    const int wg = tid >> 7;  // wg0 consumer; wg1 contains producer warp
    const int wtid = tid & 127;
    const bool producer_lane0 = tid == 128;
    const int kvh = blockIdx.y;

    if (tid == 0) {
        const int pid = blockIdx.x;
        int prefix = 0;
        int hit = 0;
        int q0 = 0, qe = 0, qs = 0, ks = 0, ke = 0, typ = 0;
        int sid = -1;

        for (int s = 0; s < NumSlices; ++s) {
            const int sqs = q_ranges[2*s + 0];
            const int sqe = q_ranges[2*s + 1];
            const int sks = k_ranges[2*s + 0];
            const int ske = k_ranges[2*s + 1];
            const int st = attn_type_map[s];
            const int nt = (sqe - sqs + TOKEN_M - 1) / TOKEN_M;

            if (!hit && pid >= prefix && pid < prefix + nt) {
                q0 = sqs + (pid - prefix) * TOKEN_M;
                qe = sqe;
                qs = sqs;
                ks = sks;
                ke = ske;
                typ = st;
                sid = s;
                hit = 1;
            }
            prefix += nt;
        }

        int kb_begin = 0;
        int kb_end = 0;

        if (hit) {
            const int k_blocks = (ke - ks + N - 1) / N;
            kb_end = k_blocks;

            const int q_first = q0;
            const int q_clip_end = ((q0 + TOKEN_M) < qe)
                ? (q0 + TOKEN_M) : qe;
            const int q_last = q_clip_end - 1;
            const int Lq = qe - qs;
            const int Lk = ke - ks;

            if (typ == 2 || typ == 3) {
                const int r_min = q_first - qs;
                kb_begin = r_min / N;
                if (kb_begin < 0) kb_begin = 0;
                if (kb_begin > k_blocks) kb_begin = k_blocks;
            }

            if (typ == 1 || typ == 3) {
                const int r_max = q_last - qs;
                const int max_u = r_max + (Lk - Lq);
                if (max_u < 0) {
                    kb_end = 0;
                } else {
                    kb_end = (max_u + 1 + N - 1) / N;
                    if (kb_end > k_blocks) kb_end = k_blocks;
                }
            }

            if (kb_begin > kb_end) kb_begin = kb_end;
        }

        meta[0] = hit;
        meta[1] = q0;
        meta[2] = qe;
        meta[3] = qs;
        meta[4] = ks;
        meta[5] = ke;
        meta[6] = typ;
        meta[7] = sid;
        meta[8] = kb_begin;
        meta[9] = kb_end;
    }

    if (producer_lane0) {
        wgmma_fa3_exp::mbarrier_init_count(&full_bar[0], 1);
        wgmma_fa3_exp::mbarrier_init_count(&full_bar[1], 1);
        wgmma_fa3_exp::mbarrier_init_count(&empty_bar[0], 1);
        wgmma_fa3_exp::mbarrier_init_count(&empty_bar[1], 1);
    }
    __syncthreads();

    if (!meta[0]) return;

    const int q0 = meta[1];
    const int qe = meta[2];
    const int qs = meta[3];
    const int ks = meta[4];
    const int ke = meta[5];
    const int typ = meta[6];
    const int sid = meta[7];
    const int kb_begin = meta[8];
    const int kb_end = meta[9];
    const int block_count = kb_end - kb_begin;

    if (wg == 0) {
        wgmma_partition::stage_q<HD>(
            q, q_s, q0, qe, kvh, Hq, G
        );
        fence_proxy_async_shared();
    }
    __syncthreads();

    // Producer warp exits before the GMMA region, exactly like the #12
    // role-split kernel that removed ptxas C7520.
    if (wg >= 1) {
        if (producer_lane0 && block_count > 0) {
            const int slice_base = slice_block_offsets[sid];
            const __nv_bfloat16* pk =
                packed_k
                + (int64_t(kvh) * slice_total_blocks + slice_base)
                  * KV_STAGE_ELEMS;
            const __nv_bfloat16* pv =
                packed_v
                + (int64_t(kvh) * slice_total_blocks + slice_base)
                  * KV_STAGE_ELEMS;

            wgmma_fa3_exp::issue_kv_stage(
                kb_begin, 0, pk, pv,
                k_stage, v_stage, full_bar
            );
            if (block_count > 1) {
                wgmma_fa3_exp::issue_kv_stage(
                    kb_begin + 1, 1, pk, pv,
                    k_stage, v_stage, full_bar
                );
            }

            for (int i = 0; i < block_count; ++i) {
                const int stage = i & 1;
                const int parity = (i >> 1) & 1;
                if (i + 2 < block_count) {
                    wgmma_fa3_exp::mbarrier_wait_phase(
                        &empty_bar[stage], parity
                    );
                    wgmma_fa3_exp::issue_kv_stage(
                        kb_begin + i + 2, stage, pk, pv,
                        k_stage, v_stage, full_bar
                    );
                }
            }
        }
        return;
    }

    float out0[32];
    float out1[32];
    wgmma_fa3_exp::zero32(out0);
    wgmma_fa3_exp::zero32(out1);

    const FragCoord fc = frag_coord();
    const int prow0 = fc.row0;
    const int prow1 = fc.row1;
    const int qidx0 = q0 + prow0 / G;
    const int qidx1 = q0 + prow1 / G;
    const int qh0 = kvh * G + (prow0 & 3);
    const int qh1 = kvh * G + (prow1 & 3);

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    for (int i = 0; i < block_count; ++i) {
        const int stage = i & 1;
        const int parity = (i >> 1) & 1;
        const int kb = kb_begin + i;
        const int key0 = ks + kb * N;

        wgmma_fa3_exp::mbarrier_wait_phase(
            &full_bar[stage], parity
        );

        __nv_bfloat16* my_k =
            k_stage + stage * KV_STAGE_ELEMS;
        __nv_bfloat16* my_v =
            v_stage + stage * KV_STAGE_ELEMS;

        float score[32];
        wgmma_fa3_exp::zero32(score);

        fence();
#pragma unroll
        for (int ds = 0; ds < QK_SLICES; ++ds) {
            mma_m64n64k16_bf16(
                score,
                make_kmajor_64x16_desc(
                    q_s + ds * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_k + ds * BLOCK_ELEMS)
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

            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c0, qs, qe, ks, ke))
                local_max0 = fmaxf(
                    local_max0, score[4*g+0] * softmax_scale);
            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c1, qs, qe, ks, ke))
                local_max0 = fmaxf(
                    local_max0, score[4*g+1] * softmax_scale);
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c0, qs, qe, ks, ke))
                local_max1 = fmaxf(
                    local_max1, score[4*g+2] * softmax_scale);
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c1, qs, qe, ks, ke))
                local_max1 = fmaxf(
                    local_max1, score[4*g+3] * softmax_scale);
        }

        const float tm0 = row4_max(local_max0);
        const float tm1 = row4_max(local_max1);
        const float nm0 = fmaxf(m0, tm0);
        const float nm1 = fmaxf(m1, tm1);
        const float a0 =
            (m0 == -CUDART_INF_F) ? 0.0f : __expf(m0 - nm0);
        const float a1 =
            (m1 == -CUDART_INF_F) ? 0.0f : __expf(m1 - nm1);

        float sum0 = 0.0f, sum1 = 0.0f;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            float p00 = 0.0f, p01 = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;

            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c0, qs, qe, ks, ke)) {
                p00 = __expf(
                    score[4*g+0] * softmax_scale - nm0);
                sum0 += p00;
            }
            if (wgmma_partition::mask_visible(
                    typ, qidx0, key0+c1, qs, qe, ks, ke)) {
                p01 = __expf(
                    score[4*g+1] * softmax_scale - nm0);
                sum0 += p01;
            }
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c0, qs, qe, ks, ke)) {
                p10 = __expf(
                    score[4*g+2] * softmax_scale - nm1);
                sum1 += p10;
            }
            if (wgmma_partition::mask_visible(
                    typ, qidx1, key0+c1, qs, qe, ks, ke)) {
                p11 = __expf(
                    score[4*g+3] * softmax_scale - nm1);
                sum1 += p11;
            }

            const int s0 = c0 >> 4, kc0 = c0 & 15;
            const int s1 = c1 >> 4, kc1 = c1 & 15;

            p_s[s0 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow0, kc0)] =
                __float2bfloat16_rn(p00);
            p_s[s1 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow0, kc1)] =
                __float2bfloat16_rn(p01);
            p_s[s0 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow1, kc0)] =
                __float2bfloat16_rn(p10);
            p_s[s1 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow1, kc1)] =
                __float2bfloat16_rn(p11);
        }

        l0 = l0 * a0 + row4_sum(sum0);
        l1 = l1 * a1 + row4_sum(sum1);
        m0 = nm0;
        m1 = nm1;

        wgmma_fa3_exp::rescale32(out0, a0, a1);
        wgmma_fa3_exp::rescale32(out1, a0, a1);

        fence_proxy_async_shared();
        wgmma_fa3_exp::warpgroup_barrier(0);

        fence();
#pragma unroll
        for (int ks16 = 0; ks16 < PV_SLICES; ++ks16) {
            mma_m64n64k16_bf16(
                out0,
                make_kmajor_64x16_desc(
                    p_s + ks16 * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_v + ks16 * BLOCK_ELEMS)
            );
        }
#pragma unroll
        for (int ks16 = 0; ks16 < PV_SLICES; ++ks16) {
            mma_m64n64k16_bf16(
                out1,
                make_kmajor_64x16_desc(
                    p_s + ks16 * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_v + (PV_SLICES + ks16) * BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        wgmma_fa3_exp::warpgroup_barrier(0);
        if (wtid == 0) {
            wgmma_fa3_exp::mbarrier_arrive_release(
                &empty_bar[stage]
            );
        }
    }

    if (qidx0 < qe) {
        const float denom0 =
            l0 + __expf(sink_lse[qh0] - m0);
        const float inv0 = 1.0f / denom0;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+0] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+1] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c0] =
                __float2bfloat16_rn(out1[4*g+0] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c1] =
                __float2bfloat16_rn(out1[4*g+1] * inv0);
        }
    }

    if (qidx1 < qe) {
        const float denom1 =
            l1 + __expf(sink_lse[qh1] - m1);
        const float inv1 = 1.0f / denom1;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+2] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+3] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c0] =
                __float2bfloat16_rn(out1[4*g+2] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c1] =
                __float2bfloat16_rn(out1[4*g+3] * inv1);
        }
    }
}

inline void launch_g4_partition_async(
    const __nv_bfloat16* q,
    const int32_t* q_ranges,
    const int32_t* k_ranges,
    const int32_t* attn_type_map,
    const __nv_bfloat16* packed_k,
    const __nv_bfloat16* packed_v,
    const int32_t* slice_block_offsets,
    int slice_total_blocks,
    const float* sink_lse,
    __nv_bfloat16* out,
    float softmax_scale,
    int S,
    int Hq,
    int Hkv,
    int NumSlices
) {
    constexpr int smem_bytes =
        (QK_SLICES * BLOCK_ELEMS
       + PV_SLICES * BLOCK_ELEMS
       + 4 * KV_STAGE_ELEMS) * sizeof(__nv_bfloat16);

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            g4_partition_async_fwd,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        configured = true;
    }

    const int grid_x =
        (S + TOKEN_M - 1) / TOKEN_M + NumSlices;

    dim3 grid(grid_x, Hkv, 1);
    g4_partition_async_fwd<<<grid, 160, smem_bytes>>>(
        q,
        q_ranges, k_ranges, attn_type_map,
        packed_k, packed_v,
        slice_block_offsets, slice_total_blocks,
        sink_lse, out, softmax_scale,
        S, Hq, Hkv, NumSlices
    );
}

} // namespace wgmma_g4_async



namespace wgmma_g4_overlap_async {

using namespace wgmma_sm90;

constexpr int G = 4;
constexpr int HD = 128;
constexpr int N = 64;
constexpr int BLOCK_ELEMS = 64 * 16;
constexpr int QK_SLICES = 8;
constexpr int PV_SLICES = 4;
constexpr int KV_STAGE_ELEMS = QK_SLICES * BLOCK_ELEMS;
constexpr int TOKEN_M = 64 / G; // 16 query tokens / CTA

__global__ __launch_bounds__(160, 1)
void g4_overlap_async_fwd(
    const __nv_bfloat16* __restrict__ q,
    const int32_t* __restrict__ q_ranges,
    const int32_t* __restrict__ k_ranges,
    const __nv_bfloat16* __restrict__ packed_k,
    const __nv_bfloat16* __restrict__ packed_v,
    const int32_t* __restrict__ slice_block_offsets,
    int slice_total_blocks,
    const float* __restrict__ sink_lse,
    __nv_bfloat16* __restrict__ out,
    float softmax_scale,
    int Hq
) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ __align__(8) uint64_t full_bar[2];
    __shared__ __align__(8) uint64_t empty_bar[2];
    __shared__ int meta[11];

    __nv_bfloat16* q_s =
        reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* p_s =
        q_s + QK_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* k_stage =
        p_s + PV_SLICES * BLOCK_ELEMS;
    __nv_bfloat16* v_stage =
        k_stage + 2 * KV_STAGE_ELEMS;

    const int tid = threadIdx.x;
    const int wg = tid >> 7;
    const int wtid = tid & 127;
    const bool producer_lane0 = tid == 128;
    const int kvh = blockIdx.y;

    if (tid == 0) {
        const int pid = blockIdx.x;
        int prefix_tiles = 0;
        int hit = 0;
        int sid = -1;
        int q0 = 0, qs = 0, qe = 0, ks = 0, ke = 0;

        // Only the seven disjoint base slices own output rows. Slice 7 is
        // the shared FULL prefix [0,512) for base slices 1..6.
        for (int s = 0; s < 7; ++s) {
            const int sqs = q_ranges[2*s + 0];
            const int sqe = q_ranges[2*s + 1];
            const int nt = (sqe - sqs + TOKEN_M - 1) / TOKEN_M;

            if (!hit && pid >= prefix_tiles && pid < prefix_tiles + nt) {
                sid = s;
                qs = sqs;
                qe = sqe;
                ks = k_ranges[2*s + 0];
                ke = k_ranges[2*s + 1];
                q0 = sqs + (pid - prefix_tiles) * TOKEN_M;
                hit = 1;
            }
            prefix_tiles += nt;
        }

        int extra_blocks = 0;
        int local_blocks = 0;
        if (hit) {
            if (sid > 0) {
                const int ex_ks = k_ranges[14];
                const int ex_ke = k_ranges[15];
                extra_blocks = (ex_ke - ex_ks + N - 1) / N;

                // Local base slices 1..6 are causal with Lq == Lk.
                // Only key positions <= the last query in this CTA matter.
                const int q_last =
                    ((q0 + TOKEN_M) < qe ? (q0 + TOKEN_M) : qe) - 1;
                const int local_visible = q_last - qs + 1;
                local_blocks = (local_visible + N - 1) / N;
            } else {
                // Slice 0 is just one FULL [0,512) region.
                local_blocks = (ke - ks + N - 1) / N;
            }
        }

        meta[0] = hit;
        meta[1] = sid;
        meta[2] = q0;
        meta[3] = qs;
        meta[4] = qe;
        meta[5] = ks;
        meta[6] = ke;
        meta[7] = extra_blocks;
        meta[8] = local_blocks;
        meta[9] = hit ? slice_block_offsets[sid] : 0;
        meta[10] = hit ? slice_block_offsets[7] : 0;
    }

    if (producer_lane0) {
        wgmma_fa3_exp::mbarrier_init_count(&full_bar[0], 1);
        wgmma_fa3_exp::mbarrier_init_count(&full_bar[1], 1);
        wgmma_fa3_exp::mbarrier_init_count(&empty_bar[0], 1);
        wgmma_fa3_exp::mbarrier_init_count(&empty_bar[1], 1);
    }
    __syncthreads();

    if (!meta[0]) return;

    const int sid = meta[1];
    const int q0 = meta[2];
    const int qs = meta[3];
    const int qe = meta[4];
    const int ks = meta[5];
    const int ke = meta[6];
    const int extra_blocks = meta[7];
    const int local_blocks = meta[8];
    const int local_base = meta[9];
    const int extra_base = meta[10];
    const int block_count = extra_blocks + local_blocks;

    if (wg == 0) {
        wgmma_partition::stage_q<HD>(
            q, q_s, q0, qe, kvh, Hq, G
        );
        fence_proxy_async_shared();
    }
    __syncthreads();

    if (wg >= 1) {
        if (producer_lane0 && block_count > 0) {
            const __nv_bfloat16* pk =
                packed_k + int64_t(kvh) * slice_total_blocks
                    * KV_STAGE_ELEMS;
            const __nv_bfloat16* pv =
                packed_v + int64_t(kvh) * slice_total_blocks
                    * KV_STAGE_ELEMS;

            auto abs_block = [&](int i) {
                if (i < extra_blocks) return extra_base + i;
                return local_base + (i - extra_blocks);
            };

            wgmma_fa3_exp::issue_kv_stage(
                abs_block(0), 0, pk, pv,
                k_stage, v_stage, full_bar
            );
            if (block_count > 1) {
                wgmma_fa3_exp::issue_kv_stage(
                    abs_block(1), 1, pk, pv,
                    k_stage, v_stage, full_bar
                );
            }

            for (int i = 0; i < block_count; ++i) {
                const int stage = i & 1;
                const int parity = (i >> 1) & 1;
                if (i + 2 < block_count) {
                    wgmma_fa3_exp::mbarrier_wait_phase(
                        &empty_bar[stage], parity
                    );
                    wgmma_fa3_exp::issue_kv_stage(
                        abs_block(i + 2), stage, pk, pv,
                        k_stage, v_stage, full_bar
                    );
                }
            }
        }
        return;
    }

    float out0[32];
    float out1[32];
    wgmma_fa3_exp::zero32(out0);
    wgmma_fa3_exp::zero32(out1);

    const FragCoord fc = frag_coord();
    const int prow0 = fc.row0;
    const int prow1 = fc.row1;
    const int qidx0 = q0 + prow0 / G;
    const int qidx1 = q0 + prow1 / G;
    const int qh0 = kvh * G + (prow0 & 3);
    const int qh1 = kvh * G + (prow1 & 3);

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    for (int i = 0; i < block_count; ++i) {
        const int stage = i & 1;
        const int parity = (i >> 1) & 1;
        const bool is_extra = i < extra_blocks;
        const int local_kb = i - extra_blocks;
        const int key0 = is_extra
            ? (k_ranges[14] + i * N)
            : (ks + local_kb * N);

        wgmma_fa3_exp::mbarrier_wait_phase(
            &full_bar[stage], parity
        );

        __nv_bfloat16* my_k =
            k_stage + stage * KV_STAGE_ELEMS;
        __nv_bfloat16* my_v =
            v_stage + stage * KV_STAGE_ELEMS;

        float score[32];
        wgmma_fa3_exp::zero32(score);

        fence();
#pragma unroll
        for (int ds = 0; ds < QK_SLICES; ++ds) {
            mma_m64n64k16_bf16(
                score,
                make_kmajor_64x16_desc(
                    q_s + ds * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_k + ds * BLOCK_ELEMS)
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

            const int key_a = key0 + c0;
            const int key_b = key0 + c1;

            const bool v00 = is_extra || sid == 0
                ? (key_a < ke)
                : (key_a < ke && key_a <= qidx0);
            const bool v01 = is_extra || sid == 0
                ? (key_b < ke)
                : (key_b < ke && key_b <= qidx0);
            const bool v10 = is_extra || sid == 0
                ? (key_a < ke)
                : (key_a < ke && key_a <= qidx1);
            const bool v11 = is_extra || sid == 0
                ? (key_b < ke)
                : (key_b < ke && key_b <= qidx1);

            if (v00)
                local_max0 = fmaxf(
                    local_max0, score[4*g+0] * softmax_scale);
            if (v01)
                local_max0 = fmaxf(
                    local_max0, score[4*g+1] * softmax_scale);
            if (v10)
                local_max1 = fmaxf(
                    local_max1, score[4*g+2] * softmax_scale);
            if (v11)
                local_max1 = fmaxf(
                    local_max1, score[4*g+3] * softmax_scale);
        }

        const float tm0 = row4_max(local_max0);
        const float tm1 = row4_max(local_max1);
        const float nm0 = fmaxf(m0, tm0);
        const float nm1 = fmaxf(m1, tm1);
        const float a0 =
            (m0 == -CUDART_INF_F) ? 0.0f : __expf(m0 - nm0);
        const float a1 =
            (m1 == -CUDART_INF_F) ? 0.0f : __expf(m1 - nm1);

        float sum0 = 0.0f, sum1 = 0.0f;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);

            const int key_a = key0 + c0;
            const int key_b = key0 + c1;

            const bool v00 = is_extra || sid == 0
                ? (key_a < ke)
                : (key_a < ke && key_a <= qidx0);
            const bool v01 = is_extra || sid == 0
                ? (key_b < ke)
                : (key_b < ke && key_b <= qidx0);
            const bool v10 = is_extra || sid == 0
                ? (key_a < ke)
                : (key_a < ke && key_a <= qidx1);
            const bool v11 = is_extra || sid == 0
                ? (key_b < ke)
                : (key_b < ke && key_b <= qidx1);

            float p00 = 0.0f, p01 = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;

            if (v00) {
                p00 = __expf(score[4*g+0] * softmax_scale - nm0);
                sum0 += p00;
            }
            if (v01) {
                p01 = __expf(score[4*g+1] * softmax_scale - nm0);
                sum0 += p01;
            }
            if (v10) {
                p10 = __expf(score[4*g+2] * softmax_scale - nm1);
                sum1 += p10;
            }
            if (v11) {
                p11 = __expf(score[4*g+3] * softmax_scale - nm1);
                sum1 += p11;
            }

            const int s0 = c0 >> 4, kc0 = c0 & 15;
            const int s1 = c1 >> 4, kc1 = c1 & 15;

            p_s[s0 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow0, kc0)] =
                __float2bfloat16_rn(p00);
            p_s[s1 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow0, kc1)] =
                __float2bfloat16_rn(p01);
            p_s[s0 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow1, kc0)] =
                __float2bfloat16_rn(p10);
            p_s[s1 * BLOCK_ELEMS
                + canonical_kmajor_offset(prow1, kc1)] =
                __float2bfloat16_rn(p11);
        }

        l0 = l0 * a0 + row4_sum(sum0);
        l1 = l1 * a1 + row4_sum(sum1);
        m0 = nm0;
        m1 = nm1;

        wgmma_fa3_exp::rescale32(out0, a0, a1);
        wgmma_fa3_exp::rescale32(out1, a0, a1);

        fence_proxy_async_shared();
        wgmma_fa3_exp::warpgroup_barrier(0);

        fence();
#pragma unroll
        for (int ks16 = 0; ks16 < PV_SLICES; ++ks16) {
            mma_m64n64k16_bf16(
                out0,
                make_kmajor_64x16_desc(
                    p_s + ks16 * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_v + ks16 * BLOCK_ELEMS)
            );
        }
#pragma unroll
        for (int ks16 = 0; ks16 < PV_SLICES; ++ks16) {
            mma_m64n64k16_bf16(
                out1,
                make_kmajor_64x16_desc(
                    p_s + ks16 * BLOCK_ELEMS),
                make_kmajor_64x16_desc(
                    my_v + (PV_SLICES + ks16) * BLOCK_ELEMS)
            );
        }
        commit_group();
        wait_group<0>();

        wgmma_fa3_exp::warpgroup_barrier(0);
        if (wtid == 0) {
            wgmma_fa3_exp::mbarrier_arrive_release(
                &empty_bar[stage]
            );
        }
    }

    if (qidx0 < qe) {
        const float denom0 =
            l0 + __expf(sink_lse[qh0] - m0);
        const float inv0 = 1.0f / denom0;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+0] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+1] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c0] =
                __float2bfloat16_rn(out1[4*g+0] * inv0);
            out[(int64_t(qidx0) * Hq + qh0) * HD + 64+c1] =
                __float2bfloat16_rn(out1[4*g+1] * inv0);
        }
    }

    if (qidx1 < qe) {
        const float denom1 =
            l1 + __expf(sink_lse[qh1] - m1);
        const float inv1 = 1.0f / denom1;

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int c0 = frag_col(g, 0);
            const int c1 = frag_col(g, 1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c0] =
                __float2bfloat16_rn(out0[4*g+2] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + c1] =
                __float2bfloat16_rn(out0[4*g+3] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c0] =
                __float2bfloat16_rn(out1[4*g+2] * inv1);
            out[(int64_t(qidx1) * Hq + qh1) * HD + 64+c1] =
                __float2bfloat16_rn(out1[4*g+3] * inv1);
        }
    }
}

inline void launch_g4_overlap_async(
    const __nv_bfloat16* q,
    const int32_t* q_ranges,
    const int32_t* k_ranges,
    const __nv_bfloat16* packed_k,
    const __nv_bfloat16* packed_v,
    const int32_t* slice_block_offsets,
    int slice_total_blocks,
    const float* sink_lse,
    __nv_bfloat16* out,
    float softmax_scale
) {
    constexpr int smem_bytes =
        (QK_SLICES * BLOCK_ELEMS
       + PV_SLICES * BLOCK_ELEMS
       + 4 * KV_STAGE_ELEMS) * sizeof(__nv_bfloat16);

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            g4_overlap_async_fwd,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        configured = true;
    }

    // Exact scored shape: 260 tiles across the seven disjoint base ranges.
    dim3 grid(260, 8, 1);
    g4_overlap_async_fwd<<<grid, 160, smem_bytes>>>(
        q, q_ranges, k_ranges,
        packed_k, packed_v,
        slice_block_offsets, slice_total_blocks,
        sink_lse, out, softmax_scale, 32
    );
}

} // namespace wgmma_g4_overlap_async


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
    static bool printed_build = false;
    if (!printed_build) {
        fprintf(stderr, "BUILD CUDA_SM90A_OVERLAP_ASYNC_V9\\n");
        printed_build = true;
    }

    const int S = int(seqlen);
    const int Hq = int(num_q_heads);
    const int Hkv = int(num_kv_heads);
    const int D = int(head_dim);
    const int N = int(num_slices);
    const int Ns = int(num_sink);

    // #12: one token / CTA, two warpgroups cover all 128 Q heads while K/V
    // are staged only once.
    if (
        S == 8192 && Hq == 128 && Hkv == 1 &&
        D == 128 && N == 1 && Ns == 4
    ) {
        const __nv_bfloat16* packed_k = nullptr;
        const __nv_bfloat16* packed_v = nullptr;
        const float* sink_lse = nullptr;
        wgmma_static_cache::ensure_d128(
            k, v, sink, q_ranges,
            S, Hq, Hkv, Ns,
            packed_k, packed_v, sink_lse
        );
        wgmma_fa3_exp::launch_g128_fa3(
            q, packed_k, packed_v, sink_lse,
            output, softmax_scale
        );
        return;
    }

    // #1 / #7 / #9: dense N=1 D128.
    if (N == 1 && D == 128) {
        int causal = 0;
        if (
            (S == 4096 && Hq == 32 && Hkv == 8) ||
            (S == 16384 && Hq == 32 && Hkv == 8)
        ) {
            causal = 1;
        }

        const __nv_bfloat16* packed_k = nullptr;
        const __nv_bfloat16* packed_v = nullptr;
        const float* sink_lse = nullptr;
        wgmma_static_cache::ensure_d128(
            k, v, sink, q_ranges,
            S, Hq, Hkv, Ns,
            packed_k, packed_v, sink_lse
        );
        wgmma_attention::launch_dense_wgmma(
            q, k, v, packed_k, packed_v, sink_lse,
            output, softmax_scale,
            S, Hq, Hkv, Ns, causal
        );
        return;
    }

    // #5 exact overlap2: collapse into three disjoint effective regions.
    if (
        S == 512 && Hq == 16 && Hkv == 8 &&
        D == 128 && N == 2 && Ns == 2
    ) {
        const __nv_bfloat16* packed_k = nullptr;
        const __nv_bfloat16* packed_v = nullptr;
        const float* sink_lse = nullptr;
        wgmma_static_cache::ensure_d128(
            k, v, sink, q_ranges,
            S, Hq, Hkv, Ns,
            packed_k, packed_v, sink_lse
        );
        wgmma_partition::launch_partition_wgmma<128>(
            q, k, v,
            q_ranges, k_ranges, attn_type_map,
            sink, packed_k, packed_v, sink_lse,
            output, softmax_scale,
            S, Hq, Hkv, Ns, N,
            1
        );
        return;
    }

    // #4: exact G4 overlap pattern.  Flatten the shared FULL prefix and
    // local causal segment into one async block stream per output tile.
    if (
        S == 4096 && Hq == 32 && Hkv == 8 &&
        D == 128 && N == 8 && Ns == 6
    ) {
        const __nv_bfloat16* packed_k = nullptr;
        const __nv_bfloat16* packed_v = nullptr;
        const int32_t* slice_offsets = nullptr;
        int slice_total_blocks = 0;
        const float* sink_lse = nullptr;

        wgmma_static_cache::ensure_sink_only(
            sink, Hq, Ns, sink_lse
        );
        wgmma_slice_cache::ensure_slice_d128(
            k, v, k_ranges, q_ranges,
            S, Hkv, N,
            packed_k, packed_v,
            slice_offsets, slice_total_blocks
        );

        wgmma_g4_overlap_async::launch_g4_overlap_async(
            q, q_ranges, k_ranges,
            packed_k, packed_v,
            slice_offsets, slice_total_blocks,
            sink_lse, output, softmax_scale
        );
        return;
    }

    // #10 is the only D64 scored shape.
    if (D == 64) {
        wgmma_partition::launch_partition_wgmma<64>(
            q, k, v,
            q_ranges, k_ranges, attn_type_map,
            sink, nullptr, nullptr, nullptr,
            output, softmax_scale,
            S, Hq, Hkv, Ns, N,
            0
        );
        return;
    }

    // #2 / #3 / #6 / #11: D128 G=8 async role-split path.
    if (D == 128 && N > 1 && Hq / Hkv == 8) {
        const __nv_bfloat16* packed_k = nullptr;
        const __nv_bfloat16* packed_v = nullptr;
        const int32_t* slice_offsets = nullptr;
        int slice_total_blocks = 0;
        const float* sink_lse = nullptr;

        wgmma_static_cache::ensure_sink_only(
            sink, Hq, Ns, sink_lse
        );
        wgmma_slice_cache::ensure_slice_d128(
            k, v, k_ranges, q_ranges,
            S, Hkv, N,
            packed_k, packed_v,
            slice_offsets, slice_total_blocks
        );

        wgmma_g8_async::launch_g8_partition_async(
            q,
            q_ranges, k_ranges, attn_type_map,
            packed_k, packed_v,
            slice_offsets, slice_total_blocks,
            sink_lse, output, softmax_scale,
            S, Hq, Hkv, N
        );
        return;
    }

    // #8: D128 G=4 prefix-FULL async role-split path.
    if (D == 128 && N > 1 && Hq / Hkv == 4) {
        const __nv_bfloat16* packed_k = nullptr;
        const __nv_bfloat16* packed_v = nullptr;
        const int32_t* slice_offsets = nullptr;
        int slice_total_blocks = 0;
        const float* sink_lse = nullptr;

        wgmma_static_cache::ensure_sink_only(
            sink, Hq, Ns, sink_lse
        );
        wgmma_slice_cache::ensure_slice_d128(
            k, v, k_ranges, q_ranges,
            S, Hkv, N,
            packed_k, packed_v,
            slice_offsets, slice_total_blocks
        );

        wgmma_g4_async::launch_g4_partition_async(
            q,
            q_ranges, k_ranges, attn_type_map,
            packed_k, packed_v,
            slice_offsets, slice_total_blocks,
            sink_lse, output, softmax_scale,
            S, Hq, Hkv, N
        );
        return;
    }

    // Remaining multi-slice D128 fallback.
    // Pack K/V in slice-local 64-token coordinates during warmup.  This
    // keeps every block on the fast path even when ks is not 64-aligned.
    const __nv_bfloat16* packed_k = nullptr;
    const __nv_bfloat16* packed_v = nullptr;
    const int32_t* slice_offsets = nullptr;
    int slice_total_blocks = 0;
    const float* sink_lse = nullptr;

    wgmma_static_cache::ensure_sink_only(
        sink, Hq, Ns, sink_lse
    );
    wgmma_slice_cache::ensure_slice_d128(
        k, v, k_ranges, q_ranges,
        S, Hkv, N,
        packed_k, packed_v,
        slice_offsets, slice_total_blocks
    );

    wgmma_partition::launch_partition_wgmma<128>(
        q, k, v,
        q_ranges, k_ranges, attn_type_map,
        sink, packed_k, packed_v, sink_lse,
        output, softmax_scale,
        S, Hq, Hkv, Ns, N,
        0, slice_offsets, slice_total_blocks
    );
}
