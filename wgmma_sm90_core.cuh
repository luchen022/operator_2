#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

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
    const int warp = threadIdx.x >> 5;
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
