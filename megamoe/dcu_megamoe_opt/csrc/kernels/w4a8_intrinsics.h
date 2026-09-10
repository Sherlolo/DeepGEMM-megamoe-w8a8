#pragma once

#include <stdint.h>
#include <type_traits>

#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

template <typename scalar_t, int len>
using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

using bhalf_t = __hip_bfloat16;
using half_t = __half;
using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using vec8_bf16 =
    __attribute__((__vector_size__(8 * sizeof(unsigned short)))) unsigned short;
using vec2_bf16 =
    __attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short;
using vec8_fp16 =
    __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using vec2_fp16 =
    __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;

template <typename scalar_t>
union vec_element_8 {};

template <>
union vec_element_8<bhalf_t> {
  vec8_bf16 data;
};

template <>
union vec_element_8<__half> {
  vec8_fp16 data;
};

template <typename scalar_t>
union vec_element_2 {};

template <>
union vec_element_2<bhalf_t> {
  vec2_bf16 data;
};

template <>
union vec_element_2<__half> {
  vec2_fp16 data;
};

template <typename Element, size_t len>
union union_vec_opt {
  int8_t int8_array[len * sizeof(Element)];
  Element scalar_array[len];
  vec<Element, 2> scalar2_array[len / 2];
  int int_array[len * sizeof(Element) / 4];
  float float_array[len * sizeof(Element) / 4];
  int32_t uint_array[len * sizeof(Element) / 4];
  int64_t uint64_array[len * sizeof(Element) / 8];
  vec<int8_t, 8> int8t_array[len * sizeof(Element) / 8];
  vec<int, 2> int2_array[len * sizeof(Element) / 8];
  vec<int, 4> int4_array[len * sizeof(Element) / 16];
  vec<float, 4> float4_array[len * sizeof(Element) / 16];
};

#define vmcnt_wait(X)                         \
  __builtin_amdgcn_sched_barrier(0);          \
  asm volatile("s_waitcnt vmcnt(%0)\n\t"      \
               "s_barrier\n"                 \
               :                              \
               : "i"(X)                       \
               :);                            \
  __builtin_amdgcn_sched_barrier(0);

#define vmcnt_only_wait(X)                    \
  __builtin_amdgcn_sched_barrier(0);          \
  asm volatile("s_waitcnt vmcnt(%0)\n\t"      \
               :                              \
               : "i"(X)                       \
               :);                            \
  __builtin_amdgcn_sched_barrier(0);

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dword_lds(
    DataType* const shared_addr, const vec4_uint global_addr,
    const int& lds_offset, const int& gvOffset_s, const int& gvOffset_v) {
  int ldsAddrPerWave =
      reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n"
               :
               : "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr),
                 "s"(offset_s)
               :);
}

template <const int stride, typename T>
__device__ __forceinline__ vec4_uint tcp_cache_swizzle_func_b8(const T* ptr) {
  vec4_uint res;
  *reinterpret_cast<uint64_t*>(&res) = reinterpret_cast<uint64_t>(ptr);
  res[1] += 0x40000000 | (stride << 16);
  res[2] = 0x80000000;
  res[3] = 0x00020000;
  return res;
}

// Same direct-to-LDS instruction/layout as DeepGEMM's W4A8 nwave loader.
template <class T>
__forceinline__ __device__ void load_a_dwordx4_lds(
    T* lds, const vec4_uint resource, int lds_byte_offset,
    int global_scalar_offset, int global_vector_offset) {
  const int lds_address = static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_byte_offset;
  asm volatile("s_mov_b32 m0, %1\n\t"
               "buffer_load_dwordx4 %0, %2, %3, offen offset:0, lds\n\t"
               :
               : "v"(global_vector_offset), "s"(lds_address), "s"(resource),
                 "s"(global_scalar_offset)
               : "memory");
}

template <bool ForceUniform = true, typename T>
__forceinline__ __device__ void buffer_load_reg_dwordx4_w4a8(
    const T* ptr, vec<int, 4>& rsrc, const int s_offset, int v_offset) {
  const uint64_t address = reinterpret_cast<uint64_t>(ptr);
  intx4 global_ptr;
  if constexpr (ForceUniform) {
    // Narrow tile specializations need an explicit scalar descriptor.
    global_ptr[0] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address));
    global_ptr[1] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address >> 32));
  } else {
    *reinterpret_cast<uint64_t*>(&global_ptr) = address;
  }
  global_ptr[1] += 0x00800000;
  global_ptr[2] = 0x80000000;
  global_ptr[3] = 0x00020000;

  v_offset = v_offset * sizeof(T);
  const int s_offset_bytes = s_offset * sizeof(T);

  asm volatile("buffer_load_dwordx4 %0, %1, %2 ,%3 offen  offset:0 \n"
               : "=v"(rsrc)
               : "v"(v_offset), "s"(global_ptr), "s"(s_offset_bytes)
               : "memory");
}

template <class Element>
__device__ vec<int, 4> mmac(const vec<Element, 8>& v1,
                            const vec<Element, 8>& v2,
                            vec<int, 4>& v3) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__) || \
    defined(__gfx938__)
  v3 = __builtin_hcu_mmac_i32_16x16x32_i8(v1, v2, v3);
#endif
  return v3;
}

template <typename T>
static __device__ inline T b32_to_b16(float f) {
  if constexpr (std::is_same_v<T, __hip_bfloat16>) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__)
    union {
      float f32;
      uint32_t u32;
    } in = {f};
    uint32_t rounded = in.u32 + 0x7fff + ((in.u32 >> 16) & 1);
    if (f != f) {
      rounded = 0x7fff0000;
    }
    union {
      uint16_t u16;
      __hip_bfloat16 bf16;
    } out = {uint16_t(rounded >> 16)};
    return out.bf16;
#elif defined(__gfx938__)
    __builtin_amdgcn_sched_barrier(0);
    __hip_bfloat16 res;
    asm volatile("v_cvt_bf16_f32  %0, %1 \n\t" : "=v"(res) : "v"(f));
    __builtin_amdgcn_sched_barrier(0);
    return res;
#else
    return __float2bfloat16(f);
#endif
  } else if constexpr (std::is_same_v<T, __half>) {
    return __float2half(f);
  } else {
    static_assert(std::is_same_v<T, __half> ||
                      std::is_same_v<T, __hip_bfloat16>,
                  "b32_to_b16 only supports __half and __hip_bfloat16");
  }
}
