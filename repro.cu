// Minimal reproducer: cuTensorMapEncodeTiled with a CUtensorMap that the
// compiler has aligned to 64 bytes (alignas(64) in cuda.h), obtained through
// the driver entry point as torch/CUTLASS do.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <new>

typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                             const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                             const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeFn g_encode;
static CUtensorMap g_static_map;  // CUtensorMap is alignas(64) in cuda.h

// Same arguments as the failing HALOps call.
static CUresult halops_call(CUtensorMap* m, void* ptr) {
  cuuint64_t size[2] = {56, 650};
  cuuint64_t strides[1] = {576};
  cuuint32_t box[2] = {56, 1};
  cuuint32_t es[2] = {1, 1};
  return g_encode(m, (CUtensorMapDataType)5, 2, ptr, size, strides, box, es,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                  (CUtensorMapL2promotion)2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

int main() {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q;
  cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                                   cudaEnableDefault, &q);
  printf("get entry point: cuda error %d, query %d, fn %p\n", (int)e, (int)q, fn);
  if (e != cudaSuccess || !fn) return 2;
  g_encode = (EncodeFn)fn;

  void* buf = nullptr;
  printf("cudaMalloc: %d\n", (int)cudaMalloc(&buf, 650 * 576));
  printf("data ptr %p\n", buf);

  CUtensorMap local;  // as in csrc/kernel/mla/mla_host_utils.h
  printf("A local alignas(64) CUtensorMap at %p (mod 64 = %zu): result %d\n", (void*)&local,
         (size_t)((uintptr_t)&local % 64), (int)halops_call(&local, buf));

  printf("B static alignas(64) CUtensorMap at %p (mod 64 = %zu): result %d\n",
         (void*)&g_static_map, (size_t)((uintptr_t)&g_static_map % 64),
         (int)halops_call(&g_static_map, buf));

  CUtensorMap* heap = new CUtensorMap;  // C++17 aligned new
  printf("C heap CUtensorMap at %p (mod 64 = %zu): result %d\n", (void*)heap,
         (size_t)((uintptr_t)heap % 64), (int)halops_call(heap, buf));

  // Control: an ordinary uint8 2-D map.
  CUtensorMap ctl;
  cuuint64_t size[2] = {128, 64};
  cuuint64_t strides[1] = {128};
  cuuint32_t box[2] = {64, 8};
  cuuint32_t es[2] = {1, 1};
  void* buf2 = nullptr;
  cudaMalloc(&buf2, 128 * 64);
  CUresult rc = g_encode(&ctl, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, buf2, size, strides, box, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  printf("D control uint8 128x64 at %p (mod 64 = %zu): result %d\n", (void*)&ctl,
         (size_t)((uintptr_t)&ctl % 64), (int)rc);
  return 0;
}
