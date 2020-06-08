#ifndef NVSHMEMI_BARRIER_COMMON_CPU_H
#define NVSHMEMI_BARRIER_COMMON_CPU_H 1
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif
int call_barrier_on_stream_kern(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                cudaStream_t stream);
int call_barrier_all_on_stream_kern(cudaStream_t stream);

int call_sync_on_stream_kern(int PE_start, int logPE_stride, int PE_size, long *pSync,
                             cudaStream_t stream);
int call_sync_all_on_stream_kern(cudaStream_t stream);
#if __cplusplus
}
#endif

#endif
