#ifndef NVSHMEMI_ALLTOALL_COMMON_CPU_H
#define NVSHMEMI_ALLTOALL_COMMON_CPU_H 1
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif

#define CALL_ALLTOALL_ON_STREAM_KERN(BITS)                                                   \
    void call_alltoall##BITS##_on_stream_kern(void *dest, const void *source, size_t nelems, \
                                              int PE_start, int logPE_stride, int PE_size,   \
                                              long *pSync, cudaStream_t stream);

CALL_ALLTOALL_ON_STREAM_KERN(32);
CALL_ALLTOALL_ON_STREAM_KERN(64);

#if __cplusplus
}
#endif

#endif
