#ifndef NVSHMEMI_COLLECT_COMMON_CPU_H
#define NVSHMEMI_COLLECT_COMMON_CPU_H 1
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif

#define CALL_COLLECT_ON_STREAM_KERN(BITS)                                                   \
    void call_collect##BITS##_on_stream_kern(void *dest, const void *source, size_t nelems, \
                                             int PE_start, int logPE_stride, int PE_size,   \
                                             long *pSync, cudaStream_t stream);

CALL_COLLECT_ON_STREAM_KERN(32);
CALL_COLLECT_ON_STREAM_KERN(64);

#if __cplusplus
}
#endif

#endif
