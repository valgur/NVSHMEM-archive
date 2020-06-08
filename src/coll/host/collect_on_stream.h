#ifndef NVSHMEMI_COLLECT_ON_STREAM_CPU_H
#define NVSHMEMI_COLLECT_ON_STREAM_CPU_H 1
#include "collect_common.h"

int nvshmemxi_collect_cpu_all_bcast_on_stream(void *dest, const void *source, int type_size,
                                              size_t nelems, int PE_start, int logPE_stride,
                                              int PE_size, long *pSync, cudaStream_t stream);
int nvshmemxi_collect_cpu_p2p_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                 size_t nelems, int PE_start, int logPE_stride,
                                                 int PE_size, long *pSync, cudaStream_t stream);
int nvshmemxi_collect_cpu_p2p_all_push_on_stream(void *dest, const void *source, int type_size,
                                                 size_t nelems, int PE_start, int logPE_stride,
                                                 int PE_size, long *pSync, cudaStream_t stream);
void nvshmemxi_collect_on_stream(void *dest, const void *source, int type_size, size_t nelems,
                                 int PE_start, int logPE_stride, int PE_size, long *pSync,
                                 cudaStream_t stream);
#endif
