#ifndef NVSHMEMI_BROADCAST_ON_STREAM_CPU_H
#define NVSHMEMI_BROADCAST_ON_STREAM_CPU_H 1
#include "broadcast_common.h"
int nvshmemxi_broadcast_cpu_ipc_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                   size_t nelems, int PE_root, int PE_start,
                                                   int logPE_stride, int PE_size, long *pSync,
                                                   cudaStream_t stream);
int nvshmemxi_broadcast_cpu_p2p_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                   size_t nelems, int PE_root, int PE_start,
                                                   int logPE_stride, int PE_size, long *pSync,
                                                   cudaStream_t stream);
int nvshmemxi_broadcast_cpu_p2p_all_push_on_stream(void *dest, const void *source, int type_size,
                                                   size_t nelems, int PE_root, int PE_start,
                                                   int logPE_stride, int PE_size, long *pSync,
                                                   cudaStream_t stream);
void nvshmemxi_broadcast_on_stream(void *dest, const void *source, int type_size, size_t nelems,
                                   int PE_root, int PE_start, int logPE_stride, int PE_size,
                                   long *pSync, cudaStream_t stream);
#endif
