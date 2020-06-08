#ifndef NVSHMEMI_ALLTOALL_ON_STREAM_CPU_H
#define NVSHMEMI_ALLTOALL_ON_STREAM_CPU_H 1
#include "alltoall_common.h"
int nvshmemxi_scatter_cpu_ipc_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                 size_t nelems, int PE_root, int PE_start,
                                                 int logPE_stride, int PE_size, long *pSync,
                                                 cudaStream_t stream);
int nvshmemxi_alltoall_cpu_ipc_all_scatter_on_stream(void *dest, const void *source, int type_size,
                                                     size_t nelems, int PE_start, int logPE_stride,
                                                     int PE_size, long *pSync, cudaStream_t stream);
int nvshmemxi_alltoall_cpu_p2p_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                  size_t nelems, int PE_start, int logPE_stride,
                                                  int PE_size, long *pSync, cudaStream_t stream);
int nvshmemxi_alltoall_cpu_p2p_all_push_on_stream(void *dest, const void *source, int type_size,
                                                  size_t nelems, int PE_start, int logPE_stride,
                                                  int PE_size, long *pSync, cudaStream_t stream);
void nvshmemxi_alltoall_on_stream(void *dest, const void *source, int type_size, size_t nelems,
                                  int PE_start, int logPE_stride, int PE_size, long *pSync,
                                  cudaStream_t stream);
#endif
