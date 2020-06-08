#ifndef NVSHMEMI_BARRIER_ON_STREAM_CPU_H
#define NVSHMEMI_BARRIER_ON_STREAM_CPU_H 1
#include "barrier_common.h"

int nvshmemxi_barrier_cpu_base_on_stream(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                         cudaStream_t stream);
void nvshmemxi_sync_p2p_on_stream(int PE_start, int logPE_stride, int PE_size, cudaStream_t stream);
int nvshmemxi_barrier_all_cpu_base_on_stream(cudaStream_t stream);

#endif
