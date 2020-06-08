#ifndef NVSHMEMI_ALLTOALL_CPU_H
#define NVSHMEMI_ALLTOALL_CPU_H 1
#include "alltoall_common.h"

int nvshmemi_scatter_cpu_ipc_all_pull(void *dest, const void *source, int type_size, size_t nelems,
                                      int PE_root, int PE_start, int logPE_stride, int PE_size,
                                      long *pSync);
int nvshmemi_alltoall_cpu_ipc_all_scatter(void *dest, const void *source, int type_size,
                                          size_t nelems, int PE_start, int logPE_stride,
                                          int PE_size, long *pSync);
int nvshmemi_alltoall_cpu_p2p_all_push(void *dest, const void *source, int type_size, size_t nelems,
                                       int PE_start, int logPE_stride, int PE_size, long *pSync);
int nvshmemi_alltoall_cpu_p2p_all_pull(void *dest, const void *source, int type_size, size_t nelems,
                                       int PE_start, int logPE_stride, int PE_size, long *pSync);
void nvshmemi_alltoall(void *dest, const void *source, int type_size, size_t nelems, int PE_start,
                       int logPE_stride, int PE_size, long *pSync);

#endif
