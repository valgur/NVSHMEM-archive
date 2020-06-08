#ifndef NVSHMEMI_COLLECT_CPU_H
#define NVSHMEMI_COLLECT_CPU_H 1
#include "collect_common.h"

int nvshmemi_collect_cpu_all_bcast(void *dest, const void *source, int type_size, size_t nelems,
                                   int PE_start, int logPE_stride, int PE_size, long *pSync);
int nvshmemi_collect_cpu_p2p_all_pull(void *dest, const void *source, int type_size, size_t nelems,
                                      int PE_start, int logPE_stride, int PE_size, long *pSync);
int nvshmemi_collect_cpu_p2p_all_push(void *dest, const void *source, int type_size, size_t nelems,
                                      int PE_start, int logPE_stride, int PE_size, long *pSync);
void nvshmemi_collect(void *dest, const void *source, int type_size, size_t nelems, int PE_start,
                      int logPE_stride, int PE_size, long *pSync);
#endif
