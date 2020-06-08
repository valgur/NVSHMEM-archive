#ifndef NVSHMEMI_BARRIER_CPU_H
#define NVSHMEMI_BARRIER_CPU_H 1
#include "barrier_common.h"
void nvshmemi_sync(int PE_start, int logPE_stride, int PE_size, long *pSync);
void nvshmemi_sync_p2p(int PE_start, int logPE_stride, int PE_size, long *pSync);
void nvshmemi_barrier_shm(int PE_start, int logPE_stride, int PE_size, long *pSync);
void nvshmemi_barrier_all_shm();
#endif
