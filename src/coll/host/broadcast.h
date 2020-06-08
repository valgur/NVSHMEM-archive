#ifndef NVSHMEMI_BROADCAST_CPU_H
#define NVSHMEMI_BROADCAST_CPU_H 1
#include "broadcast_common.h"

int nvshmemi_broadcast_cpu_ipc_all_pull(void *dest, const void *source, int type_size,
                                        size_t nelems, int PE_root, int PE_start, int logPE_stride,
                                        int PE_size, long *pSync);
int nvshmemi_broadcast_cpu_p2p_all_pull(void *dest, const void *source, int type_size,
                                        size_t nelems, int PE_root, int PE_start, int logPE_stride,
                                        int PE_size, long *pSync);
int nvshmemi_broadcast_cpu_p2p_all_push(void *dest, const void *source, int type_size,
                                        size_t nelems, int PE_root, int PE_start, int logPE_stride,
                                        int PE_size, long *pSync);
void nvshmemi_broadcast(void *dest, const void *source, int type_size, size_t nelems, int PE_root,
                        int PE_start, int logPE_stride, int PE_size, long *pSync);
int bcast_sync(int root, int val);
#endif
