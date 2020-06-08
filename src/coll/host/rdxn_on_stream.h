#ifndef NVSHMEMI_RDXN_ON_STREAM_CPU_H
#define NVSHMEMI_RDXN_ON_STREAM_CPU_H 1

#include "rdxn_common.h"

int nvshmemxi_reduction_op_cpu_in_mem_ring_on_stream(rdxn_opr_t *rdx_op);
int nvshmemxi_rdxn_op_cpu_slxn_on_stream(rdxn_opr_t *rdx_op);

#endif
