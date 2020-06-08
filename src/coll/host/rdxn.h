#ifndef NVSHMEMI_RDXN_CPU_H
#define NVSHMEMI_RDXN_CPU_H 1

#include "rdxn_common.h"

int nvshmemi_reduction_op_cpu_in_kern_ring(rdxn_opr_t *rdx_op);
int nvshmemi_reduction_op_cpu_in_mem_ring(rdxn_opr_t *rdx_op);
int nvshmemi_reduction_op_cpu_p2p_allgather(rdxn_opr_t *rdx_op);
int nvshmemi_reduction_op_cpu_p2p_on_demand_gather(rdxn_opr_t *rdx_op);
int nvshmemi_reduction_op_cpu_p2p_segmented_gather(rdxn_opr_t *rdx_op);
int nvshmemi_rdxn_op_cpu_slxn(rdxn_opr_t *rdx_op);
int nvshmemi_rdxn_r_op_cpu_slxn(rdxn_opr_t *rdx_op);
int nvshmemi_rdxn_c_op_cpu_slxn(rdxn_opr_t *rdx_op);

#endif
