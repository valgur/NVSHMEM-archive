/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_ib_common.h"

#define IBV_REG_DMABUF_MR 4294967296ULL

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                      struct ibv_pd *pd, nvshmem_mem_handle_t *mem_handle,
                                      void *buf, size_t length,  bool local_only, bool dmabuf_support) {
    struct nvshmemt_ib_common_mem_handle *handle = (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    struct ibv_mr *mr = NULL;
    int status = 0;

    assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    handle->fd = 0;
    mr = ftable->reg_mr(pd, buf, length,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                            IBV_ACCESS_REMOTE_ATOMIC);
    if (mr == NULL) {
        ERROR_PRINT("ibv_reg_mr failed.\n");

#if CUDA_VERSION >= 11070
        bool host_memory = false;
        if (length >= IBV_REG_DMABUF_MR) {
            ERROR_PRINT("Only buffers up to 4GiB are supported with ibv_reg_dmabuf_mr.\n");
            ERROR_PRINT("In order to use larger memory allocations, please load nv_peer_mem or nvidia_peermem.\n");
            goto reg_dmabuf_failure;
        }

        if (local_only) {
            cudaPointerAttributes attr;
            status = cudaPointerGetAttributes(&attr, buf);
            if (status != cudaSuccess) {
                host_memory = true;
                status = 0;
                cudaGetLastError();
            } else if(attr.type != cudaMemoryTypeDevice) {
                host_memory = true;
            }
        }

        if (ftable->reg_dmabuf_mr != NULL && !host_memory && dmabuf_support && CUPFN(cuMemGetHandleForAddressRange)) {
            ERROR_PRINT("Falling back to ibv_reg_dmabuf_mr.\n");
            size_t page_size = sysconf(_SC_PAGESIZE);
            CUdeviceptr p;
            size_t size_aligned = ROUNDUP(length, page_size);
            p = (CUdeviceptr)((uintptr_t)buf & ~(page_size - 1));

            CUCHECKGOTO(cuMemGetHandleForAddressRange(&handle->fd, (CUdeviceptr)p, size_aligned,
                                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0), status, out);

            mr = ftable->reg_dmabuf_mr(pd, 0, size_aligned, (uint64_t)p, handle->fd,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
            if (mr == NULL) {
                ERROR_PRINT("ibv_reg_dmabuf_mr failed.\n");
                goto reg_dmabuf_failure;
            }

            INFO(NVSHMEM_TRANSPORT, "ibv_reg_dmabuf_mr handle %p handle->mr %p", handle, handle->mr);
        } else {
            ERROR_PRINT("Unable to fall back to ibv_reg_dmabuf_mr. Not supported by the current configuration.\n");
            ERROR_PRINT("In order to register memory, please load nv_peer_mem or nvidia_peermem.\n");
        }
reg_dmabuf_failure:
        ERROR_PRINT("Unable to fall back to ibv_reg_dmabuf_mr.\n");
#else
        ERROR_PRINT("Unable to fall back to ibv_reg_dmabuf_mr. Not supported with the currently compiled CUDA version.\n");
#endif
    }

    NULL_ERROR_JMP(mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "mem registration failed \n");
    INFO(NVSHMEM_TRANSPORT, "ibv_reg_mr handle %p handle->mr %p", handle, handle->mr);

    handle->lkey = mr->lkey;
    handle->rkey = mr->rkey;
    handle->mr = mr;

out:
    return status;
}

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle) {
    int status = 0;
    struct nvshmemt_ib_common_mem_handle *handle = (struct nvshmemt_ib_common_mem_handle *)mem_handle;

    INFO(NVSHMEM_TRANSPORT, "ibv_dereg_mr handle %p handle->mr %p", handle, handle->mr);
    if (handle->mr) {
        status = ftable->dereg_mr((struct ibv_mr *)handle->mr);
    }
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");

out:
    return status;
}

#ifdef NVSHMEM_MLX5_CODE
bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context) {
    int status;
    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {0,};
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {0,};

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE |
        (MLX5_CAP_GENERAL << 1) |
        HCA_CAP_OPMOD_GET_CUR
    );

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out, sizeof(cmd_cap_out));

    if (status == 0) {
        return true;
    }
    return false;
}
#endif
