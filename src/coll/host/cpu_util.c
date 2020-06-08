int check_ipc_mem() {
    int status = NVSHMEMI_COLL_CPU_STATUS_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;
    int *tmp_int_ptr;
    int dev_int_ptr;

    if (0 != nvshmem_state->mype) {
        tmp_int_ptr = nvshm_cpu_coll_info.ipc_shm_addr;
        cuda_status =
            cudaMemcpy((void *)&(tmp_int_ptr[nvshmem_state->mype]), (void *)&(nvshmem_state->mype),
                       sizeof(int), cudaMemcpyHostToDevice);
        if (cudaSuccess != cuda_status) NVSHMEMI_COLL_CPU_CUDA_ERR_POP(cuda_status);
        cudaDeviceSynchronize();
    }

    status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);

    if (0 != nvshmem_state->mype) {
        tmp_int_ptr = nvshm_cpu_coll_info.ipc_shm_addr;

        for (int ii = 1; ii < nvshmem_state->npes; ii++) {
            cuda_status = cudaMemcpy((void *)&dev_int_ptr, (void *)&(tmp_int_ptr[ii]), sizeof(int),
                                     cudaMemcpyDeviceToHost);
            if (cudaSuccess != cuda_status) NVSHMEMI_COLL_CPU_CUDA_ERR_POP(cuda_status);
            cudaDeviceSynchronize();

            fprintf(stderr, "[%d] inval = %d\n", nvshmem_state->mype, dev_int_ptr);
        }
    }

    goto fn_out;
fn_out:
    return status;
fn_fail:
    return status;
}

nvshmemi_coll_support_cpu_modes_t cpu_supported_modes;

/*
 * should return a way to indicate all the modes supported
 *
 * should be a combination of:
 * - none
 * - intra-node intra-socket
 * - intra-node inter-socket
 * - inter-node inter-socket
 */
int nvshmemi_coll_common_cpu_return_modes(nvshmemi_coll_support_cpu_modes_t *cpu_supported_modes) {
    int status = NVSHMEMI_COLL_CPU_STATUS_SUCCESS;

    // logic to check supported modes goes here

    // return intra-node intra-socket by default for now
    cpu_supported_modes->none = 0;
    cpu_supported_modes->intra_node_intra_sock = 1;
    cpu_supported_modes->intra_node_inter_sock = 0;
    cpu_supported_modes->inter_node_inter_sock = 0;

    goto fn_out;
fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_event_xchange() {
    cuda_status = cudaEventCreate(&(nvshm_cpu_coll_info.peer_cuda_events[nvshmem_state->mype]),
                                  cudaEventDisableTiming | cudaEventInterprocess);
    if (cudaSuccess != cuda_status) NVSHMEMI_COLL_CPU_CUDA_ERR_POP(cuda_status);

    cuda_status = cudaIpcGetEventHandle(
        (cudaIpcEventHandle_t *)&(
            nvshm_cpu_coll_info.peer_handles[nvshmem_state->mype].evnt_handle),
        nvshm_cpu_coll_info.peer_cuda_events[nvshmem_state->mype]);
    if (cudaSuccess != cuda_status) NVSHMEMI_COLL_CPU_CUDA_ERR_POP(cuda_status);

    my_ipc_handle = nvshm_cpu_coll_info.peer_handles[nvshmem_state->mype];
    status = nvshmem_state->boot_handle.allgather(
        (void *)&my_ipc_handle, (void *)(nvshm_cpu_coll_info.peer_handles),
        sizeof(cpu_coll_cuda_ipc_exch_t), &nvshmem_state->boot_handle);

    for (int ii = 0; ii < nvshmem_state->npes; ii++) {
        if (ii == nvshmem_state->mype) continue;

        cuda_status = cudaIpcOpenEventHandle(&(nvshm_cpu_coll_info.peer_cuda_events[ii]),
                                             nvshm_cpu_coll_info.peer_handles[ii].evnt_handle);
        if (cudaSuccess != cuda_status) NVSHMEMI_COLL_CPU_CUDA_ERR_POP(cuda_status);
    }
}
