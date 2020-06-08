/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "cpu_coll.h"

cpu_coll_info_t nvshm_cpu_coll_info;
int nvshm_enable_cpu_coll = 1;
int nvshm_enable_p2p_cpu_coll = 0;
int nvshm_use_p2p_cpu_push = 1;
int nvshm_use_tg_for_stream_coll = 1;
int nvshm_use_tg_for_cpu_coll = 1;
int nvshm_cpu_coll_initialized = 0;
int nvshm_cpu_coll_offset_reqd = 0;
int nvshm_cpu_coll_sync_reqd = 1;
int nvshm_cpu_rdxn_seg_size = 16384;
int nvshm_cpu_coll_gpu_ipc_sync_size;
int nvshm_cpu_ipc_size;
int nvshm_rdx_num_tpb = 32;
int nvshm_use_p2p_cpu_rdxn_allgather = 0;
int nvshm_use_p2p_cpu_rdxn_od_gather = 0;
char *cu_err_string;

int bcast_sync(int root, int val) {
    int status = 0;
    volatile int *bcast_sync_arr = nvshm_cpu_coll_info.cpu_bcast_int_sync_arr;

    if (root == nvshmem_state->mype) {
        // wait for all sync arr vals to be 0
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
                // do nothing
            }
        }

        // insert val
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            bcast_sync_arr[ii] = val;
        }

        // wait for all sync arr vals to be 0
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
                // do nothing
            }
        }

    } else {
        while (val != bcast_sync_arr[nvshmem_state->mype]) {
        }

        bcast_sync_arr[nvshmem_state->mype] = 0;
    }

    return status;
}

/*
  General collective rules:
  1. PE_start, logPE_stride, and PE_size is the same at all active PEs
  2. Active PEs pass same psync array (psync contents restored after call)
  3. All collectives are blocking and return after completion

  Barrier_all / barrier rules:
  1. Blocks until all other PEs have arrived + all local/remote updates finish

  Broadcast (replicate) rules:
  1. suffix 32/64 => argument is in multiples of 32/64 bits.
  2. This is integer by default in fortran
  3. PE_root shouldn't copy data from source to dest
  4. Synchronization is required prior broadcast (via barrier)

  Collect/Fcollect (concatenate) rules:
  1. dest >= concatenated size of all PEs
  2. 32/64 => element size = 32/64
  3. nelems have to be the same for fcollect but not collect across PEs

  Reduction rules:
  1. nreduce has to be the same across PEs
  2. source and dest can be the same but not overlapping

  Alltoalls (strided alltoall) rules:
  1. Same as alltoall but the data in source and dest needn't be contiguous
  2. stride = 1 => contiguous. stride = 2 => 2 = distance b/w contig elements
  3. stride >= 1
  4. strides and nelems are the same across PEs
 */

int nvshmemi_coll_common_cpu_read_env() {
    int status = 0;
    char *value = NULL;

    nvshm_enable_cpu_coll        = nvshmemi_options.ENABLE_CPU_COLL;
    nvshm_enable_p2p_cpu_coll    = nvshmemi_options.ENABLE_P2P_CPU_COLL;
    nvshm_use_p2p_cpu_push       = nvshmemi_options.USE_P2P_CPU_PUSH;
    nvshm_use_tg_for_stream_coll = nvshmemi_options.USE_TG_FOR_STREAM_COLL;
    nvshm_use_tg_for_cpu_coll    = nvshmemi_options.USE_TG_FOR_CPU_COLL;
    nvshm_cpu_coll_offset_reqd   = nvshmemi_options.CPU_COLL_OFFSET_REQD;
    nvshm_cpu_coll_sync_reqd     = nvshmemi_options.CPU_COLL_SYNC_REQD;
    nvshm_cpu_rdxn_seg_size      = nvshmemi_options.CPU_RDXN_SEG_SIZE;
    nvshm_use_p2p_cpu_rdxn_allgather = nvshmemi_options.USE_P2P_CPU_RDXN_ALLGATHER;
    nvshm_use_p2p_cpu_rdxn_od_gather = nvshmemi_options.USE_P2P_CPU_RDXN_OD_GATHER;
    nvshm_rdx_num_tpb            = nvshmemi_options.RDX_NUM_TPB;

    if (0 == nvshm_enable_cpu_coll)
        fprintf(stderr, "Warning: nvshm collectives disabled\n");

    if (32 > nvshm_rdx_num_tpb) {
        nvshm_rdx_num_tpb = 32;
        fprintf(stderr, "WARN: #threads/block < 32; Using 32 instead\n");
    }

fn_out:
    return status;
}

int nvshmemi_coll_common_cpu_init_memory() {
    int status = 0;
    char *tmp;
    char *base;
    int offset;
    struct stat sb;
    void *pid;
    void *peer_pids;

    nvshm_cpu_ipc_size = (CPU_SYNC_SIZE + CPU_DATA_SIZE) * nvshmem_state->npes;
    nvshm_cpu_coll_gpu_ipc_sync_size = (CPU_GPU_SYNC_SIZE * nvshmem_state->npes);

    nvshm_cpu_coll_info.my_pid = getpid();
    nvshm_cpu_coll_info.peer_pids = NULL;

    nvshm_cpu_coll_info.peer_pids = (int *)malloc(sizeof(int) * nvshmem_state->npes);
    nvshm_cpu_coll_info.peer_handles =
        (cpu_coll_cuda_ipc_exch_t *)malloc(sizeof(cpu_coll_cuda_ipc_exch_t) * nvshmem_state->npes);
    nvshm_cpu_coll_info.peer_cuda_events =
        (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * nvshmem_state->npes);

    if (0 == nvshmem_state->mype) {
        sprintf(nvshm_cpu_coll_info.fname, "/nvshm-shm-file-%d", nvshm_cpu_coll_info.my_pid);

        nvshm_cpu_coll_info.shm_fd =
            shm_open(nvshm_cpu_coll_info.fname, (O_RDWR | O_CREAT | O_EXCL), S_IRWXU);
        if (-1 == nvshm_cpu_coll_info.shm_fd) {
            fprintf(stderr, "[%d] shm_open() error %s\n", nvshmem_state->mype, strerror(errno));
            NVSHMEMI_COLL_CPU_ERR_POP();
        }

        tmp = (char *)malloc(sizeof(char) * nvshm_cpu_ipc_size);
        memset(tmp, 0, sizeof(char) * nvshm_cpu_ipc_size);

        status = (int)write(nvshm_cpu_coll_info.shm_fd, tmp, sizeof(char) * nvshm_cpu_ipc_size);
        if (-1 == status) {
            fprintf(stderr, "[%d] write() error %s\n", nvshmem_state->mype, strerror(errno));
            NVSHMEMI_COLL_CPU_ERR_POP();
        }
        /*XXX:is a cache flush needed here?*/
        free(tmp);
    }

    pid = (void *)&(nvshm_cpu_coll_info.my_pid);
    peer_pids = (void *)(nvshm_cpu_coll_info.peer_pids);
    status = nvshmem_state->boot_handle.allgather(pid, peer_pids, sizeof(int),
                                                  &nvshmem_state->boot_handle);
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();
    sprintf(nvshm_cpu_coll_info.fname, "nvshm-shm-file-%d", nvshm_cpu_coll_info.peer_pids[0]);

    // setup cpu shared memory
    //=======================

    nvshm_cpu_coll_info.shm_fd = shm_open(nvshm_cpu_coll_info.fname, O_RDWR, S_IRWXU);
    if (-1 == nvshm_cpu_coll_info.shm_fd) {
        fprintf(stderr, "[%d] shm_open() error %s\n", nvshmem_state->mype, strerror(errno));
        NVSHMEMI_COLL_CPU_ERR_POP();
    }

    status = (int)ftruncate(nvshm_cpu_coll_info.shm_fd, sizeof(char) * nvshm_cpu_ipc_size);
    if (-1 == status) {
        fprintf(stderr, "[%d] ftruncate() error %s\n", nvshmem_state->mype, strerror(errno));
        NVSHMEMI_COLL_CPU_ERR_POP();
    }

    do {
        if (-1 == fstat(nvshm_cpu_coll_info.shm_fd, &sb)) {
            fprintf(stderr, "[%d] fstat() error %s\n", nvshmem_state->mype, strerror(errno));
            NVSHMEMI_COLL_CPU_ERR_POP();
        }
    } while (sb.st_size != sizeof(char) * nvshm_cpu_ipc_size);

    nvshm_cpu_coll_info.shm_addr =
        mmap(NULL, sizeof(char) * nvshm_cpu_ipc_size, (PROT_READ | PROT_WRITE), MAP_SHARED,
             nvshm_cpu_coll_info.shm_fd, 0);
    if (nvshm_cpu_coll_info.shm_addr == (void *)-1) {
        fprintf(stderr, "[%d] mmap() error %s\n", nvshmem_state->mype, strerror(errno));
        NVSHMEMI_COLL_CPU_ERR_POP();
    }

    status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

    nvshm_cpu_coll_info.cpu_bcast_int_sync_arr = (volatile int *)nvshm_cpu_coll_info.shm_addr;
    nvshm_cpu_coll_info.cpu_bcast_int_data_arr =
        (volatile int *)((char *)nvshm_cpu_coll_info.shm_addr +
                         (sizeof(int) * nvshmem_state->npes));

    bcast_sync(0, 1);

    // setup gpu shared memory
    //=======================

    nvshm_cpu_coll_info.own_shm_addr = NULL;
    nvshm_cpu_coll_info.ipc_shm_addr = NULL;

    nvshm_cpu_coll_info.own_shm_addr = nvshmemi_malloc(nvshm_cpu_coll_gpu_ipc_sync_size);
    if (!nvshm_cpu_coll_info.own_shm_addr) {
        fprintf(stderr, "nvshmemi_malloc failed \n");
        goto fn_out;
    }

    base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
    offset = (char *)nvshm_cpu_coll_info.own_shm_addr - base;
    tmp = (char *)malloc(nvshm_cpu_coll_gpu_ipc_sync_size * sizeof(int));
    memset(tmp, 0, nvshm_cpu_coll_gpu_ipc_sync_size * sizeof(int));
    CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)nvshm_cpu_coll_info.own_shm_addr, (const void *)tmp,
                            nvshm_cpu_coll_gpu_ipc_sync_size * sizeof(int)));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, fn_fail, "error in cuMemcpyHtoD\n");
    cuStreamSynchronize(0);
    free(tmp);

    nvshm_cpu_coll_info.ipc_shm_addr = (void *)((char *)nvshmem_state->peer_heap_base[0] + offset);

    /* all algorithms that use this are bounded by
     * size = (sizeof(int) * nvshmem_state->npes)
     */
    nvshm_cpu_coll_info.gpu_bcast_int_sync_arr = (volatile int *)nvshm_cpu_coll_info.ipc_shm_addr;

    // adding this to avoid use of newly allocated memory for first barrier
    nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);

fn_out:

    if (0 == nvshmem_state->mype) {
        shm_unlink(nvshm_cpu_coll_info.fname);
    }

    return status;
fn_fail:

    if (0 == nvshmem_state->mype) {
        shm_unlink(nvshm_cpu_coll_info.fname);
    }

    return status;
}

int nvshmemi_coll_common_cpu_init() {
    int status = 0;

    nvshmemi_rdxn_fxn_ptrs_init();

    status = nvshmemi_coll_common_cpu_read_env();
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

    if (0 == nvshm_enable_cpu_coll) {
        return status;
    }

    // status = nvshmemi_coll_common_cpu_init_memory();
    // if (status) NVSHMEMI_COLL_CPU_ERR_POP();

    nvshm_cpu_coll_initialized = 1;

fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_coll_common_cpu_finalize() {
    int status = 0;

    if ((0 == nvshm_enable_cpu_coll) || (0 == nvshm_cpu_coll_initialized)) return status;

    free(nvshm_cpu_coll_info.peer_pids);
    free(nvshm_cpu_coll_info.peer_handles);
    free(nvshm_cpu_coll_info.peer_cuda_events);

#if 0
    if (0 == nvshmem_state->mype) {
        if (0 != munmap((void *)nvshm_cpu_coll_info.shm_addr,
                        sizeof(char) * nvshm_cpu_ipc_size))
            NVSHMEMI_COLL_CPU_ERR_POP();
    }

    if (-1 == close(nvshm_cpu_coll_info.shm_fd)) NVSHMEMI_COLL_CPU_ERR_POP();

    if (NULL != nvshm_cpu_coll_info.own_shm_addr)
        nvshmemi_free((void *)nvshm_cpu_coll_info.own_shm_addr);
#endif

fn_out:
    return status;
fn_fail:
    return status;
}
