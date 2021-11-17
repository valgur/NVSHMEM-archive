/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>      

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"

#include <stdlib.h>
#include <string.h>
#include "topo.h"
#include "util.h"
#include "cpu_coll.h"
#include "gpu_coll.h"
#include "unistd.h"
#include "debug.h"

static void nvshmemi_init_debug(void);
static void nvshmemi_init_msg(void);

nvshmemi_state_t *nvshmemi_state;
nvshmem_options_t nvshmem_options;
int nvshmemi_cuda_driver_version;
bool nvshmemi_use_cuda_vmm = 0;
const char *p_err_str;
int nvshmem_debug_level;
uint64_t nvshmem_debug_mask = NVSHMEM_INIT;  // Default debug sub-system mask is INIT
pthread_mutex_t nvshmem_debug_output_lock;
bool nvshmemi_is_limited_mpg_run = 0;
FILE *nvshmem_debug_file = stdout;
static char shm_name[100];

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point nvshmem_epoch;
#endif

static int nvshmemi_transport_cap_support_rma(int cap) {
    if (cap & (NVSHMEM_TRANSPORT_CAP_CPU_READ   |
               NVSHMEM_TRANSPORT_CAP_CPU_WRITE  |
               NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD |
               NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST)) {
                   return 1;
    }
    return 0;
}

static int nvshmemi_transport_cap_support_amo(int cap) {
    if (cap & (NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS |
               NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS)) {
                   return 1;
    }
    return 0;
}

int nvshmemi_bootstrap(int flags, nvshmemx_init_attr_t *nvshmem_attr, nvshmemi_state_t *state) {
    int status = 0;
    uint64_t myHostHash = 0;
    uint64_t *hostHash = 0;
    int mype_node = 0, npes_node = 0;

    if (flags & NVSHMEMX_INIT_WITH_MPI_COMM) {
        bootstrap_attr_t boot_attr;
        boot_attr.mpi_comm = nvshmem_attr->mpi_comm;
        status = bootstrap_init(BOOTSTRAP_MPI, &boot_attr, &state->boot_handle);
    } else if (flags & NVSHMEMX_INIT_WITH_SHMEM) {
        bootstrap_attr_t boot_attr;
        boot_attr.initialize_shmem = 0;
        status = bootstrap_init(BOOTSTRAP_SHMEM, &boot_attr, &state->boot_handle);
    } else {
        /* User called nvshmem_init or supplied no flags to nvshmemx_init_attr */
        if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "PMI") == 0) {
            status = bootstrap_init(BOOTSTRAP_PMI, NULL, &state->boot_handle);
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "MPI") == 0) {
            status = bootstrap_init(BOOTSTRAP_MPI, NULL, &state->boot_handle);
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "SHMEM") == 0) {
            status = bootstrap_init(BOOTSTRAP_SHMEM, NULL, &state->boot_handle);
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "plugin") == 0) {
            status = bootstrap_init(BOOTSTRAP_PLUGIN, NULL, &state->boot_handle);
        } else {
            ERROR_PRINT("Invalid bootstrap '%s'\n", nvshmemi_options.BOOTSTRAP);
            status = 1;
        }
    }
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_init failed \n");

    state->mype = state->boot_handle.pg_rank;
    state->npes = state->boot_handle.pg_size;

    nvshmem_nvtx_set_thread_name(state->mype);

    myHostHash = getHostHash();
    hostHash = (uint64_t *)malloc(sizeof(uint64_t) * state->npes);
    status = state->boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                          &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of host hashes failed \n");
    for (int i = 0; i < state->npes; i++) {
        if (hostHash[i] == myHostHash) {
            if (i == state->mype) mype_node = npes_node;
            npes_node++;
        }
    }
    state->mype_node = mype_node;
    state->npes_node = npes_node;

out:
    if (hostHash) {
        free(hostHash);
    }
    return status;
}

int nvshmemi_get_cucontext(nvshmemi_state_t *state) {
    int status = 0;
    int leastPriority, greatestPriority;

    status = cuInit(0);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "CUDA init failed \n");

    status = cuCtxGetDevice(&state->cudevice);
    if (status) {
        TRACE(NVSHMEM_INIT, "GPU not selected, cuCtxGetDevice failed, err: %d", status);
        status = NVSHMEMX_ERROR_GPU_NOT_SELECTED;
        goto out;
    } else {
        CUresult cres = cuCtxSynchronize();
        if (cres) {
            INFO(NVSHMEM_INIT,
                 "[%d] nvshmemi_get_cucontext->cuCtxSynchronize->%d(CUDA_ERROR_NOT_INITIALIZED %d) "
                 "my_stream %llu",
                 state->mype, cres, CUDA_ERROR_NOT_INITIALIZED, state->my_stream);
            status = cuDevicePrimaryCtxRetain(&state->cucontext, state->cudevice);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "get primary context failed \n");

            status = cuCtxSetCurrent(state->cucontext);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "set context failed \n");

            INFO(NVSHMEM_INIT, "retained primary context for device: %d", state->cudevice);

        } else {
            INFO(NVSHMEM_INIT,
                 "[%d] nvshmemi_get_cucontext->cuCtxSynchronize->CUDA_SUCCESS) my_stream %p",
                 state->mype, state->my_stream);
            status = cuCtxGetCurrent(&state->cucontext);

            INFO(NVSHMEM_INIT,
                 "int get_cucontext, queried and saved context for device: %d context: %p",
                 state->cudevice, state->cucontext);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "get context failed \n");
        }

        // identify device id
        int count;
        CUdevice curr_device;
        status = cuDeviceGetCount(&count);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cudaDeviceGetCount failed \n");

        for (int i = 0; i < count; i++) {
            status = cuDeviceGet(&curr_device, i);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "cudaDeviceGet failed \n");
            if (curr_device == state->cudevice) {
                state->device_id = i;
                break;
            }
        }

        status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cudaDeviceGetStreamPriorityRange failed \n");

        status =
            cuStreamCreateWithPriority(&state->my_stream, CU_STREAM_NON_BLOCKING, greatestPriority);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                     "cudaStreamCreateWithPriority failed \n");
        INFO(NVSHMEM_INIT,
             "[%d] nvshmemi_get_cucontext->cuCtxGetDevice->%d(CUDA_ERROR_INVALID_CONTEXT %d) "
             "cuStreamCreateWithPriority my_stream %p",
             state->mype, cres, CUDA_ERROR_INVALID_CONTEXT, state->my_stream);
    }
out:
    return status;
}

int nvshmemi_setup_stream_priorities(nvshmemi_state_t *state) {
    int status = 0;
    int leastPriority, greatestPriority;

    status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaDeviceGetStreamPriorityRange failed \n");

    status =
        cuStreamCreateWithPriority(&state->my_stream, CU_STREAM_NON_BLOCKING, greatestPriority);
    NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaStreamCreateWithPriority failed \n");
out:
    return status;
}

int nvshmemi_setup_memory_ordering(nvshmemi_state_t *state, int selected_transport) {
    int status = 0;

    state->fence[selected_transport] = state->transports[selected_transport]->host_ops.fence;
    state->quiet[selected_transport] = state->transports[selected_transport]->host_ops.quiet;

    return status;
}

int nvshmemi_teardown_handles(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_teardown_handles");
    int status = 0;
    free(state->rma);
    free(state->selected_transport_for_rma);
    free(state->amo);
    free(state->selected_transport_for_amo);
    free(state->fence);
    free(state->quiet);
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        status = cuStreamDestroy(state->custreams[i]);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuStreamDestroy failed \n");
        status = cuEventDestroy(state->cuevents[i]);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cuEventDestroy failed \n");
    }
    nvshmemx_buffer_unregister_all();
    free(state->registered_buffers);
    if (pthread_rwlock_destroy(&state->registered_buffer_lock)) {
        ERROR_PRINT("Unable to destroy registered buffer lock.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
    }
out:
    return status;
}

static int nvshmemi_setup_nvshmem_handles(nvshmemi_state_t *state) {
    int status = 0;
    int dev_attr = 0;
    /* TODO: We should really check all of these allocations. */
    state->rma = (rma_handle *)calloc(state->npes, sizeof(rma_handle));
    state->amo = (amo_handle *)calloc(state->npes, sizeof(amo_handle));
    state->fence = (fence_handle *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(fence_handle));
    state->quiet = (quiet_handle *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(quiet_handle));
    state->selected_transport_for_rma = (int *)calloc(state->npes, sizeof(int));
    state->selected_transport_for_amo = (int *)calloc(state->npes, sizeof(int));
    state->registered_buffers = (nvshmem_local_buf_handle_t **)calloc(64, sizeof(nvshmem_local_buf_handle_t *));
    state->registered_buffer_array_size = 64;
    state->registered_buffer_array_used = 0;
    CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&dev_attr,
                                              cudaDevAttrCanUseHostPointerForRegisteredMem,
                                              state->device_id));
    state->host_memory_registration_supported = dev_attr & cudaDevAttrCanUseHostPointerForRegisteredMem;

    status = pthread_rwlock_init(&state->registered_buffer_lock, NULL);
    if (status) {
        return status;
    }

    for (int pe = 0; pe < state->npes; pe++) {
        state->rma[pe] = 0;
        state->amo[pe] = 0;
        state->selected_transport_for_rma[pe] = -1;
        state->selected_transport_for_amo[pe] = -1;
    }
    for (int t = 0; t < NVSHMEM_TRANSPORT_COUNT; t++) {
        state->fence[t] = 0;
        state->quiet[t] = 0;
    }
    int memory_ordering_initialized = 0;
    int tbitmap;
    for (int i = 0; i < state->npes; i++) {
        bool amo_initialized = false, rma_initialized = false;
        tbitmap = state->transport_bitmap;
        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            if (!(state->transports[j])) {
                tbitmap >>= 1;
                continue;
            } 

            if ((tbitmap & 1)) {
                if (!rma_initialized &&
                    nvshmemi_transport_cap_support_rma(nvshmemi_state->transports[j]->cap[i])) {
                    state->rma[i] = state->transports[j]->host_ops.rma;
                    rma_initialized = true;
                    state->selected_transport_for_rma[i] = j;
                }

                if (!amo_initialized &&
                    nvshmemi_transport_cap_support_amo(nvshmemi_state->transports[j]->cap[i])) {
                    state->amo[i] = state->transports[j]->host_ops.amo;
                    amo_initialized = true;
                    state->selected_transport_for_amo[i] = j;
                }

            if (((state->selected_transport_for_amo[i] == j) ||
                     (state->selected_transport_for_rma[i] == j)) &&
                    !(memory_ordering_initialized & (1 << j))) {
                    nvshmemi_setup_memory_ordering(state, j);
                    memory_ordering_initialized |= 1 << j;
                }
            }
            tbitmap >>= 1;
        }
    }

    return status;
}

static int nvshmemi_setup_cuda_handles(nvshmemi_state_t *state) {
    int status = 0;
    state->custreams = (CUstream *)malloc(MAX_PEER_STREAMS * sizeof(CUstream));
    state->cuevents = (CUevent *)malloc(MAX_PEER_STREAMS * sizeof(CUevent));
    int leastPriority, greatestPriority;
    status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaDeviceGetStreamPriorityRange failed \n");
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        status = cuStreamCreateWithPriority(&state->custreams[i], CU_STREAM_NON_BLOCKING,
                                            greatestPriority);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuStreamCreateWithPriority failed \n");
        status = cuEventCreate(&state->cuevents[i], CU_EVENT_DISABLE_TIMING);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cuEventCreate failed \n");
    }
    state->curets = (CUdeviceptr *)malloc(state->npes * sizeof(CUdeviceptr));
    for (int i = 0; i < state->npes; i++) {
        status = cuMemAlloc(&state->curets[i], sizeof(long long));
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuMemAlloc failed \n");  // XXX:renamed curetval to curets
    }
out:
    return status;
}

static int shared_memory_create(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
  int status = 0;

  info->size = sz;

  info->shm_fd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shm_fd < 0) {
    INFO(NVSHMEM_INIT, "shm_open failed");
    return errno;
  }

  status = ftruncate(info->shm_fd, sz);
  if (status != 0) {
    INFO(NVSHMEM_INIT, "ftruncate failed");
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
  if (info->addr == NULL) {
    INFO(NVSHMEM_INIT, "mmap failed");
    return errno;
  }

  return status;
}


int shared_memory_open(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
  int status = 0;
  info->size = sz;
  struct stat stat_shm;

  info->shm_fd = shm_open(name, O_RDWR, 0777);
  if (info->shm_fd < 0) {
    return errno;
  }

  status = fstat(info->shm_fd, &stat_shm);
  if(status != 0) {
    INFO(NVSHMEM_INIT, "fstat failed");
    return status;
  }
  assert(stat_shm.st_size == (intmax_t)sz);  

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return status;
}

static void shared_memory_close(nvshmemi_shared_memory_info *info) {
    if (info->addr) {
        munmap(info->addr, info->size);
    }

    shm_unlink(shm_name);
}

static bool trimNewline(char *str)
{
    size_t len = strlen(str);
    if (len > 0 && str[len - 1] == '\n')
    {
        str[len - 1] = '\0';
    }
    return strlen(str) > 0;
}

static bool mpsServerRunning(int *serverPID)
{
    const int lineSize = 256;
    char line[lineSize];
    int ret;
    bool status = false;
    bool serverExist = false;

    FILE *output = popen("echo get_server_list | nvidia-cuda-mps-control 2> /dev/null", "r");
    if (!output) {
        INFO(NVSHMEM_INIT, "popen retuned NULL");
        return false;
    }

    while (fgets(line, lineSize, output) != NULL) {
        serverExist = true;
    }

    ret = pclose(output);
    status = (ret != -1) && WIFEXITED(ret) && (WEXITSTATUS(ret) == 0);
    if (!status) {
        INFO(NVSHMEM_INIT, "pclose retuned error");
        return false;
    }

    if (!serverExist) {
        return false;
    }

    if (!trimNewline(line)) {
        return false;
    }

    if (serverPID) {
        stringstream ss(line);
        int result;
        ss >> result;
        *serverPID = result;
    }

    return true;
}

static bool get_mps_server_active_thread_percentage(float *percentage) {
    FILE *output;
    const int lineSize = 256;
    char line[lineSize];
    int ret;
    char *retstr = NULL;
    bool status = false;
    stringstream cmd;
    int serverPID;
    /* one PE per node queries the control daemon */
    if (nvshmemi_state->mype == nvshmemi_team_node.start) {
        if (!mpsServerRunning(&serverPID)) {
            return false;
        }

        cmd << "echo get_active_thread_percentage " << serverPID << " | nvidia-cuda-mps-control";
        output = popen(cmd.str().c_str(), "r");

        if (!output) {
            return false;
        }

        retstr = fgets(line, lineSize, output);

        ret = pclose(output);
        status = (ret != -1) && WIFEXITED(ret) && (WEXITSTATUS(ret) == 0);
        if (!status || retstr == NULL) {
            return false;
        }

        if (!trimNewline(line)) {
            return false;
        }

        if (percentage) {
            int result;
            stringstream ss(line);
            ss >> result;
            *percentage = result;
        }
    }
    float *scratch = (float *)malloc(sizeof(float) * nvshmemi_state->npes);
    /* for lack of a better available bootstrap collective, using allagther */
    status = nvshmemi_state->boot_handle.allgather((void *)percentage,
                                          (void *)scratch, sizeof(float),
                                          &nvshmemi_state->boot_handle);
    *percentage = scratch[nvshmemi_team_node.start];
    free(scratch);

    return true;
}

static int nvshmemi_determine_mpg_support_level() {
    int status = 0;
    bool is_mps_server_running;
    if (nvshmemi_state->mype == nvshmemi_team_node.start) {
        is_mps_server_running = mpsServerRunning(NULL);
    }
    bool *scratch = (bool *)malloc(sizeof(bool) * nvshmemi_state->npes);
    /* for lack of a better available bootstrap collective, using allagther */
    status = nvshmemi_state->boot_handle.allgather((void *)&is_mps_server_running,
                                          (void *)scratch, sizeof(bool),
                                          &nvshmemi_state->boot_handle);
    is_mps_server_running = scratch[nvshmemi_team_node.start];
    free(scratch);

    if (!is_mps_server_running) {
        INFO(NVSHMEM_INIT, "Multiple PEs per GPU (MPG) detected but MPS is not running. "
                           "Hence limited MPG support is available");
        nvshmemi_is_limited_mpg_run = 1;
    }
    else {
        float active_thread_percentage = 0;
        bool success = get_mps_server_active_thread_percentage(&active_thread_percentage);
        if (!success) {
            INFO(NVSHMEM_INIT, "failed in get_mps_server_active_thread_percentage");
            exit(-1);
        }
        char *env = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
        if (env)
            active_thread_percentage = atof(env);

        float *active_percentages = (float *)malloc(sizeof(float) * nvshmemi_state->npes);
        status = nvshmemi_state->boot_handle.allgather((void *)&active_thread_percentage,
                                              (void *)active_percentages, sizeof(float),
                                              &nvshmemi_state->boot_handle);
        float total_percentage = 0;
        for (int i = 0; i < nvshmemi_team_same_gpu.size; i += 1) {
            total_percentage += *((float *)active_percentages +
                                  nvshmemi_team_same_gpu.start +
                                  i * nvshmemi_team_same_gpu.stride);
        }
        if (total_percentage <= 100.0) {
            nvshmemi_is_limited_mpg_run = 0;
            INFO(NVSHMEM_INIT, "Multiple PEs per GPU (MPG) detected, MPS is also available, "
                                "and active thread percentages for PEs on the same GPU add "
                                "up to be <= 100. Hence full MPG support is available");
        }
        else {
            nvshmemi_is_limited_mpg_run = 1;
            INFO(NVSHMEM_INIT, "Multiple PEs per PU (MPG) detected, MPS is also available, "
                               "but active thread percentages for PEs on the same GPU add "
                               "up to be greater than 100. Hence limited MPG support is available");
        }
        free(active_percentages);
    }
    return status;
}

static int  nvshmemi_setup_limited_mpg_support() {
    int status = 0;
    nvshmemi_mps_shmdata *shm = NULL;
    nvshmemi_shared_memory_info_t *info = &nvshmemi_state->shm_info;
    cudaEvent_t event;
    int counter = 0;

    /* Ensure supported MPS runs */
    /* Do reduction to check to that each GPU has same stride and size for team_same_gpu */
    int ret = snprintf(shm_name, 100, "mps_shm_%d", nvshmemi_team_same_gpu.start);
    if (ret < 0) {
        INFO(NVSHMEM_INIT, "snprintf failed");
        return ret;
    }

    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        if (shared_memory_create(shm_name, sizeof(nvshmemi_mps_shmdata), info) != 0) {
            ERROR_EXIT("Failed to create shared memory slab\n");
        }
    }
    status = nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);
    if (nvshmemi_team_same_gpu.start != nvshmemi_state->mype) {
        if (shared_memory_open(shm_name, sizeof(nvshmemi_mps_shmdata), info) != 0) {
            ERROR_EXIT("Failed to open shared memory slab\n");
        }
    }

    shm = (nvshmemi_mps_shmdata *)info->addr;
    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        shm->nprocesses = nvshmemi_team_same_gpu.size;
        shm->barrier = 0;
        shm->sense = 0;
    }
    CUDA_RUNTIME_CHECK(cudaEventCreate(
        &nvshmemi_state->mps_event, cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_RUNTIME_CHECK(cudaIpcGetEventHandle(
            (cudaIpcEventHandle_t *)&shm->event_handle[nvshmemi_team_same_gpu.my_pe],
            nvshmemi_state->mps_event));

    std::atomic_thread_fence(std::memory_order_seq_cst); //flush the data 
    status = nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap barrier failed \n");
    
    for(int i = 0; i < nvshmemi_team_same_gpu.size; i++) {
        if (i == nvshmemi_team_same_gpu.my_pe) continue;
        CUDA_RUNTIME_CHECK(cudaIpcOpenEventHandle(
                            &event, *(cudaIpcEventHandle_t *)&shm->event_handle[i]));
        nvshmemi_state->same_gpu_other_pe_mps_events[counter++] = event;
    }
    
    /*Close the shared memory file */
    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        if (info->shm_fd) {
            close(info->shm_fd);
        }
    }

out:
    return status;
}

static int nvshmemi_mpg_finalize(){
    shared_memory_close(&nvshmemi_state->shm_info);
    CUDA_CHECK(cuEventDestroy(nvshmemi_state->mps_event));
    return 0;
}

int nvshmemi_common_init(nvshmemi_state_t *state) {
    int status = 0;
    cpu_set_t my_set;
    CPU_ZERO(&my_set);

    if (state->initialized) return 0;
    
    CUDA_CHECK(cuDriverGetVersion(&nvshmemi_cuda_driver_version));
#if CUDART_VERSION >= 11030
    if (nvshmemi_cuda_driver_version >= 11030 && nvshmemi_options.DISABLE_CUDA_VMM == 0)
        nvshmemi_use_cuda_vmm = 1;
    else
        nvshmemi_use_cuda_vmm = 0;
#endif

    status = nvshmemi_get_cucontext(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem get cucontext failed \n");

    status = nvshmemi_detect_same_device(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect same device failed \n");

    if (nvshmemi_is_mpg_run) {
        status = nvshmemi_determine_mpg_support_level();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "call to nvshmemi_determine_mpg_support_level failed \n");
    }

    status = nvshmemi_setup_stream_priorities(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup stream priorities failed \n");

    status = nvshmemi_setup_local_heap(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup local heap failed \n");

    status = nvshmemi_transport_init(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect topo failed \n");

    status = nvshmemi_build_transport_map(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "building transport map failed \n");

    status = nvshmemi_setup_cuda_handles(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuda handles setup failed \n");

    status = nvshmemi_setup_nvshmem_handles(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem handles setup failed \n");

    status = nvshmemi_setup_connections(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup connections failed \n");

    status = nvshmemi_setup_symmetric_heap(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup heap failed \n");

    status = nvshmemi_setup_collective_launch(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup collective launch failed \n");

    status = nvshmemi_init_device_state(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem device state setup failed \n");
    
    state->initialized = 1;

    // coll init uses nvshmem_malloc directly
    // better to have state->initialized = 1
    status = nvshmemi_coll_common_cpu_init();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cpu collective setup failed \n");

    status = nvshmemi_coll_common_gpu_init();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gpu collective setup failed \n");

    if (sched_getaffinity(0, sizeof(my_set), &my_set) == 0) {
        int core_count = 0;

        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &my_set))
                core_count++;
        }

        if (core_count == 1) {
            WARN("Proxy thread shares a core with the main PE, performance may be impacted");
        }
    }

    status = nvshmemi_proxy_init(state, nvshmemi_proxy_level(state));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy initialization failed \n");

    nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);
    status = nvshmemi_team_init();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "team setup failed \n");
    
    nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);
    if (nvshmemi_is_limited_mpg_run) {
        status = nvshmemi_setup_limited_mpg_support();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mps setup failed \n");
    }
    nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);

    nvshmemi_barrier_all();
out:
    return status;
}

int nvshmemi_try_common_init(nvshmemi_state_t *state) {
    int status = 0;

    status = nvshmemi_common_init(state);

    if (status) {
        INFO(NVSHMEM_INIT, "nvshmemi_common_init failed, continuing");
        status = 0;
    }

    return status;
}

int nvshmemi_init_thread(int requested, int *provided) {
    int status = 0;

    NVSHMEMU_THREAD_CS_INIT();

    nvshmemi_init_debug();

    nvshmemi_state = (nvshmemi_state_t *)calloc(1, sizeof(nvshmemi_state_t));
    NULL_ERROR_JMP(nvshmemi_state, status, NVSHMEMX_ERROR_INTERNAL, out,
                   "nvshmemi_init_thread/calloc failed \n");

    status = nvshmemi_bootstrap(0, NULL, nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_bootstrap failed \n");

    nvshmemi_init_msg();

    nvshmemi_state->scratch = (int *)calloc(
        nvshmemi_state->npes, sizeof(int)); /*XXX:scratch used by nvshmemi_try_common_init*/

    status = nvshmemi_try_common_init(nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem common init failed \n");

    *provided = NVSHMEM_THREAD_SERIALIZED;

out:
    if (status) NVSHMEMU_THREAD_CS_FINALIZE();

    return status;
}

void nvshmem_init() {
    nvshmemi_options_init();
    nvshmem_nvtx_init();
    NVTX_FUNC_RANGE_IN_GROUP(INIT);

    int status = 0, requested = NVSHMEM_THREAD_SERIALIZED, provided;
    status = nvshmemi_init_thread(requested, &provided);
    NZ_EXIT(status, "aborting due to error in nvshmemi_init_thread \n");
}

int nvshmem_init_thread(int requested, int *provided) {
    nvshmemi_options_init();
    nvshmem_nvtx_init();
    NVTX_FUNC_RANGE_IN_GROUP(INIT);

    int status = 0;
    status = nvshmemi_init_thread(requested, provided);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_init_thread failed \n");
out:
    return status;
}

int nvshmemx_init_thread(int requested, int *provided) {
    return nvshmem_init_thread(requested, provided);
}

void nvshmem_query_thread(int *provided) { *provided = NVSHMEM_THREAD_SERIALIZED; }

void nvshmemx_query_thread(int *provided) { nvshmem_query_thread(provided); }

int nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t *attr) {
    nvshmemi_options_init();
    nvshmem_nvtx_init();
    NVTX_FUNC_RANGE_IN_GROUP(INIT);
    int status;

    NVSHMEMU_THREAD_CS_INIT();

    nvshmemi_init_debug();

    nvshmemi_state = (nvshmemi_state_t *)calloc(1, sizeof(nvshmemi_state_t));
    p_err_str = (char *)malloc(MAX_LENGTH_ERROR_STRING);

    status = nvshmemi_bootstrap(flags, attr, nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_bootstrap failed \n");

    nvshmemi_init_msg();

    status = nvshmemi_try_common_init(nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem topo init failed \n");

    nvshmemi_state->scratch = (int *)calloc(nvshmemi_state->npes, sizeof(int));

out:
    return status;
}

void nvshmem_global_exit(int status) {
    nvshmemi_state->boot_handle.global_exit(status);
}

void nvshmem_finalize() {
    NVTX_FUNC_RANGE_IN_GROUP(INIT);
    int status = 0;
    int pid = getpid();

    INFO(NVSHMEM_INIT, "[%d] in nvshmem_finalize:", pid);

    if (nvshmemi_state->initialized) {
        nvshmemi_barrier_all();
        nvshmemx_quiet_on_stream(nvshmemi_state->my_stream); /* wait for signal ops from barrier to complete */
        status = cudaDeviceSynchronize();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Teams cleanup device synchronization failed \n");
        
        /* barrier to ensure all previous ops are complete */
        nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);

        /* mps finalize */
        if (nvshmemi_is_limited_mpg_run) {
            status = nvshmemi_mpg_finalize();
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "MPS cleanup failed \n");
        }

        /* teams cleanup */
        status = nvshmemi_team_finalize();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Teams cleanup failed \n");

        /*cleaning up proxy*/
        if (nvshmemi_proxy_level(nvshmemi_state)) {
            status = nvshmemi_proxy_finalize(nvshmemi_state);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy cleanup failed \n");
        }
        
        /* collective cleanup */
        status = nvshmemi_coll_common_cpu_finalize();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "CPU collectives cleanup failed \n");

        status = nvshmemi_coll_common_gpu_finalize();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "GPU collectives cleanup failed \n");

        status = nvshmemi_teardown_handles(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "handles cleanup failed \n");

        status = nvshmemi_cleanup_symmetric_heap(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "symmetric heap cleanup failed \n");

        status = nvshmemi_transport_finalize(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem transport finalize failed \n");

        status = nvshmemi_teardown_collective_launch(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "collective launch cleanup failed \n");
    } else
        nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);

    status = bootstrap_finalize(&nvshmemi_state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_finalize failed \n");

    /* cleanup state */
    if (nvshmemi_state->scratch_size) free(nvshmemi_state->scratch_space);

    if (nvshmemi_state->scratch) free(nvshmemi_state->scratch);
    free(nvshmemi_state);

    NVSHMEMU_THREAD_CS_FINALIZE();

out:
    if (status) {
        ERROR_PRINT("aborting due to error in nvshmem_finalize \n");
        exit(-1);
    }
}

static void nvshmemi_init_debug() {
    const char *nvshmem_debug = nvshmemi_options.DEBUG;
    if (!nvshmemi_options.DEBUG_provided && !nvshmemi_options.DEBUG_SUBSYS_provided) {
        nvshmem_debug_level = NVSHMEM_LOG_NONE;
    } else if (strcmp_case_insensitive(nvshmem_debug, "VERSION") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_VERSION;
    } else if (strcmp_case_insensitive(nvshmem_debug, "WARN") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_WARN;
    } else if (strcmp_case_insensitive(nvshmem_debug, "INFO") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    } else if (strcmp_case_insensitive(nvshmem_debug, "ABORT") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_ABORT;
    } else if (strcmp_case_insensitive(nvshmem_debug, "TRACE") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_TRACE;
    }
    else {
        /* OpenSHMEM spec treats SHMEM_DEBUG as a boolean, enable INFO logging
         * when user-supplied value does match one of the above. */
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    }

    /* Parse the NVSHMEM_DEBUG_SUBSYS env var
     * This can be a comma separated list such as INIT,COLL
     * or ^INIT,COLL etc
     */
    /* Note: strtok will modify the string, operate on a copy */
    char *nvshmem_debug_subsys = strdup(nvshmemi_options.DEBUG_SUBSYS);
    if (nvshmem_debug_subsys != NULL) {
        char *subsys = strtok(nvshmem_debug_subsys, ",");
        while (subsys != NULL) {
            int invert = 0;
            uint64_t mask = 0;
            if (subsys[0] == '^') {
                invert = 1;
                subsys++;
            }
            if (strcmp_case_insensitive(subsys, "INIT") == 0) {
                mask = NVSHMEM_INIT;
            } else if (strcmp_case_insensitive(subsys, "COLL") == 0) {
                mask = NVSHMEM_COLL;
            } else if (strcmp_case_insensitive(subsys, "P2P") == 0) {
                mask = NVSHMEM_P2P;
            } else if (strcmp_case_insensitive(subsys, "PROXY") == 0) {
                mask = NVSHMEM_PROXY;
            } else if (strcmp_case_insensitive(subsys, "TRANSPORT") == 0) {
                mask = NVSHMEM_TRANSPORT;
            } else if (strcmp_case_insensitive(subsys, "MEM") == 0) {
                mask = NVSHMEM_MEM;
            } else if (strcmp_case_insensitive(subsys, "BOOTSTRAP") == 0) {
                mask = NVSHMEM_BOOTSTRAP;
            } else if (strcmp_case_insensitive(subsys, "TOPO") == 0) {
                mask = NVSHMEM_TOPO;
            } else if (strcmp_case_insensitive(subsys, "UTIL") == 0) {
                mask = NVSHMEM_UTIL;
            } else if (strcmp_case_insensitive(subsys, "ALL") == 0) {
                mask = NVSHMEM_ALL;
            } else {
                mask = 0;
                WARN("Unrecognized value in DEBUG_SUBSYS: %s%s", invert ? "^" : "", subsys);
            }
            if (mask) {
                if (invert)
                    nvshmem_debug_mask &= ~mask;
                else
                    nvshmem_debug_mask |= mask;
            }
            subsys = strtok(NULL, ",");
        }

        free(nvshmem_debug_subsys);
    }

    /* Parse and expand the NVSHMEM_DEBUG_FILE path and
     * then create the debug file. But don't bother unless the
     * NVSHMEM_DEBUG level is > VERSION
     */
    const char *nvshmem_debug_filename = nvshmemi_options.DEBUG_FILE;
    if (nvshmem_debug_level > NVSHMEM_LOG_VERSION && nvshmemi_options.DEBUG_FILE_provided) {
        int c = 0;
        char debugFn[PATH_MAX + 1] = "";
        char *dfn = debugFn;
        while (nvshmem_debug_filename[c] != '\0' && c < PATH_MAX) {
            if (nvshmem_debug_filename[c++] != '%') {
                *dfn++ = nvshmem_debug_filename[c - 1];
                continue;
            }
            switch (nvshmem_debug_filename[c++]) {
                case '%':  // Double %
                    *dfn++ = '%';
                    break;
                case 'h':  // %h = hostname
                    char hostname[1024];
                    nvshmemu_gethostname(hostname, 1024);
                    dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
                    break;
                case 'p':  // %p = pid
                    dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
                    break;
                default:  // Echo everything we don't understand
                    *dfn++ = '%';
                    *dfn++ = nvshmem_debug_filename[c - 1];
                    break;
            }
        }
        *dfn = '\0';
        if (debugFn[0] != '\0') {
            FILE *file = fopen(debugFn, "w");
            if (file != NULL) {
                INFO(NVSHMEM_ALL, "DEBUG file is '%s'", debugFn);
                nvshmem_debug_file = file;
            }
        }
    }
    pthread_mutex_init(&nvshmem_debug_output_lock, NULL);

#ifdef ENABLE_TRACE
    nvshmem_epoch = std::chrono::high_resolution_clock::now();
#endif
}

static void nvshmemi_init_msg(void) {
    if (0 == nvshmemi_state->mype) {
        if (nvshmemi_options.VERSION)
            printf("%s\n", NVSHMEM_VENDOR_STRING);

        if (nvshmemi_options.DEBUG_provided) {
            int runtimeVersion, driverVersion;
            cudaError_t err;

            printf("NVSHMEM configuration:\n");

            printf("  %-28s %d\n", "CUDA API", CUDA_VERSION);

            err = cudaRuntimeGetVersion(&runtimeVersion);
            if (err != cudaSuccess) runtimeVersion = -1;
            printf("  %-28s %d\n", "CUDA Runtime", runtimeVersion);

            err = cudaDriverGetVersion(&driverVersion);
            if (err != cudaSuccess) driverVersion = -1;
            printf("  %-28s %d\n", "CUDA Driver", driverVersion);

            printf("  %-28s %s %s\n", "Build Timestamp", __DATE__, __TIME__);
            printf("  %-28s%s", "Build Options",
#ifdef NVSHMEM_X86_64
                    " NVSHMEM_X86_64"
#endif
#ifdef NVSHMEM_PPC64LE
                    " NVSHMEM_PPC64LE"
#endif
#ifdef NVSHMEM_MPI_SUPPORT
                    " NVSHMEM_MPI_SUPPORT"
#endif
#ifdef NVSHMEM_SHMEM_SUPPORT
                    " NVSHMEM_SHMEM_SUPPORT"
#endif
#ifdef NVSHMEM_PMIX_SUPPORT
                    " NVSHMEM_PMIX_SUPPORT"
#endif
#ifdef NVSHMEM_DEFAULT_PMIX
                    " NVSHMEM_DEFAULT_PMIX"
#endif
#ifdef NVSHMEM_DEFAULT_PMI2
                    " NVSHMEM_DEFAULT_PMI2"
#endif
#ifdef NVSHMEM_USE_GDRCOPY
                    " NVSHMEM_USE_GDRCOPY"
#endif
#ifdef NVSHMEM_COMPLEX_SUPPORT
                    " NVSHMEM_COMPLEX_SUPPORT"
#endif
#ifdef NVSHMEM_DISABLE_COLL_POLL
                    " NVSHMEM_DISABLE_COLL_POLL"
#endif
#ifdef NVSHMEM_GPU_COLL_USE_LDST
                    " NVSHMEM_GPU_COLL_USE_LDST"
#endif
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
                    " NVSHMEM_TIMEOUT_DEVICE_POLLING"
#endif
#ifdef NVSHMEM_ENABLE_TRACE
                    " NVSHMEM_ENABLE_TRACE"
#endif
                    "\n");

            printf("\n");
        }
        if (nvshmemi_options.INFO)
            nvshmemi_options_print();
    }

    if (nvshmemi_options.DEBUG_provided || nvshmemi_options.DEBUG_SUBSYS_provided)
        nvshmemu_debug_log_cpuset(NVSHMEM_INIT, "process");
}
