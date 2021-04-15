/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"

#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <map>
#include "nvshmem_internal.h"
#include "nvshmemx_error.h"
#include "nvshmem_nvtx.hpp"
#include "dlmalloc.h"
#include "util.h"
#include "sockets.h"
#include "nvshmemi_team.h"

#define IPC_CHECK(ipcFuncResult)                \
    if (ipcFuncResult == -1) {                  \
        fprintf(stderr, "Failure at %u %s\n",   \
        __LINE__, __FILE__);                    \
        exit(EXIT_FAILURE);                     \
    }

#define CUMEM_ALIGNMENT (1 << 29) // 512 MB
static int is_mem_handle_null(nvshmem_mem_handle_t handle) {
    assert(sizeof(nvshmem_mem_handle_t) % sizeof(uint64_t) == 0);

    for (size_t i = 0; i < (sizeof(nvshmem_mem_handle_t) / sizeof(uint64_t)); i++) {
        if (*((uint64_t *)&handle + i) != (uint64_t)0) return 0;
    }

    return 1;
}

static int cleanup_local_handles(nvshmem_mem_handle_t *handles, nvshmemi_state_t *state) {
    int status = 0;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (state->transport_bitmap & (1 << i)) {
            if (!is_mem_handle_null(state->handles[i])) {
                status = state->transports[i]->host_ops.release_mem_handle(handles[i], state->transports[i]);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                             "transport release memhandle failed \n");
            }
        }
    }

out:
    return status;
}

template <typename T>
int check_for_symmetry(T value) {
    int status = 0;
    nvshmemi_state_t *state = nvshmemi_state;

    /*TODO: need to handle multi-threaded scenarios*/

    if (!nvshmemi_options.ENABLE_ERROR_CHECKS) return 0;

    if (state->scratch_size < sizeof(T) * state->npes) {
        if (state->scratch_size) free(state->scratch_space);

        state->scratch_space = (char *)malloc(sizeof(T) * state->npes);
        NULL_ERROR_JMP(state->scratch_space, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                       "error allocating scratch space \n");
        state->scratch_size = sizeof(T) * state->npes;
    }

    status = state->boot_handle.allgather((void *)&value, (void *)state->scratch_space, sizeof(T),
                                          &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather in symmetry check failed \n");

    for (int i = 0; i < state->npes; i++) {
        status = (*((T *)state->scratch_space + i) == value) ? 0 : 1;
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_SYMMETRY, out, "symmetry check failed \n");
    }

out:
    return status;
}

int mspace_track_large_chunks(mspace msp, int enable);
size_t destroy_mspace(mspace msp);

int nvshmemi_setup_memory_space(nvshmemi_state_t *state) {
    int status = 0;
    mspace heap_mspace = 0;

    heap_mspace = create_mspace_with_base(state->heap_base, state->heap_size, 0);
    NULL_ERROR_JMP(heap_mspace, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "mspace creation failed \n");

    assert(heap_mspace != 0);
    INFO(NVSHMEM_INIT, "[%d] mspace ptr: %p", state->mype, heap_mspace);

    mspace_track_large_chunks(heap_mspace, 1);

    state->heap_mspace = heap_mspace;

out:
    return status;
}

int nvshmemi_cleanup_memory_space(nvshmemi_state_t *state) {
    int status = 0;
    size_t size;

    size = destroy_mspace(state->heap_mspace);
    status = (size == state->heap_size) ? 0 : 1;
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "destroy_mspace failed \n");

out:
    return status;
}

int nvshmemi_cleanup_symmetric_heap(nvshmemi_state_t *state) {
    int status = 0;

    if (!state->peer_heap_base) goto out;

    // TODO: work required in destroying mspace
    // status = nvshmemi_cleanup_memory_space (state);
    // NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "memory space cleanup failed \n");

    for (int i = 0; i < state->npes; i++) {
        if ((i == state->mype) && (state->heap_base != NULL)) {
            status = cleanup_local_handles(state->handles + i * NVSHMEM_TRANSPORT_COUNT, state);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "cleanup local handles failed \n");
#if CUDART_VERSION >= 11030
            if (nvshmemi_cuda_driver_version >= 11030) {
                status = cuMemRelease(state->heap_handle);
                NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemRelease failed \n");
            } else
#endif
            {
                status = cuMemFree((CUdeviceptr)state->peer_heap_base[i]);
                NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemFree failed \n");
            }

            continue;
        }

        if (state->peer_heap_base[i]) {
            int j;
            for (j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
                if ((((state->transport_bitmap) & (1 << j)) &&
                     (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP)) == 0)
                    continue;

                status = state->transports[j]->host_ops.unmap((state->peer_heap_base[i]), state->heap_size);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "unmap failed \n");

                close(*(int *)&state->handles[i * NVSHMEM_TRANSPORT_COUNT]);
            }
        }
    }

    free(state->peer_heap_base);

out:
    return status;
}

int nvshmemi_setup_local_heap(nvshmemi_state_t *state) {
    int status;
#if CUDART_VERSION >= 11030
    size_t granularity;
    CUmemAllocationProp prop = {};
#endif

    state->heap_size = nvshmemi_options.SYMMETRIC_SIZE;
    size_t heapextra = NUM_G_BUF_ELEMENTS * sizeof(g_elem_t) + nvshmemi_get_teams_mem_requirement();
    heapextra += G_COALESCING_BUF_SIZE;
    size_t alignbytes = MALLOC_ALIGNMENT;

    int warp_size = 0;
    status = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, state->cudevice );
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "querying warp size failed \n");
    if ( NVSHMEMI_WARP_SIZE != warp_size ) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                     "device warp size (%d) does not match assumed warp size (%d)\n",
                     warp_size, NVSHMEMI_WARP_SIZE);

    }

#if CUDART_VERSION >= 11030
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<int>(state->device_id);
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuMemGetAllocationGranularity failed \n");
    alignbytes = (alignbytes < granularity) ? granularity : alignbytes;
#endif

    if (heapextra % alignbytes) {
        heapextra = ((heapextra + alignbytes - 1) / alignbytes) * alignbytes;
    }
    state->heap_size += (heapextra + 4 * alignbytes +
                         20 * alignbytes);  // XXX:each allocation from SHEAP could be padded by 2x
                                            // alignbytes, providing capacity for 2 allocations for
                                            // the library and 10 allocations for the user
    INFO(NVSHMEM_INIT, "nvshmemi_setup_local_heap %lld", heapextra);
    state->heap_size = ((state->heap_size + CUMEM_ALIGNMENT - 1) / CUMEM_ALIGNMENT) * CUMEM_ALIGNMENT;

#if CUDART_VERSION >= 11030
    if (nvshmemi_cuda_driver_version >= 11030) {
        status = cuMemAddressReserve((CUdeviceptr *)&state->global_heap_base, state->heap_size * state->npes,
                                     alignbytes, (CUdeviceptr)NULL, 0);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuMemAddressReserve failed \n");

        status =
            cuMemCreate(&state->heap_handle, state->heap_size, (const CUmemAllocationProp *)&prop, 0);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemCreate failed \n");

        state->heap_base = (void *)((uintptr_t)state->global_heap_base + state->mype * state->heap_size);

        status = cuMemMap((CUdeviceptr)state->heap_base, state->heap_size, 0, state->heap_handle, 0);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemMap failed \n");

        CUmemAccessDesc access;
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = state->device_id;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        status = cuMemSetAccess((CUdeviceptr)state->heap_base, state->heap_size,
                                (const CUmemAccessDesc *)&access, 1);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemSetAccess failed \n");

        INFO(NVSHMEM_INIT, "[%d] heap base: %p NVSHMEM_SYMMETRIC_SIZE %lu total %lu heapextra %lu\n",
             state->mype, state->heap_base, nvshmemi_options.SYMMETRIC_SIZE, state->heap_size,
             heapextra);
    } else
#endif
    {
        bool data =
            true; /*A boolean attribute which when set, ensures that synchronous memory operations
                     initiated on the region of memory that ptr points to will always synchronize.*/

        status = cuMemAlloc((CUdeviceptr *)&state->heap_base, state->heap_size);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cuMemAlloc failed \n");

        INFO(NVSHMEM_INIT, "[%d] heap baseE: %p NVSHMEM_SYMMETRIC_SIZE %lu total %lu heapextra %lu",
             state->mype, state->heap_base, nvshmemi_options.SYMMETRIC_SIZE, state->heap_size,
             heapextra);

        status = cuPointerSetAttribute(&data, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                       (CUdeviceptr)state->heap_base);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                     "cuPointerSetAttribute failed \n");
    }

    status = nvshmemi_setup_memory_space(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "memory space initialization failed \n");

out:
    if (status) {
        if (state->heap_base) cuMemFree((CUdeviceptr)state->heap_base);
    }
    return status;
}

int nvshmemi_setup_symmetric_heap(nvshmemi_state_t *state) {
    int status;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_mem_handle_t local_handles[NVSHMEM_TRANSPORT_COUNT];

#if CUDART_VERSION >= 11030
    pid_t pid;
    pid_t *peer_pids;
    uint64_t myHostHash;
    uint64_t *hostHash;
    std::map<pid_t, int> processes;
    ipcHandle *myIpcHandle = NULL;
#endif

    memset(local_handles, 0, sizeof(nvshmem_mem_handle_t) * NVSHMEM_TRANSPORT_COUNT);

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if ((state->transport_bitmap & (1 << i)) && transports[i]->host_ops.get_mem_handle) {
            INFO(NVSHMEM_INIT, "calling get_mem_handle for transport: %d buf: %p size: %d", i,
                 state->heap_base, state->heap_size);
            status = transports[i]->host_ops.get_mem_handle(
                (nvshmem_mem_handle_t *)(local_handles + i),
                *(nvshmem_mem_handle_t *)&state->heap_handle /*used only by p2p transport*/,
                state->heap_base, state->heap_size, transports[i]);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport get memhandle failed \n");
            INFO(NVSHMEM_INIT, "[%d] get_mem_handle transport %d handles %p", state->mype, i,
                 local_handles + i);
        }
    }

    // assuming symmetry of transports across all PEs
    state->handles = (nvshmem_mem_handle_t *)calloc(NVSHMEM_TRANSPORT_COUNT * state->npes,
                                                    sizeof(nvshmem_mem_handle_t));
    NULL_ERROR_JMP(state->handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for peer heap base \n");

    status = state->boot_handle.allgather((void *)local_handles, (void *)state->handles,
                                          sizeof(nvshmem_mem_handle_t) * NVSHMEM_TRANSPORT_COUNT,
                                          &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of mem handles failed \n");

    state->peer_heap_base_actual = (void **)calloc(state->npes, sizeof(void *));
    NULL_ERROR_JMP(state->peer_heap_base_actual, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for peer heap base \n");

    status = state->boot_handle.allgather((void *)&state->heap_base,
                                          (void *)state->peer_heap_base_actual, sizeof(void *),
                                          &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                 "allgather of heap base ptrsmem handle failed \n");

    state->peer_heap_base = (void **)calloc(state->npes, sizeof(void *));
    NULL_ERROR_JMP(state->peer_heap_base, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for peer heap base \n");
    
#if CUDART_VERSION >= 11030
    if (nvshmemi_cuda_driver_version >= 11030) {
        pid = getpid();
        peer_pids = (pid_t *) malloc(sizeof(pid_t) * state->npes);
        status = state->boot_handle.allgather((void *)&pid, (void *)peer_pids, sizeof(pid_t), &state->boot_handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of pids failed \n");
            
        myHostHash = getHostHash();
        hostHash = (uint64_t *)malloc(sizeof(uint64_t) * state->npes);
        status = state->boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                              &state->boot_handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of host hashes failed \n");

        for(int pe = 0; pe < state->npes; pe++) {
            if (hostHash[pe] == myHostHash) {
                processes[peer_pids[pe]] = pe;
            }
        }
        IPC_CHECK(ipcOpenSocket(myIpcHandle));
        
        /* Wait for all processes to open their sockets */
        status = state->boot_handle.barrier(&state->boot_handle);

        for (std::map<pid_t, int>::iterator it1 = processes.begin(); it1 != processes.end(); ++it1) {
            pid_t sending_process = it1->first;
            if (pid == sending_process) {
                for (std::map<pid_t, int>::iterator it2 = processes.begin(); it2 != processes.end(); ++it2) {
                    pid_t receiving_process = it2->first;
                    if (pid != receiving_process) {/* Dont sent to yourself */
                        IPC_CHECK(ipcSendFd(myIpcHandle, *(int *)local_handles, receiving_process));
                    }
                }
            } else  {
               IPC_CHECK(ipcRecvFd(myIpcHandle, (int *)&state->handles[it1->second * NVSHMEM_TRANSPORT_COUNT]));
            }
            /* Putting a global barrier means assuming that all nodes are running with config */
            status = state->boot_handle.barrier(&state->boot_handle);
        }
        IPC_CHECK(ipcCloseSocket(myIpcHandle));
    }
#endif

    // pick a transport that allows LD/ST and map to get local pointers to remote heaps
    // assumes symmetry of handles received from each peer, should be true (mostly)
    for (int i = 0; i < state->npes; i++) {
        if (i == state->mype) {
            state->peer_heap_base[i] = state->heap_base;
            continue;
        }

        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            if (state->transport_map[state->mype * state->npes + i] & (1 << j)) {
                if (transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP) {
#if CUDART_VERSION >= 11030
                    if (nvshmemi_cuda_driver_version >= 11030)
                        state->peer_heap_base[i] = (void *)((uintptr_t)state->global_heap_base 
                                                            + state->heap_size * i);
#endif
                    status = transports[j]->host_ops.map(
                        (state->peer_heap_base + i), state->heap_size,
                        state->handles[i * NVSHMEM_TRANSPORT_COUNT + j]);
                    if (status) {
                           //map operation failed, remove cap of transport
                           state->transports[j]->cap[i] ^= NVSHMEM_TRANSPORT_CAP_MAP;
                           status = 0; 
                           continue;
                    }

                    char *hex = nvshmemu_hexdump(&state->handles[i * NVSHMEM_TRANSPORT_COUNT + j],
                                                 sizeof(CUipcMemHandle));
                    INFO(NVSHMEM_INIT, "[%d] cuIpcOpenMemHandle fromhandle 0x%s", state->mype, hex);
                    free(hex);

                    INFO(NVSHMEM_INIT, "[%d] cuIpcOpenMemHandle tobuf %p", state->mype,
                         *(state->peer_heap_base + i));
                    break;
                }
            }
        }
    }

out:
    if (status) {
        // if handles has been allocated, try and cleanup all heap state
        // else cleanup local handles only
        if (state->handles) {
            nvshmemi_cleanup_symmetric_heap(state);
        } else {
            cleanup_local_handles(local_handles, state);
            if (state->heap_base) cuMemFree((CUdeviceptr)state->heap_base);
        }
    }
    return status;
}

void *nvshmemi_malloc(size_t size) {
    int status = 0;
    void *ptr = NULL;

    status = check_for_symmetry(size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "symmetry check for size failed\n");

    ptr = mspace_malloc(nvshmemi_state->heap_mspace, size);
    if ((size > 0) && (ptr == NULL)) {
        ERROR_EXIT("nvshmem malloc failed (hint: check if total allocation has exceeded NVSHMEM "
                   "symmetric size = %zu, NVSHMEM symmetric size can be increased using "
                   "NVSHMEM_SYMMETRIC_SIZE environment variable) \n", nvshmemi_options.SYMMETRIC_SIZE);
    }

    INFO(NVSHMEM_INIT, "[%d] allocated %d bytes from mspace: %p ptr: %p", nvshmemi_state->mype,
         size, nvshmemi_state->heap_mspace, ptr);
out:
    return ptr;
}

void *nvshmemi_calloc(size_t count, size_t size) {
    int status = 0;
    void *ptr = NULL;

    status = check_for_symmetry(size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "symmetry check for size failed\n");

    ptr = mspace_calloc(nvshmemi_state->heap_mspace, count, size);
    if (size > 0 && count > 0 && ptr == NULL) {
        ERROR_EXIT("nvshmem calloc failed \n");
    }

    INFO(NVSHMEM_INIT, "[%d] calloc allocated %d bytes from mspace: %p ptr: %p",
            nvshmemi_state->mype, size, nvshmemi_state->heap_mspace, ptr);
out:
    return ptr;
}

void *nvshmem_malloc(size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();

    NVSHMEM_CHECK_STATE_AND_INIT();

    ptr = nvshmemi_malloc(size);

    nvshmemi_barrier_all();

    NVSHMEMU_THREAD_CS_EXIT();

    return ptr;
}

void *nvshmem_calloc(size_t count, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();

    NVSHMEM_CHECK_STATE_AND_INIT();

    ptr = nvshmemi_calloc(count, size);

    nvshmemi_barrier_all();

    NVSHMEMU_THREAD_CS_EXIT();

    return ptr;
}

void *nvshmemi_align(size_t alignment, size_t size) {
    void *ptr = NULL;
    int status = 0;

    status = check_for_symmetry(size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "symmetry check for size failed\n");

    ptr = mspace_memalign(nvshmemi_state->heap_mspace, alignment, size);
    if ((size > 0) && (ptr == NULL)) {
        ERROR_EXIT("nvshmem align failed \n");
    }

out:
    return ptr;
}

void *nvshmem_align(size_t alignment, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();

    NVSHMEM_CHECK_STATE_AND_INIT();

    ptr = nvshmemi_align(alignment, size);

    nvshmemi_barrier_all();

    NVSHMEMU_THREAD_CS_EXIT();

    return ptr;
}

void nvshmemi_free(void *ptr) {
    if (ptr == NULL) return;

    INFO(NVSHMEM_INIT, "[%d] freeing buf: %p", nvshmemi_state->mype, ptr);

    mspace_free(nvshmemi_state->heap_mspace, ptr);
}

void nvshmem_free(void *ptr) {
    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();

    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmemi_barrier_all();

    nvshmemi_free(ptr);

    NVSHMEMU_THREAD_CS_EXIT();
}

void *nvshmem_ptr(const void *ptr, int pe) {
    if (ptr >= nvshmemi_state->heap_base) {
        uintptr_t offset = (char*)ptr - (char*)nvshmemi_state->heap_base;

        if (offset < nvshmemi_state->heap_size) {
            void *peer_addr = nvshmemi_state->peer_heap_base[pe];
            if (peer_addr != NULL)
                peer_addr = (void *)((char *)peer_addr + offset);
            return peer_addr;
        }
    }

    return NULL;
}
