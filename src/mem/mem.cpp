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
#include <algorithm>
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

size_t cumem_granularity;
size_t log2_cumem_granularity;
static std::map<pid_t, int>
    p2p_processes; /* Map from p2p processes to PE id - required when using VMM */
static int is_mem_handle_null(nvshmem_mem_handle_t *handle) {
    assert(sizeof(nvshmem_mem_handle_t) % sizeof(uint64_t) == 0);

    for (size_t i = 0; i < (sizeof(nvshmem_mem_handle_t) / sizeof(uint64_t)); i++) {
        if (*((uint64_t *)handle + i) != (uint64_t)0) return 0;
    }

    return 1;
}

static int cleanup_local_handles(nvshmem_mem_handle_t *handles, nvshmemi_state_t *state) {
    int status = 0;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (state->transport_bitmap & (1 << i)) {
            if (!is_mem_handle_null(&handles[i])) {
                status = state->transports[i]->host_ops.release_mem_handle(&handles[i], state->transports[i]);
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

    status = nvshmemi_boot_handle.allgather((void *)&value, (void *)state->scratch_space, sizeof(T),
                                          &nvshmemi_boot_handle);
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

int nvshmemi_setup_memory_space(nvshmemi_state_t *state, void *heap_base, size_t size) {
    int status = 0;
    mspace heap_mspace = 0;

    heap_mspace = create_mspace_with_base(heap_base, size, 0);
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
    destroy_mspace(state->heap_mspace);
    return 0;
}

int nvshmemi_cleanup_symmetric_heap(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_cleanup_symmetric_heap()");
    int status = 0;

    if (!state->peer_heap_base) goto out;

    // TODO: work required in destroying mspace
    // status = nvshmemi_cleanup_memory_space (state);
    // NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "memory space cleanup failed \n");

    for (int i = 0; i < state->npes; i++) {
        if ((i == state->mype) && (state->heap_base != NULL)) {
            for (size_t j = 0; j < state->handles.size(); j++) {
                status =
                    cleanup_local_handles(&state->handles[j][i * NVSHMEM_TRANSPORT_COUNT], state);
                NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cleanup local handles failed \n");
            }
#if CUDART_VERSION >= 11030
            if (nvshmemi_use_cuda_vmm) {
                for (uint32_t i = 0; i < state->cumem_handles.size(); i++) {
                    status = cuMemRelease(state->cumem_handles[i]);
                    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemRelease failed \n");
                }
                state->cumem_handles.clear();
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

                for (size_t k = 0; k < state->handles.size(); k++) {
                    close(*(int *)&state->handles[k][i * NVSHMEM_TRANSPORT_COUNT]);
                }
            }
        }
    }
    state->handles.clear();
    state->idx_in_handles.clear();
    nvshmemi_cleanup_memory_space(state);

    free(state->peer_heap_base);
    INFO(NVSHMEM_INIT, "Leaving nvshmemi_cleanup_symmetric_heap()");

out:
    return status;
}

int nvshmemi_setup_local_heap(nvshmemi_state_t *state) {
    int status;
    size_t alignbytes = MALLOC_ALIGNMENT;
    size_t heapextra;
    size_t tmp;

    cumem_granularity = nvshmemi_options.CUMEM_GRANULARITY;
#if CUDART_VERSION >= 11030
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<int>(state->device_id);
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    status = cuMemGetAllocationGranularity(&cumem_granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuMemGetAllocationGranularity failed \n");
    cumem_granularity = std::max(nvshmemi_options.CUMEM_GRANULARITY, cumem_granularity);
#endif

    assert((cumem_granularity & (cumem_granularity - 1)) == 0);
    tmp = cumem_granularity;
    log2_cumem_granularity = 0;
    while (tmp >> 1) { tmp >>= 1; log2_cumem_granularity++; }

    heapextra = NUM_G_BUF_ELEMENTS * sizeof(g_elem_t) +
                nvshmemi_get_teams_mem_requirement() +
                G_COALESCING_BUF_SIZE +
                4 * alignbytes +
                20 * alignbytes; // alignbytes, providing capacity for 2 allocations for
                                // the library and 10 allocations for the user
    
    INFO(NVSHMEM_INIT, "nvshmemi_setup_local_heap, heapextra = %lld", heapextra);
    if (nvshmemi_use_cuda_vmm) {
        state->heap_size = std::max(nvshmemi_options.MAX_MEMORY_PER_GPU, heapextra);
    } else {
        state->heap_size = nvshmemi_options.SYMMETRIC_SIZE + heapextra;
    }
    state->heap_size =
        ((state->heap_size + cumem_granularity - 1) / cumem_granularity) * cumem_granularity;

#if CUDART_VERSION >= 11030
    if (nvshmemi_use_cuda_vmm) {
        status = cuMemAddressReserve((CUdeviceptr *)&state->global_heap_base,
                                     nvshmemi_options.MAX_P2P_GPUS * state->heap_size,
                                     alignbytes, (CUdeviceptr)NULL, 0);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuMemAddressReserve failed \n");

        state->heap_base = (void *)((uintptr_t)state->global_heap_base);
        state->physical_heap_size = 0;

        status = nvshmemi_setup_memory_space(state, state->heap_base, state->physical_heap_size);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "memory space initialization failed \n");
    } else
#endif
    {
        bool data =
            true; /*A boolean attribute which when set, ensures that synchronous memory operations
                     initiated on the region of memory that ptr points to will always synchronize.*/

        status = cuMemAlloc((CUdeviceptr *)&state->heap_base, state->heap_size);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cuMemAlloc failed \n");

        status = cuPointerSetAttribute(&data, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                       (CUdeviceptr)state->heap_base);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                     "cuPointerSetAttribute failed \n");

        INFO(NVSHMEM_INIT, "[%d] heap base: %p NVSHMEM_SYMMETRIC_SIZE %lu total %lu heapextra %lu",
             state->mype, state->heap_base, nvshmemi_options.SYMMETRIC_SIZE, state->heap_size,
             heapextra);

        status = nvshmemi_setup_memory_space(state, state->heap_base, state->heap_size);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "memory space initialization failed \n");
    }

out:
    if (status) {
        if (state->heap_base && nvshmemi_use_cuda_vmm==0)
            cuMemFree((CUdeviceptr)state->heap_base);
    }
    return status;
}

#ifdef NVSHMEM_GPUINITIATED_SUPPORT
int nvshmemi_gather_mem_handles(nvshmem_mem_handle_t *local_handles, nvshmemi_state_t *state, uint64_t heap_offset, size_t size) 
#else
int nvshmemi_gather_mem_handles(nvshmem_mem_handle_t *local_handles, nvshmemi_state_t *state) 
#endif
{
    int status = nvshmemi_boot_handle.allgather(
        (void *)local_handles, (void *)(state->handles.back().data()),
        sizeof(nvshmem_mem_handle_t) * NVSHMEM_TRANSPORT_COUNT, &nvshmemi_boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of mem handles failed \n");

    #ifdef NVSHMEM_GPUINITIATED_SUPPORT
    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; ++i) {
        nvshmem_transport_t tcurr = state->transports[i];
        if ((state->transport_bitmap & (1 << i)) && tcurr->host_ops.add_device_remote_mem_handles) {
            status = tcurr->host_ops.add_device_remote_mem_handles(tcurr, i, state->handles.back().data(), heap_offset, size);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "add_device_remote_mem_handles failed \n");
        }
    }
    #endif

out:
    return status;
}

int nvshmemi_setup_symmetric_heap(nvshmemi_state_t *state) {
    int status;
    int p2p_counter;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_mem_handle_t local_handles[NVSHMEM_TRANSPORT_COUNT];
    // assuming symmetry of transports across all PEs
    memset(local_handles, 0, sizeof(nvshmem_mem_handle_t) * NVSHMEM_TRANSPORT_COUNT);

    if (!nvshmemi_use_cuda_vmm){
        nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;

        for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
            if ((state->transport_bitmap & (1 << i)) && transports[i]->host_ops.get_mem_handle) {
                INFO(NVSHMEM_INIT, "calling get_mem_handle for transport: %d buf: %p size: %lu", i,
                     state->heap_base, state->heap_size);
                status = transports[i]->host_ops.get_mem_handle(
                    (nvshmem_mem_handle_t *)(local_handles + i),
                    local_handles /*dummy - used only by p2p transport when using vmm*/,
                    state->heap_base, state->heap_size, transports[i], false);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport get memhandle failed \n");
                INFO(NVSHMEM_INIT, "[%d] get_mem_handle transport %d handles %p", state->mype, i,
                     local_handles + i);
            }
        }
    
        state->handles.push_back(vector<nvshmem_mem_handle_t>(NVSHMEM_TRANSPORT_COUNT * state->npes));
        #ifdef NVSHMEM_GPUINITIATED_SUPPORT
        status = nvshmemi_gather_mem_handles(local_handles, state, 0, state->heap_size);
        #else
        status = nvshmemi_gather_mem_handles(local_handles, state);
        #endif
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of mem handles failed \n");
    }

    state->peer_heap_base_actual = (void **)calloc(state->npes, sizeof(void *));
    NULL_ERROR_JMP(state->peer_heap_base_actual, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for peer heap base \n");

    status = nvshmemi_boot_handle.allgather((void *)&state->heap_base,
                                          (void *)state->peer_heap_base_actual, sizeof(void *),
                                          &nvshmemi_boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                 "allgather of heap base ptrsmem handle failed \n");

    state->peer_heap_base = (void **)calloc(state->npes, sizeof(void *));
    NULL_ERROR_JMP(state->peer_heap_base, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for peer heap base \n");
    
    // pick a transport that allows LD/ST and map to get local pointers to remote heaps
    // assumes symmetry of handles received from each peer, should be true (mostly)
    p2p_counter = 1;
    for (int i = 0; i < state->npes; i++) {
        if (i == state->mype) {
            state->peer_heap_base[i] = state->heap_base;
            continue;
        }

        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            if (state->transport_map[state->mype * state->npes + i] & (1 << j)) {
                if (transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP) {
                    if (nvshmemi_use_cuda_vmm) {
                        state->peer_heap_base[i] = (void *)((uintptr_t)state->global_heap_base 
                                                            + state->heap_size * p2p_counter++);
                        break;
                    } else {
                        status = transports[j]->host_ops.map(
                            (state->peer_heap_base + i), state->heap_size,
                            &state->handles.back()[i * NVSHMEM_TRANSPORT_COUNT + j]);
                        if (status) {
                               //map operation failed, remove cap of transport
                               state->transports[j]->cap[i] ^= NVSHMEM_TRANSPORT_CAP_MAP;
                               status = 0; 
                               continue;
                        }

                        char *hex = nvshmemu_hexdump(
                            &state->handles.back()[i * NVSHMEM_TRANSPORT_COUNT + j],
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
    }

    /* Build p2p_processes that is used during dynaminic heap management */
    if (nvshmemi_use_cuda_vmm) {
        pid_t pid = getpid();
        pid_t *peer_pids = (pid_t *) malloc(sizeof(pid_t) * state->npes);
        status = nvshmemi_boot_handle.allgather((void *)&pid, (void *)peer_pids, sizeof(pid_t), &nvshmemi_boot_handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of pids failed \n");
            
        uint64_t myHostHash = getHostHash();
        uint64_t *hostHash = (uint64_t *)malloc(sizeof(uint64_t) * state->npes);
        status = nvshmemi_boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                              &nvshmemi_boot_handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of host hashes failed \n");

        for(int pe = 0; pe < state->npes; pe++) {
            if (hostHash[pe] == myHostHash) {
                p2p_processes[peer_pids[pe]] = pe;
            }
        }
        INFO(NVSHMEM_MEM, "I am connected to %lu p2p processes (including myself)", p2p_processes.size());
        free(peer_pids);
        free(hostHash);
    }

out:
    if (status) {
        // if handles has been allocated, try and cleanup all heap state
        // else cleanup local handles only
        nvshmemi_cleanup_symmetric_heap(state);
        if (state->heap_base) cuMemFree((CUdeviceptr)state->heap_base);
    }
    return status;
}

#if CUDART_VERSION >= 11030
int nvshmemi_add_physical_memory(size_t size) {
    int status;
    nvshmemi_state_t *state = nvshmemi_state;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_mem_handle_t local_handles[NVSHMEM_TRANSPORT_COUNT];
    int i = 0;
    pid_t pid;
    ipcHandle *myIpcHandle = NULL;

    CUmemGenericAllocationHandle cumem_handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<int>(state->device_id);
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    
    size = ((size + cumem_granularity - 1) / cumem_granularity) * cumem_granularity;
    INFO(NVSHMEM_MEM, "Adding new physical backing of size %zu bytes", size);
    status =
        cuMemCreate(&cumem_handle, size, (const CUmemAllocationProp *)&prop, 0);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemCreate failed \n");
    state->cumem_handles.push_back(cumem_handle);

    status = cuMemMap((CUdeviceptr)((char *)state->heap_base + state->physical_heap_size), size, 0, cumem_handle, 0);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemMap failed \n");

    CUmemAccessDesc access;
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = state->device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status = cuMemSetAccess((CUdeviceptr)state->heap_base + state->physical_heap_size, size,
                            (const CUmemAccessDesc *)&access, 1);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemSetAccess failed \n");


    memset(local_handles, 0, sizeof(nvshmem_mem_handle_t) * NVSHMEM_TRANSPORT_COUNT);

    /* Get mem handles */
    for (i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if ((state->transport_bitmap & (1 << i)) && transports[i]->host_ops.get_mem_handle) {
            INFO(NVSHMEM_INIT, "calling get_mem_handle for transport: %d buf: %p size: %d", i,
                 (char *)state->heap_base + state->physical_heap_size, size);
            status = transports[i]->host_ops.get_mem_handle(
                (nvshmem_mem_handle_t *)(local_handles + i),
                (nvshmem_mem_handle_t *)&cumem_handle /*used only by p2p transport*/,
                (char *)state->heap_base + state->physical_heap_size, size, transports[i], false);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport get memhandle failed \n");
            INFO(NVSHMEM_INIT, "[%d] get_mem_handle transport %d handles %p", state->mype, i,
                 local_handles + i);
        }
    }

    state->handles.push_back(vector<nvshmem_mem_handle_t>(NVSHMEM_TRANSPORT_COUNT * state->npes));
    #ifdef NVSHMEM_GPUINITIATED_SUPPORT
    status = nvshmemi_gather_mem_handles(local_handles, state, state->physical_heap_size, size);
    #else
    status = nvshmemi_gather_mem_handles(local_handles, state);
    #endif
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of mem handles failed \n");

    /* Now setup symmetric heap */
    pid = getpid();
    IPC_CHECK(ipcOpenSocket(myIpcHandle));
    
    /* Wait for all processes to open their sockets */
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

    for (std::map<pid_t, int>::iterator it1 = p2p_processes.begin();
					it1 != p2p_processes.end(); ++it1) {
        pid_t sending_process = it1->first;
        if (pid == sending_process) {
            for (std::map<pid_t, int>::iterator it2 = p2p_processes.begin();
						it2 != p2p_processes.end(); ++it2) {
                pid_t receiving_process = it2->first;
                if (pid != receiving_process) {/* Dont sent to yourself */
                    IPC_CHECK(ipcSendFd(myIpcHandle, *(int *)local_handles, receiving_process));
                }
            }
        } else  {
            IPC_CHECK(ipcRecvFd(
                myIpcHandle, (int *)&state->handles.back()[it1->second * NVSHMEM_TRANSPORT_COUNT]));
        }
        /* Putting a global barrier means assuming that all nodes are running with config */
        status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    }
    IPC_CHECK(ipcCloseSocket(myIpcHandle));

    // pick a transport that allows LD/ST and map to get local pointers to remote heaps
    // assumes symmetry of handles received from each peer, should be true (mostly)
    for (int i = 0; i < state->npes; i++) {
        if (i == state->mype) continue;

        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            if (state->transport_map[state->mype * state->npes + i] & (1 << j)) {
                if (transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP) {
                    char *map_addr = (char *)state->peer_heap_base[i] + state->physical_heap_size;
                    status = transports[j]->host_ops.map(
                        (void **)&map_addr, size,
                        &state->handles.back()[i * NVSHMEM_TRANSPORT_COUNT + j]);
                    if (status) {
                           //map operation failed, remove cap of transport
                           state->transports[j]->cap[i] ^= NVSHMEM_TRANSPORT_CAP_MAP;
                           status = 0; 
                           continue;
                    }

                    char *hex =
                        nvshmemu_hexdump(&state->handles.back()[i * NVSHMEM_TRANSPORT_COUNT + j],
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
    mspace_add_new_chunk(state->heap_mspace, (char *)state->heap_base + state->physical_heap_size, size);
    for (size_t i = 0; i < (size / cumem_granularity); i++)
        nvshmemi_state->idx_in_handles.push_back(make_tuple(nvshmemi_state->handles.size() - 1,
                                                            (char *)state->heap_base + state->physical_heap_size,
                                                            size));
    state->physical_heap_size += size;
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle); /* Wait for all PEs to setup the new memory */
out:
    if (status) {
        nvshmemi_cleanup_symmetric_heap(state);
        if (state->heap_base) cuMemFree((CUdeviceptr)state->heap_base);
    }
    return status;
}
#endif

extern "C" {
void *nvshmemi_malloc(size_t size) {
    int status = 0;
    void *ptr = NULL;

    status = check_for_symmetry(size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "symmetry check for size failed\n");

    ptr = mspace_malloc(nvshmemi_state->heap_mspace, size);
#if CUDART_VERSION >= 11030
    if (nvshmemi_use_cuda_vmm) {
        if ((size > 0) && (ptr == NULL)) {
            nvshmemi_add_physical_memory(size);
            ptr = mspace_malloc(nvshmemi_state->heap_mspace, size);
        }
        return ptr;
    } else
#endif
    {
        if ((size > 0) && (ptr == NULL)) {
            ERROR_EXIT("nvshmem malloc failed (hint: check if total allocation has exceeded NVSHMEM "
                       "symmetric size = %zu, NVSHMEM symmetric size can be increased using "
                       "NVSHMEM_SYMMETRIC_SIZE environment variable) \n", nvshmemi_options.SYMMETRIC_SIZE);
        }
    }

    INFO(NVSHMEM_INIT, "[%d] allocated %zu bytes from mspace: %p ptr: %p", nvshmemi_state->mype,
         size, nvshmemi_state->heap_mspace, ptr);

out:
    return ptr;
}
}

void *nvshmemi_calloc(size_t count, size_t size) {
    int status = 0;
    void *ptr = NULL;

    status = check_for_symmetry(size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "symmetry check for size failed\n");

    ptr = mspace_calloc(nvshmemi_state->heap_mspace, count, size);
#if CUDART_VERSION >= 11030
    if (nvshmemi_use_cuda_vmm) {
        if ((size > 0) && (ptr == NULL)) {
            nvshmemi_add_physical_memory(size);
            ptr = mspace_calloc(nvshmemi_state->heap_mspace, count, size);
        }
        return ptr;
    } else
#endif
    {
        if (size > 0 && count > 0 && ptr == NULL) {
            ERROR_EXIT("nvshmem calloc failed \n");
        }
    }

    INFO(NVSHMEM_INIT, "[%d] calloc allocated %zu bytes from mspace: %p ptr: %p",
            nvshmemi_state->mype, size, nvshmemi_state->heap_mspace, ptr);
out:
    return ptr;
}

void *nvshmem_malloc(size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();
	(*nvshmemi_check_state_and_init_fn_ptr)();

    ptr = nvshmemi_malloc(size);

    nvshmemi_barrier_all();

    NVSHMEMU_THREAD_CS_EXIT();

    return ptr;
}

void *nvshmem_calloc(size_t count, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();
	(*nvshmemi_check_state_and_init_fn_ptr)();

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
#if CUDART_VERSION >= 11030
    if (nvshmemi_use_cuda_vmm) {
        if ((size > 0) && (ptr == NULL)) {
            nvshmemi_add_physical_memory(size + alignment);
            ptr = mspace_memalign(nvshmemi_state->heap_mspace, alignment, size);
        }
        return ptr;
    } else
#endif
    {
        if ((size > 0) && (ptr == NULL)) {
            ERROR_EXIT("nvshmem align failed \n");
        }
    }

out:
    return ptr;
}

void *nvshmem_align(size_t alignment, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();
	(*nvshmemi_check_state_and_init_fn_ptr)();

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

    NVSHMEMI_CHECK_INIT_STATUS();

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

static struct nvshmem_transport *nvshmemi_get_remote_transport() {
    struct nvshmem_transport *t = NULL;

    /* Built in assumption that we only allow for one remote transport. TODO: codify that. */
    if (nvshmemi_state->transport_bitmap & (1 << NVSHMEM_TRANSPORT_ID_IBRC)) {
        t = nvshmemi_state->transports[NVSHMEM_TRANSPORT_ID_IBRC];
    } else if (nvshmemi_state->transport_bitmap & (1 << NVSHMEM_TRANSPORT_ID_UCX)) {
        t = nvshmemi_state->transports[NVSHMEM_TRANSPORT_ID_UCX];
    } else if (nvshmemi_state->transport_bitmap & (1 << NVSHMEM_TRANSPORT_ID_IBDEVX)) {
        t = nvshmemi_state->transports[NVSHMEM_TRANSPORT_ID_IBDEVX];
    } else if (nvshmemi_state->transport_bitmap & (1 << NVSHMEM_TRANSPORT_ID_FABRIC)) {
        t = nvshmemi_state->transports[NVSHMEM_TRANSPORT_ID_FABRIC];
    } 
    #if NVSHMEM_GPUINITIATED_SUPPORT
    else if (nvshmemi_state->transport_bitmap & (1 << NVSHMEM_TRANSPORT_ID_GIC)) {
        t = nvshmemi_state->transports[NVSHMEM_TRANSPORT_ID_GIC];
    }
    #endif

    return t;
}

int nvshmemx_buffer_register(void *addr, size_t length) {
    struct nvshmem_transport *t = nvshmemi_get_remote_transport();
    nvshmem_local_buf_handle_t *handle;
    size_t i;
    void *heap_end;
    int status = 0;
    int lock_status = EBUSY;
    cudaPointerAttributes attr;
#if CUDART_VERSION < 11000
    bool register_with_cuda = false;
#endif

    status = cudaPointerGetAttributes(&attr, addr);
#if CUDART_VERSION >= 11000
    if (status != cudaSuccess) {
        ERROR_PRINT("Unable to query pointer attributes.\n");
        /* clear CUDA error string. */
        cudaGetLastError();
        return NVSHMEMX_ERROR_INTERNAL;
    }
#else
    if (status != cudaSuccess) {
        /* clear CUDA error string. */
        cudaGetLastError();
        if (status == cudaErrorInvalidValue) {
            register_with_cuda = true;
        } else {
            ERROR_PRINT("Unable to query pointer attributes.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        }
    }
#endif

    if (attr.type == cudaMemoryTypeManaged) {
        ERROR_PRINT("Unable to register managed memory as it can migrate.\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    heap_end = (void *) ((char *)nvshmemi_state->heap_base + nvshmemi_state->heap_size);
    if (addr >= nvshmemi_state->heap_base && addr < heap_end) {
        ERROR_PRINT("Unable to register nvshmem heap memory. It is registered by default.\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    handle = (nvshmem_local_buf_handle_t *)calloc(1, sizeof(nvshmem_local_buf_handle_t));
    if (handle == NULL) {
        ERROR_PRINT("Unable to resize the registered buffer array.\n");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    if (t) {
        handle->handle = (nvshmem_mem_handle_t *)calloc(1, sizeof(nvshmem_mem_handle_t));
        if (handle->handle == NULL) {
            ERROR_PRINT("Unable to resize the registered buffer array.\n");
            status = NVSHMEMX_ERROR_OUT_OF_MEMORY;
            goto out_error_unlocked;
        }
    }

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_wrlock(&nvshmemi_state->registered_buffer_lock);
    }

    if (lock_status != 0) {
        ERROR_PRINT("Unable to acquire buffer registration lock with errno %d\n", lock_status);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out_error_unlocked;
    }

#if CUDART_VERSION >= 11000
    if (attr.type == cudaMemoryTypeUnregistered) {
#else
    if (register_with_cuda) {
#endif
        if (!nvshmemi_state->host_memory_registration_supported) {
            ERROR_PRINT("Unable to register host memory for this device as it doesn't support UVA.\n");
            status = NVSHMEMX_ERROR_INVALID_VALUE;
            goto out_unlock;
        }
        status = cudaHostRegister(addr, length, cudaHostRegisterDefault);
        if (status) {
            ERROR_PRINT("Unable to register host memory with CUDA.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out_unlock;
        }
        handle->registered_by_us = true;
    }

    /* We only need to register unregistered host buffers if there is no remote transport.
     * CUDA memory and registered host memory are already mapped into the address space
     * so nothing to register.
     */
    if (t == NULL && handle->registered_by_us == false) {
        status = 0;
        free(handle);
        goto out_unlock;
    }

    if (nvshmemi_state->registered_buffer_array_used == nvshmemi_state->registered_buffer_array_size) {
        size_t new_array_size = nvshmemi_state->registered_buffer_array_size * 2;
        void *new_buf;

        assert(new_array_size < (SIZE_MAX / sizeof(nvshmem_local_buf_handle_t)));
        new_buf = realloc(nvshmemi_state->registered_buffers, new_array_size * sizeof(nvshmem_local_buf_handle_t *));
        if (new_buf == NULL) {
            ERROR_PRINT("Unable to resize the registered buffer array.\n");
            status = NVSHMEMX_ERROR_OUT_OF_MEMORY;
            goto out_unlock;
        }
        nvshmemi_state->registered_buffers = (nvshmem_local_buf_handle_t **)new_buf;
        nvshmemi_state->registered_buffer_array_size = new_array_size;
    }

    /* TODO: This could be a binary search. */
    for (i = 0; i < nvshmemi_state->registered_buffer_array_used; i++) {
        nvshmem_local_buf_handle_t *tmp_handle = nvshmemi_state->registered_buffers[i];
        if (addr > tmp_handle->ptr) {
            continue;
        } else if (addr == tmp_handle->ptr) {
            if (length != tmp_handle->length) {
                ERROR_PRINT("Unable to register overlapping memory regions.\n");
                status = NVSHMEMX_ERROR_INVALID_VALUE;
                goto out_unlock;
            }
            free(handle);
            goto out_unlock;
        /* addr < tmp_handle->ptr */
        } else {
            break;
        }
    }

    if (t) {
        status = t->host_ops.get_mem_handle(handle->handle, NULL, addr, length, t, true);
        if (status) {
            ERROR_PRINT("Unable to assign new memory handle.\n");
            goto out_unlock;
        }
    }
    handle->ptr = addr;
    handle->length = length;

    assert(i < nvshmemi_state->registered_buffer_array_size);
    if (i < nvshmemi_state->registered_buffer_array_used) {
        memmove(&nvshmemi_state->registered_buffers[i + 1],
                &nvshmemi_state->registered_buffers[i],
                sizeof(nvshmem_local_buf_handle_t *) * (nvshmemi_state->registered_buffer_array_used - i));
    }
    nvshmemi_state->registered_buffers[i] = handle;
    nvshmemi_state->registered_buffer_array_used++;

out_unlock:
    pthread_rwlock_unlock(&nvshmemi_state->registered_buffer_lock);
    if (status == 0) {
        return 0;
    }

out_error_unlocked:
    if (handle->registered_by_us) {
        cudaHostUnregister(addr);
    }
    if (handle) {
        if (handle->handle) {
            free(handle->handle);
        }
        free(handle);
    }
    return status;
}

int nvshmemx_buffer_unregister(void *addr) {
    struct nvshmem_transport *t = nvshmemi_get_remote_transport();
    size_t i;
    int lock_status = EBUSY;
    int status = 0;

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_wrlock(&nvshmemi_state->registered_buffer_lock);
    }

    if (lock_status != 0) {
        ERROR_PRINT("Unable to acquire buffer registration lock with errno %d\n", lock_status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* TODO: This could be a binary search. */
    for (i = 0; i < nvshmemi_state->registered_buffer_array_used; i++) {
        nvshmem_local_buf_handle_t *tmp_handle = nvshmemi_state->registered_buffers[i];
        if (addr > tmp_handle->ptr) {
            continue;
        } else if (addr == tmp_handle->ptr) {
            if ((i + 1) < nvshmemi_state->registered_buffer_array_used) {
                memmove(&nvshmemi_state->registered_buffers[i],
                        &nvshmemi_state->registered_buffers[i + 1],
                        sizeof(nvshmem_local_buf_handle_t *) * (nvshmemi_state->registered_buffer_array_used - i));
            }
                if (t) {
                    t->host_ops.release_mem_handle(tmp_handle->handle, t);
                    free(tmp_handle->handle);
                }

                if (tmp_handle->registered_by_us) {
                    cudaHostUnregister(tmp_handle->ptr);
                }
                free(tmp_handle);
                nvshmemi_state->registered_buffer_array_used--;
                goto out_unlock;
        /* addr < tmp_handle->ptr*/
        } else {
            break;
        }
    }

    status = NVSHMEMX_ERROR_INVALID_VALUE;
out_unlock:
    pthread_rwlock_unlock(&nvshmemi_state->registered_buffer_lock);
    return status;
}

void nvshmemx_buffer_unregister_all() {
    struct nvshmem_transport *t = nvshmemi_get_remote_transport();
    int lock_status = EBUSY;

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_wrlock(&nvshmemi_state->registered_buffer_lock);
    }

    if (lock_status != 0) {
        ERROR_PRINT("Unable to acquire buffer registration lock with errno %d. Unregister all function failed.\n", lock_status);
        return;
    }

    for (size_t i = 0; i < nvshmemi_state->registered_buffer_array_used; i++) {
        if (t) {
            t->host_ops.release_mem_handle(nvshmemi_state->registered_buffers[i]->handle, t);
            free(nvshmemi_state->registered_buffers[i]->handle);
        }
        if (nvshmemi_state->registered_buffers[i]->registered_by_us) {
            cudaHostUnregister(nvshmemi_state->registered_buffers[i]->ptr);
        }
        free(nvshmemi_state->registered_buffers[i]);
    }

    nvshmemi_state->registered_buffer_array_used = 0;
    pthread_rwlock_unlock(&nvshmemi_state->registered_buffer_lock);

    return;
}

struct nvshmem_mem_handle *nvshmemi_get_registered_buffer_handle(void *addr, size_t *len) {
    nvshmem_local_buf_handle_t *tmp_handle;
    size_t min, max, mid;
    void *max_addr;
    size_t max_len;
    int lock_status = EBUSY;
    struct nvshmem_mem_handle *ret_handle = NULL;

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_rdlock(&nvshmemi_state->registered_buffer_lock);
    }

    if (lock_status != 0) {
        ERROR_PRINT("Unable to acquire buffer registration lock with errno %d.\n", lock_status);
        return ret_handle;
    }

    if (nvshmemi_state->registered_buffer_array_used == 0) {
        goto out_unlock;
    }

    min = 0;
    max = nvshmemi_state->registered_buffer_array_used;
    do {
            mid = (max - min) / 2 + min;
            /* We have gone past the end of the loop. */
            if (mid >= nvshmemi_state->registered_buffer_array_used) {
                break;
            }
            tmp_handle = nvshmemi_state->registered_buffers[mid];
            if (addr > tmp_handle->ptr) {
                max_addr = (void *)((char *)tmp_handle->ptr + tmp_handle->length);
                max_len = (uint64_t)((char *)max_addr - (char *)addr);
                if (addr < max_addr) {
                    *len = *len < max_len ? *len : max_len;
                    ret_handle = tmp_handle->handle;
                    goto out_unlock;
                }
                min = mid + 1;
            } else if (addr == tmp_handle->ptr) {
                *len = *len < tmp_handle->length ? *len : tmp_handle->length;
                ret_handle = tmp_handle->handle;
                goto out_unlock;
            } else {
                if (mid == 0) {
                    break;
                }
                max = mid - 1;
            }
    } while (max >= min);

    ERROR_PRINT("Unable to find a reference to the requested buffer address.\n");

out_unlock:
    pthread_rwlock_unlock(&nvshmemi_state->registered_buffer_lock);
    return ret_handle;
}
