/*
 * * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#ifndef _INTERNAL_H
#define _INTERNAL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "bootstrap.h"
#include "transport.h"
#include "util.h"
#include "common.h"

#define MAX_PEER_STREAMS 3

#define MAX_LENGTH_PREFIX_STRING 1024

#define MAX_TRANSPORT_EP_COUNT 1

#define SYMMETRIC_SIZE_DEFAULT 1024 * 1024 * 1024

#define MAXPATHSIZE 1024

#define NUM_G_BUF_ELEMENTS 1024 * 1024

// constansts that grow with npes
#define CPU_SYNC_SIZE sizeof(int)
#define CPU_GPU_SYNC_SIZE sizeof(int)
#define CPU_DATA_SIZE sizeof(int)
#define GPU_IPSYNC_SIZE sizeof(long)
#define COLL_NPES_FACTOR (CPU_SYNC_SIZE + CPU_GPU_SYNC_SIZE + CPU_DATA_SIZE + GPU_IPSYNC_SIZE)

// independent constansts
#define GPU_SCRATCH_SIZE 16384
#define GPU_RDXN_SCRATCH_SIZE 16384
#define GPU_IPWRK_SIZE sizeof(int4) * 2 * SYNC_SIZE
#define GPU_ICOUNTER_SIZE sizeof(long) * SYNC_SIZE
#define GPU_ICOUNTER_BARRIER_SIZE sizeof(long) * SYNC_SIZE

#define COLL_CONSTANT_FACTOR                                                         \
    (GPU_SCRATCH_SIZE + GPU_RDXN_SCRATCH_SIZE + GPU_IPWRK_SIZE + GPU_ICOUNTER_SIZE + \
     GPU_ICOUNTER_BARRIER_SIZE)

#define NVSHMEM_CHECK_STATE_AND_INIT()                                               \
    do {                                                                             \
        if (!nvshmem_state) ERROR_EXIT("nvshmem API called before nvshmem_init \n"); \
        if (!nvshmem_state->initialized) {                                           \
            if (nvshmemi_common_init(nvshmem_state)) {                               \
                ERROR_EXIT("nvshmem initialization failed, exiting \n");             \
            }                                                                        \
        }                                                                            \
    } while (0)

typedef struct {
    int multi_processor_count;
    int cooperative_launch;
} cuda_device_attributes_t;

typedef struct {
    CUstream stream;
    CUevent begin_event;
    CUevent end_event;
} collective_launch_params_t;

typedef struct nvshmem_state_dec {
    /*PE state*/
    int mype;
    int npes;
    int mype_node;
    int npes_node;
    /*environment info*/
    char *prefix;
    /*device state*/
    CUdevice cudevice;
    CUcontext cucontext;
    /*symmetric heap state*/
    size_t heap_size;
    int initialized;
    void *heap_base;
    void **peer_heap_base_actual;
    void **peer_heap_base;
    void *heap_mspace;
    struct nvshmem_mem_handle *handles;
    bootstrap_handle_t boot_handle;
    /*transport info*/
    int transport_count;
    int transport_bitmap;
    int *transport_map;
    struct nvshmem_transport_pe_info *pe_info;
    struct nvshmem_transport **transports;
    int *selected_transport_for_rma;
    int *selected_transport_for_amo;
    /*consolidated rma ops*/
    rma_handle *rma;
    amo_handle *amo;
    fence_handle *fence;
    quiet_handle *quiet;
    /*scratch space*/
    char *scratch_space;
    size_t scratch_size;
    int *scratch;
    cuda_device_attributes_t cu_dev_attrib;
    int *p2p_attrib_native_atomic_support;
    collective_launch_params_t claunch_params;
    CUstream my_stream;
    // proxy
    void *proxy;
    CUstream *custreams;
    CUevent *cuevents;
    CUdeviceptr *curets;
} nvshmem_state_t;

typedef struct {
    volatile uint64_t data;
    volatile uint64_t flag;
} g_elem_t;

typedef struct {
    int error_checks;
} nvshmem_options_t;

extern nvshmem_state_t *nvshmem_state;
extern nvshmem_options_t nvshmem_options;
extern __device__ unsigned long long test_wait_any_start_idx_d;
extern int nvshmemi_job_connectivity;

int nvshmemi_common_init(nvshmem_state_t *state);
int nvshmemi_init_g_buffer();
int nvshmemi_init_device_state(nvshmem_state_t *state);
int nvshmemi_setup_local_heap(nvshmem_state_t *state);
int nvshmemi_setup_symmetric_heap(nvshmem_state_t *state);
int nvshmemi_setup_connections(nvshmem_state_t *state);
int nvshmemi_cleanup_symmetric_heap(nvshmem_state_t *state);
int nvshmemi_setup_collective_launch(nvshmem_state_t *state);
int nvshmemi_teardown_collective_launch(nvshmem_state_t *state);
int nvshmemi_setup_mops_kernels(nvshmem_state_t *state);
void *nvshmemi_malloc(size_t size);
void *nvshmemi_calloc(size_t count, size_t size);
void *nvshmemi_align(size_t alignment, size_t size);
void nvshmemi_free(void *ptr);

void nvshmemi_barrier_all();

int nvshmemi_proxy_init(nvshmem_state_t *state);
int nvshmemi_proxy_finalize(nvshmem_state_t *state);

#endif
