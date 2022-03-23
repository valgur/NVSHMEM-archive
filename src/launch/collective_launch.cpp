/****
 * Copyright (c) 2016-2018, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"

#include <iostream>
#include <cassert>
#include "util.h"

int nvshmemi_setup_collective_launch(nvshmemi_state_t *state) {
    int leastPriority, greatestPriority, status = 0;
    status = cuDeviceGetAttribute(&(state->cu_dev_attrib.multi_processor_count),
                                  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceGetAttribute of CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT failed \n");
    state->cu_dev_attrib.cooperative_launch = 0;

    status = cuDeviceGetAttribute(&(state->cu_dev_attrib.cooperative_launch),
                                  CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceGetAttribute CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH failed \n");

    if (!state->cu_dev_attrib.cooperative_launch)
        WARN_PRINT(
            "Cooperative launch not supported on PE %d; GPU-side synchronize may cause hang\n",
            state->mype);

    status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuCtxGetStreamPriorityRange failed \n");

    status = cuStreamCreateWithPriority(&state->claunch_params.stream, CU_STREAM_NON_BLOCKING,
                                        greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuStreamCreateWithPriority failed \n");

    status = cuEventCreate(&state->claunch_params.begin_event, CU_EVENT_DISABLE_TIMING);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventCreate for begin event failed \n");

    status = cuEventCreate(&state->claunch_params.end_event, CU_EVENT_DISABLE_TIMING);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventCreate for end event failed \n");

out:
    return status;
}

int nvshmemi_teardown_collective_launch(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_teardown_collective_launch");
    int status = 0;

    if (!nvshmemi_is_nvshmem_initialized) goto out;

    status = cuStreamDestroy(state->claunch_params.stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuStreamDestroy failed \n");

    status = cuEventDestroy(state->claunch_params.begin_event);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventDestroy for begin event failed \n");

    status = cuEventDestroy(state->claunch_params.end_event);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventDestroy for end event failed \n");

out:
    return status;
}
