/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "util.h"
#include "gpu_coll.h"
#include "nvshmem_internal.h"

extern "C" int init_shm_kernel_shm_ptr() {
    int status = 0;
    nvshmemi_device_state.gpu_coll_env_params_var = gpu_coll_env_params_var;
    return status;
}
