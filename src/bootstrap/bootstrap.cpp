/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"

#include "util.h"
#include "nvshmemx_error.h"
#include "bootstrap.h"
#include "bootstrap_internal.h"

int bootstrap_preinit(bootstrap_preinit_handle_t *handle) {
    ERROR_PRINT("not implemented");
    return NVSHMEMX_ERROR_INTERNAL;
}

int bootstrap_init(int mode, bootstrap_attr_t *attr, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;

#ifdef NVSHMEM_MPI_SUPPORT
    if (mode == BOOTSTRAP_MPI) {
        status = bootstrap_mpi_init(attr->mpi_comm, handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_mpi_init returned error \n");
    } else
#endif
#ifdef NVSHMEM_SHMEM_SUPPORT
    if (mode == BOOTSTRAP_SHMEM) {
        status = bootstrap_shmem_init(handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                     "bootstrap_shmem_init returned error \n");
    } else
#endif
    if (mode == BOOTSTRAP_STATIC) {
        status = bootstrap_pmi_init(handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmi_init returned error \n");
    } else if (mode == BOOTSTRAP_DYNAMIC) {
        ERROR_PRINT("not implemented");
        status = NVSHMEMX_ERROR_INVALID_VALUE;
    } else {
	if(mode == BOOTSTRAP_MPI) 
      	    ERROR_PRINT("MPI-based initialization selected but NVSHMEM not built with MPI Support \n");
	else if (mode == BOOTSTRAP_SHMEM)
      	    ERROR_PRINT("OpenSHMEM-based bootstrap selected but NVSHMEM not built with OpenSHMEM Support \n");
	else
      	    ERROR_PRINT("Invalid bootstrap mode selected \n");
	status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    handle->mode = mode;

out:
    return status;
}

int bootstrap_finalize(bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;
    int mode = handle->mode;

#ifdef NVSHMEM_MPI_SUPPORT
    if (mode == BOOTSTRAP_MPI) {
        status = bootstrap_mpi_finalize(handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                     "bootstrap_mpi_finalize returned error \n");
    } else
#endif
#ifdef NVSHMEM_SHMEM_SUPPORT
        if (mode == BOOTSTRAP_SHMEM) {
        status = bootstrap_shmem_finalize(handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                     "bootstrap_shmem_finalize returned error \n");
    } else
#endif
        if (mode == BOOTSTRAP_STATIC) {
        status = bootstrap_pmi_finalize(handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                     "bootstrap_pmi_finalize returned error \n");
    } else {
        ERROR_PRINT("invalid initialization mode \n");
        status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

out:
    return status;
}
