/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"

#include "util.h"
#include "nvshmemx_error.h"
#include "bootstrap_internal.h"

int bootstrap_init(int mode, bootstrap_attr_t *attr, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;

    switch (mode) {
        case BOOTSTRAP_MPI:
#ifdef NVSHMEM_MPI_SUPPORT
            const char *plugin_name;
            if (nvshmemi_options.BOOTSTRAP_PLUGIN_provided)
                plugin_name = nvshmemi_options.BOOTSTRAP_PLUGIN;
            else
                plugin_name = BOOTSTRAP_MPI_PLUGIN;

            status = bootstrap_loader_init(plugin_name, (attr != NULL) ? attr->mpi_comm : NULL, handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_loader_init returned error\n");
#else
            status = 1;
            ERROR_PRINT("MPI bootstrap requested but NVSHMEM was not built with MPI support\n");
            goto out;
#endif
            break;
        case BOOTSTRAP_SHMEM:
#ifdef NVSHMEM_SHMEM_SUPPORT
#ifndef NVSHMEM_MPI_SUPPORT
            const char *plugin_name;
#endif
            if (nvshmemi_options.BOOTSTRAP_PLUGIN_provided)
                plugin_name = nvshmemi_options.BOOTSTRAP_PLUGIN;
            else
                plugin_name = BOOTSTRAP_SHMEM_PLUGIN;

            status = bootstrap_loader_init(plugin_name, (attr != NULL) ? &attr->initialize_shmem : NULL, handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_loader_init returned error\n");
#else
            status = 1;
            ERROR_PRINT("OpenSHMEM bootstrap requested but NVSHMEM was not built with OpenSHMEM support\n");
            goto out;
#endif
            break;
        case BOOTSTRAP_PMI:
            if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMIX") == 0) {
#ifdef NVSHMEM_PMIX_SUPPORT
                status = bootstrap_loader_init(BOOTSTRAP_PMIX_PLUGIN, NULL, handle);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_loader_init returned error\n");
#else
                ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "PMIx bootstrap requested but NVSHMEM was not built with PMIx support\n");
#endif
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI-2") == 0 ||
                       strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI2") == 0) {
                status = bootstrap_pmi2_init(handle);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmi_init returned error\n");
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI") == 0) {
                status = bootstrap_pmi_init(handle);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmi_init returned error\n");
            } else {
                ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmi_init invalid bootstrap '%s'\n",
                        nvshmemi_options.BOOTSTRAP_PMI);
            }
            break;
        case BOOTSTRAP_PLUGIN:
            assert(attr == NULL);

            if (!nvshmemi_options.BOOTSTRAP_PLUGIN_provided) {
                ERROR_PRINT("Plugin bootstrap requires NVSHMEM_BOOTSTRAP_PLUGIN to be set\n");
                status = 1;
                goto out;
            }

            status = bootstrap_loader_init(nvshmemi_options.BOOTSTRAP_PLUGIN, NULL, handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_loader_init returned error\n");
            break;
        default:
            ERROR_PRINT("Invalid bootstrap mode selected\n");
    }

out:
    return status;
}

int bootstrap_finalize(bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;

    status = handle->finalize(handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
            "bootstrap finalization returned error\n");

out:
    return status;
}
