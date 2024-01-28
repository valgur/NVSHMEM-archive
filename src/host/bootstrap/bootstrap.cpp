/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>
#include <stdio.h>   // IWYU pragma: keep
#include <stdint.h>  // IWYU pragma: keep

#include "internal/common/debug.h"
#include "modules/transport/env_defs_internal.h"
#include "internal/common/nvshmem_internal.h"
#include "modules/common/nvshmemi_bootstrap_defines.h"
#include "internal/host/nvshmemi_bootstrap_library.h"
#include "host/nvshmemx_error.h"
#include "internal/util.h"
#include <unordered_map>

static std::unordered_map<int, string> bootstrap_modes = {{BOOTSTRAP_MPI, "MPI"},
                                                          {BOOTSTRAP_SHMEM, "SHMEM"},
                                                          {BOOTSTRAP_PMI, "PMI"},
                                                          {BOOTSTRAP_PLUGIN, "PLUGIN"},
                                                          {BOOTSTRAP_UID, "UID"}};

int bootstrap_preinit(int mode, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;
    bootstrap_env_attr_t attr = {};
    const char *plugin_name = NULL;
    switch (mode) {
        case BOOTSTRAP_MPI:
        case BOOTSTRAP_SHMEM:
        case BOOTSTRAP_PMI:
        case BOOTSTRAP_PLUGIN:
            /* NOOP for other modalities */
            return (status);
        case BOOTSTRAP_UID:
            plugin_name = nvshmemi_options.BOOTSTRAP_UID_PLUGIN;
            attr.uid_session_id =
                strlen(nvshmemi_options.BOOTSTRAP_UID_SESSION_ID) > 0
                    ? const_cast<char *>(nvshmemi_options.BOOTSTRAP_UID_SESSION_ID)
                    : nullptr;
            attr.uid_socket_ifname =
                strlen(nvshmemi_options.BOOTSTRAP_UID_SOCK_IFNAME) > 0
                    ? const_cast<char *>(nvshmemi_options.BOOTSTRAP_UID_SOCK_IFNAME)
                    : nullptr;
            attr.uid_socket_family =
                strlen(nvshmemi_options.BOOTSTRAP_UID_SOCK_FAMILY) > 0
                    ? const_cast<char *>(nvshmemi_options.BOOTSTRAP_UID_SOCK_FAMILY)
                    : nullptr;
            status = bootstrap_loader_preinit(plugin_name, (void *)(&attr), handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_preinit returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        default:
            NVSHMEMI_ERROR_PRINT("Invalid bootstrap mode selected\n");
    }

out:
    return status;
}

int bootstrap_init(int mode, bootstrap_attr_t *attr, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;
    const char *plugin_name = NULL;

    switch (mode) {
        case BOOTSTRAP_MPI:
            plugin_name = nvshmemi_options.BOOTSTRAP_MPI_PLUGIN;

            status =
                bootstrap_loader_init(plugin_name, (attr != NULL) ? attr->mpi_comm : NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        case BOOTSTRAP_SHMEM:
            plugin_name = nvshmemi_options.BOOTSTRAP_SHMEM_PLUGIN;

            status = bootstrap_loader_init(plugin_name,
                                           (attr != NULL) ? &attr->initialize_shmem : NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        case BOOTSTRAP_PMI:
            if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMIX") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMIX_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_loader_init returned error for mode %s\n",
                                      bootstrap_modes[mode].c_str());
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI-2") == 0 ||
                       strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI2") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMI2_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_pmi_init returned error for mode %s\n",
                                      bootstrap_modes[mode].c_str());
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMI_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_pmi_init returned error for mode %s\n",
                                      bootstrap_modes[mode].c_str());
            } else {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "bootstrap_pmi_init invalid PMI bootstrap '%s'\n",
                                   nvshmemi_options.BOOTSTRAP_PMI);
            }
            break;
        case BOOTSTRAP_PLUGIN:
            assert(attr == NULL);

            if (!nvshmemi_options.BOOTSTRAP_PLUGIN_provided) {
                NVSHMEMI_ERROR_PRINT(
                    "Plugin bootstrap requires NVSHMEM_BOOTSTRAP_PLUGIN to be set\n");
                status = 1;
                goto out;
            }

            plugin_name = nvshmemi_options.BOOTSTRAP_PLUGIN;

            status = bootstrap_loader_init(nvshmemi_options.BOOTSTRAP_PLUGIN, NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        case BOOTSTRAP_UID:
            assert(attr != NULL);
            plugin_name = nvshmemi_options.BOOTSTRAP_UID_PLUGIN;

            status = bootstrap_loader_init(plugin_name, (attr->uid_args), handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;

        default:
            NVSHMEMI_ERROR_PRINT("Invalid bootstrap mode selected\n");
    }

out:
    return status;
}

void bootstrap_finalize() {
    int status = NVSHMEMX_SUCCESS;

    if (nvshmemi_is_nvshmem_bootstrapped) {
        status = bootstrap_loader_finalize(&nvshmemi_boot_handle);
        NVSHMEMI_NZ_EXIT(status, "bootstrap finalization returned error\n");
        // Finalize the nvshmemi_session
        if (nvshmemi_default_session) {
            free(nvshmemi_default_session);
            nvshmemi_default_session = nullptr;
        }
        NVSHMEMU_THREAD_CS_FINALIZE();
    }
}
