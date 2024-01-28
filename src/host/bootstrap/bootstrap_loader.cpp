/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "modules/common/nvshmemi_bootstrap_defines.h"
#include "modules/bootstrap/nvshmemi_bootstrap.h"
#include "host/nvshmemx_error.h"

#define GET_SYMBOL(lib_handle, name, var, status)                                                \
    do {                                                                                         \
        void **var_ptr = (void **)&(var);                                                        \
        void *tmp = (void *)dlsym(lib_handle, name);                                             \
        NVSHMEMI_NULL_ERROR_JMP(tmp, status, NVSHMEMX_ERROR_INTERNAL, out,                       \
                                "Bootstrap failed to get symbol '%s'\n\t%s\n", name, dlerror()); \
        *var_ptr = tmp;                                                                          \
    } while (0)

static void *plugin_hdl = nullptr;
static char *plugin_name = nullptr;

int bootstrap_loader_finalize(bootstrap_handle_t *handle) {
    int status = handle->finalize(handle);

    if (status != 0)
        NVSHMEMI_ERROR_PRINT("Bootstrap plugin finalize failed for '%s'\n", plugin_name);

    dlclose(plugin_hdl);
    plugin_hdl = nullptr;
    free(plugin_name);
    plugin_name = nullptr;

    return 0;
}

static int _bootstrap_loader_init_helper(const char *func, const char *plugin, void *arg,
                                         bootstrap_handle_t *handle) {
    int (*bootstrap_plugin_initops)(void *arg, bootstrap_handle_t *handle, int nvshmem_version);
    int status = 0;

    dlerror(); /* Clear any existing error */
    if (plugin_name == nullptr) {
        plugin_name = strdup(plugin);
    }

    if (plugin_hdl == nullptr) {
        plugin_hdl = dlopen(plugin, RTLD_NOW);
    }

    NVSHMEMI_NULL_ERROR_JMP(plugin_hdl, status, -1, error, "Bootstrap unable to load '%s'\n\t%s\n",
                            plugin, dlerror());

    dlerror(); /* Clear any existing error */
    GET_SYMBOL(plugin_hdl, func, bootstrap_plugin_initops, status);

    status = bootstrap_plugin_initops(arg, handle, NVSHMEMI_BOOTSTRAP_ABI_VERSION);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap plugin init failed for '%s'\n", plugin);

    goto out;

error:
    if (plugin_hdl != nullptr) {
        dlclose(plugin_hdl);
        plugin_hdl = nullptr;
    }

    if (plugin_name != nullptr) {
        free(plugin_name);
        plugin_name = nullptr;
    }

out:
    return status;
}

int bootstrap_loader_preinit(const char *plugin, void *arg, bootstrap_handle_t *handle) {
    return _bootstrap_loader_init_helper("nvshmemi_bootstrap_plugin_pre_init", plugin, arg, handle);
}

int bootstrap_loader_init(const char *plugin, void *arg, bootstrap_handle_t *handle) {
    return _bootstrap_loader_init_helper("nvshmemi_bootstrap_plugin_init", plugin, arg, handle);
}
