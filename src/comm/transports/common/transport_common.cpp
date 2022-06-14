/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_common.h"

#include <dlfcn.h>

int nvshmemt_parse_hca_list(const char *string, struct nvshmemt_hca_info *hca_list, int max_count) {
    if (!string) return 0;

    const char *ptr = string;
    // Ignore "^" name, will be detected outside of this function
    if (ptr[0] == '^') ptr++;

    int if_num = 0;
    int if_counter = 0;
    int segment_counter = 0;
    char c;
    do {
        c = *ptr;
        if (c == ':') {
            if (segment_counter == 0) {
                if (if_counter > 0) {
                    hca_list[if_num].name[if_counter] = '\0';
                    hca_list[if_num].port = atoi(ptr + 1);
                    hca_list[if_num].found = 0;
                    if_num++;
                    if_counter = 0;
                    segment_counter++;
                }
            } else {
                hca_list[if_num - 1].count = atoi(ptr + 1);
                segment_counter = 0;
            }
            c = *(ptr + 1);
            while (c != ',' && c != ':' && c != '\0') {
                ptr++;
                c = *(ptr + 1);
            }
        } else if (c == ',' || c == '\0') {
            if (if_counter > 0) {
                hca_list[if_num].name[if_counter] = '\0';
                hca_list[if_num].found = 0;
                if_num++;
                if_counter = 0;
            }
            segment_counter = 0;
        } else {
            if (if_counter == 0) {
                hca_list[if_num].port = -1;
                hca_list[if_num].count = 1;
            }
            hca_list[if_num].name[if_counter] = c;
            if_counter++;
        }
        ptr++;
    } while (if_num < max_count && c);

    INFO(NVSHMEM_INIT, "Begin - Parsed HCA list provided by user - ");
    for (int i = 0; i < if_num; i++) {
        INFO(NVSHMEM_INIT,
             "Parsed HCA list provided by user - i=%d (of %d), name=%s, port=%d, count=%d", i,
             if_num, hca_list[i].name, hca_list[i].port, hca_list[i].count);
    }
    INFO(NVSHMEM_INIT, "End - Parsed HCA list provided by user");

    return if_num;
}

int nvshmemt_ib_iface_get_mlx_path(const char *ib_name, char **path) {
    int status;

    char device_path[MAXPATHSIZE];
    status = snprintf(device_path, MAXPATHSIZE, "/sys/class/infiniband/%s/device", ib_name);
    if (status < 0 || status >= MAXPATHSIZE) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to fill in device name.\n");
    } else {
        status = NVSHMEMX_SUCCESS;
    }

    *path = realpath(device_path, NULL);
    NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

#ifdef NVSHMEM_USE_GDRCOPY
bool nvshmemt_gdrcopy_ftable_init(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc, void **gdrcopy_handle) {
    bool use_gdrcopy = true;
    void *local_gdrcopy_handle;
    if (nvshmemi_options.DISABLE_GDRCOPY) {
        use_gdrcopy = false;
        goto skip_gdrcopy_dlsym;
    }

    *gdrcopy_handle = dlopen("libgdrapi.so.2", RTLD_LAZY);
    if (!*gdrcopy_handle) {
        INFO(NVSHMEM_INIT, "GDRCopy library not found. disabling GDRCopy.\n");
        use_gdrcopy = false;
        goto skip_gdrcopy_dlsym;
    } else {
        local_gdrcopy_handle = *gdrcopy_handle;
        LOAD_SYM(local_gdrcopy_handle, "gdr_runtime_get_version", gdrcopy_ftable->runtime_get_version);
        if (!gdrcopy_ftable->runtime_get_version) {
            INFO(NVSHMEM_INIT, "GDRCopy library found by version older than 2.0. disabling GDRCopy.\n");
            use_gdrcopy = false;
            goto skip_gdrcopy_dlsym;
        }
        LOAD_SYM(local_gdrcopy_handle, "gdr_runtime_get_version", gdrcopy_ftable->driver_get_version);
        LOAD_SYM(local_gdrcopy_handle, "gdr_open", gdrcopy_ftable->open);
        LOAD_SYM(local_gdrcopy_handle, "gdr_close", gdrcopy_ftable->close);
        LOAD_SYM(local_gdrcopy_handle, "gdr_pin_buffer", gdrcopy_ftable->pin_buffer);
        LOAD_SYM(local_gdrcopy_handle, "gdr_unpin_buffer", gdrcopy_ftable->unpin_buffer);
        LOAD_SYM(local_gdrcopy_handle, "gdr_map", gdrcopy_ftable->map);
        LOAD_SYM(local_gdrcopy_handle, "gdr_unmap", gdrcopy_ftable->unmap);
        LOAD_SYM(local_gdrcopy_handle, "gdr_get_info", gdrcopy_ftable->get_info);
        LOAD_SYM(local_gdrcopy_handle, "gdr_copy_from_mapping", gdrcopy_ftable->copy_from_mapping);
        LOAD_SYM(local_gdrcopy_handle, "gdr_copy_to_mapping", gdrcopy_ftable->copy_to_mapping);
    }

    *gdr_desc = gdrcopy_ftable->open();
    if (!*gdr_desc) {
        dlclose(*gdrcopy_handle);
        *gdrcopy_handle = NULL;
        INFO(NVSHMEM_INIT, "GDRCopy open call failed, disabling GDRCopy.\n");
    }

skip_gdrcopy_dlsym:
    return use_gdrcopy;
}

void nvshmemt_gdrcopy_ftable_fini(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc, void **gdrcopy_handle) {
    if (gdrcopy_ftable->close && gdr_desc) {
        gdrcopy_ftable->close(*gdr_desc);
    }

    if (gdrcopy_handle && *gdrcopy_handle) {
        dlclose(*gdrcopy_handle);
        *gdrcopy_handle = NULL;
    }
}
#endif

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable) {
    *ibv_handle = dlopen("libibverbs.so.1", RTLD_LAZY);
    if (*ibv_handle == NULL) {
        INFO(NVSHMEM_INIT, "libibverbs not found on the system.");
        return -1;
    }

    LOAD_SYM(*ibv_handle, "ibv_fork_init", ftable->fork_init);
    LOAD_SYM(*ibv_handle, "ibv_create_ah", ftable->create_ah);
    LOAD_SYM(*ibv_handle, "ibv_get_device_list", ftable->get_device_list);
    LOAD_SYM(*ibv_handle, "ibv_get_device_name", ftable->get_device_name);
    LOAD_SYM(*ibv_handle, "ibv_open_device", ftable->open_device);
    LOAD_SYM(*ibv_handle, "ibv_close_device", ftable->close_device);
    LOAD_SYM(*ibv_handle, "ibv_query_port", ftable->query_port);
    LOAD_SYM(*ibv_handle, "ibv_query_device", ftable->query_device);
    LOAD_SYM(*ibv_handle, "ibv_alloc_pd", ftable->alloc_pd);
    LOAD_SYM(*ibv_handle, "ibv_reg_mr", ftable->reg_mr);
    LOAD_SYM(*ibv_handle, "ibv_reg_dmabuf_mr", ftable->reg_dmabuf_mr);
    LOAD_SYM(*ibv_handle, "ibv_dereg_mr", ftable->dereg_mr);
    LOAD_SYM(*ibv_handle, "ibv_create_cq", ftable->create_cq);
    LOAD_SYM(*ibv_handle, "ibv_create_qp", ftable->create_qp);
    LOAD_SYM(*ibv_handle, "ibv_create_srq", ftable->create_srq);
    LOAD_SYM(*ibv_handle, "ibv_modify_qp", ftable->modify_qp);
    LOAD_SYM(*ibv_handle, "ibv_query_gid", ftable->query_gid);
    LOAD_SYM(*ibv_handle, "ibv_destroy_qp", ftable->destroy_qp);
    LOAD_SYM(*ibv_handle, "ibv_destroy_cq", ftable->destroy_cq);
    LOAD_SYM(*ibv_handle, "ibv_destroy_srq", ftable->destroy_srq);

    return 0;
}

void nvshmemt_ibv_ftable_fini(void **ibv_handle) {
    int status;

    if (ibv_handle) {
        status = dlclose(*ibv_handle);
        if (status) {
            WARN("Unable to close libibverbs handle.");
        }
    }
}
