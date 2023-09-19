/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __NVSHMEMI_TRANSPORT_DEFINES_H
#define __NVSHMEMI_TRANSPORT_DEFINES_H

#define NVSHMEM_MEM_HANDLE_SIZE 512

typedef struct pcie_identifier {
    int dev_id;
    int bus_id;
    int domain_id;
} pcie_id_t;

typedef struct nvshmem_mem_handle {
    char reserved[NVSHMEM_MEM_HANDLE_SIZE];
} nvshmem_mem_handle_t;

#endif
