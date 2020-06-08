/*
 * * Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */
#ifndef _BOOTSTRAP_INTERNAL_H_
#define _BOOTSTRAP_INTERNAL_H_

// MPI
int bootstrap_mpi_init(void *mpi_comm, bootstrap_handle_t *handle);
int bootstrap_mpi_finalize(bootstrap_handle_t *handle);

// SHMEM
int bootstrap_shmem_init(bootstrap_handle_t *handle);
int bootstrap_shmem_finalize(bootstrap_handle_t *handle);

// PMI
int bootstrap_pmi_init(bootstrap_handle_t *handle);
int bootstrap_pmi_finalize(bootstrap_handle_t *handle);

// Sockets

#endif
