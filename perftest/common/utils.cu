/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "utils.h"

double *d_latency = NULL;
double *d_avg_time = NULL;
double *d_r_pWrk, *h_r_pWrk;
long *d_r_pSync, *h_r_pSync;
double *latency = NULL;
double *avg_time = NULL;
double *r_pWrk;
long *r_pSync;
int mype = 0;
int npes = 0;
int use_mpi = 0;
int use_shmem = 0;
__device__ int clockrate;

void select_device() {
    cudaDeviceProp prop;
    int dev_count;
    int mype_node;

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node % dev_count));

    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    fprintf(stderr, "mype: %d device name: %s bus id: %d \n", mype_node, prop.name, prop.pciBusID);
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0,
                                  cudaMemcpyHostToDevice));
}

void init_wrapper(int *c, char ***v) {
    char *value;

#ifdef NVSHMEM_MPI_SUPPORT
    value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);
#endif

#ifdef NVSHMEM_SHMEM_SUPPORT
    value = getenv("NVSHMEMTEST_USE_SHMEM_LAUNCHER");
    if (value) use_shmem = atoi(value);
#endif

#ifdef NVSHMEM_MPI_SUPPORT
    if (use_mpi) {
        MPI_Init(c, v);
        int rank, nranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        DEBUG_PRINT("MPI: [%d of %d] hello MPI world! \n", rank, nranks);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        select_device();

        return;
    }
#endif

#ifdef NVSHMEM_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_init();
        mype = shmem_my_pe();
        npes = shmem_n_pes();
        DEBUG_PRINT("SHMEM: [%d of %d] hello SHMEM world! \n", my_pe, n_pes);

        latency = (double *)shmem_malloc(sizeof(double));
        if (!latency) ERROR_EXIT("(shmem_malloc) failed \n");

        avg_time = (double *)shmem_malloc(sizeof(double));
        if (!avg_time) ERROR_EXIT("(shmem_malloc) failed \n");

        r_pWrk = (double *)shmem_malloc(sizeof(long) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE);
        if (!r_pWrk) ERROR_EXIT("(shmem_malloc) failed \n");

        r_pSync = (long *)shmem_malloc(sizeof(long) * NVSHMEM_REDUCE_SYNC_SIZE);
        if (!r_pSync) ERROR_EXIT("(shmem_malloc) failed \n");

        for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
            r_pSync[i] = SHMEM_SYNC_VALUE;
        }

        for (int i = 0; i < SHMEM_REDUCE_MIN_WRKDATA_SIZE; i++) {
            r_pWrk[i] = SHMEM_WRK_VALUE;
        }

        nvshmemx_init_attr_t attr;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);

        select_device();

        return;
    }
#endif

    value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    // if test set value, extend it for reduction
    if (value) {
        char size_string[100];

        size_t size = 0;
        size = atoi(value);
        size += ((NVSHMEM_REDUCE_SYNC_SIZE * sizeof(long)) +
                 ((NVSHMEM_REDUCE_MIN_WRKDATA_SIZE + 2) * sizeof(double)));
        sprintf(size_string, "%lu", size);

        int status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
        if (status) ERROR_EXIT("setenv failed \n");
    }

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    select_device();

    d_latency = (double *)nvshmem_malloc(sizeof(double));
    if (!d_latency) ERROR_EXIT("nvshmem_malloc failed \n");

    d_avg_time = (double *)nvshmem_malloc(sizeof(double));
    if (!d_avg_time) ERROR_EXIT("nvshmem_malloc failed \n");

    d_r_pWrk = (double *)nvshmem_malloc(sizeof(long) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE);
    if (!d_r_pWrk) ERROR_EXIT("nvshmem_malloc failed \n");

    d_r_pSync = (long *)nvshmem_malloc(sizeof(long) * NVSHMEM_REDUCE_SYNC_SIZE);
    if (!d_r_pSync) ERROR_EXIT("nvshmem_malloc failed \n");

    CUDA_CHECK(cudaHostAlloc(&h_r_pWrk, sizeof(double) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE,
                             cudaHostAllocDefault));

    CUDA_CHECK(
        cudaHostAlloc(&h_r_pSync, sizeof(long) * NVSHMEM_REDUCE_SYNC_SIZE, cudaHostAllocDefault));

    for (int i = 0; i < NVSHMEM_REDUCE_SYNC_SIZE; i++) {
        h_r_pSync[i] = NVSHMEM_SYNC_VALUE;
    }

    for (int i = 0; i < NVSHMEM_REDUCE_MIN_WRKDATA_SIZE; i++) {
        h_r_pWrk[i] = SHMEM_WRK_VALUE;
    }

    DEBUG_PRINT("end of init \n");
    return;
}

void finalize_wrapper() {
#ifdef NVSHMEM_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_free(r_pWrk);
        shmem_free(r_pSync);
        shmem_free(latency);
        shmem_free(avg_time);
    }
#endif

#if !defined(NVSHMEM_SHMEM_SUPPORT) && !defined(NVSHMEM_MPI_SUPPORT)
    if (!use_mpi && !use_shmem) {
        CUDA_CHECK(cudaFreeHost(h_r_pWrk));
        CUDA_CHECK(cudaFreeHost(h_r_pSync));
        nvshmem_free(d_r_pWrk);
        nvshmem_free(d_r_pSync);
        nvshmem_free(d_latency);
        nvshmem_free(d_avg_time);
    }
#endif
    nvshmem_finalize();

#ifdef NVSHMEM_MPI_SUPPORT
    if (use_mpi) MPI_Finalize();
#endif
#ifdef NVSHMEM_SHMEM_SUPPORT
    if (use_shmem) shmem_finalize();
#endif
}

void reduce_double_wrapper(double *source, double *target) {
#ifdef NVSHMEM_MPI_SUPPORT
    if (use_mpi) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(source, target, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        return;
    }
#endif
#ifdef NVSHMEM_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_barrier_all();
        shmem_double_sum_to_all(avg_time, latency, 1, 0, 0, npes, r_pWrk, r_pSync);

        return;
    }
#endif

    CUDA_CHECK(cudaMemcpy(d_latency, source, sizeof(double), cudaMemcpyHostToDevice));
    nvshmem_barrier_all();
    nvshmem_double_sum_to_all(d_avg_time, d_latency, 1, 0 /*PE_start*/, 0 /*logPE_stride*/, npes,
                              d_r_pWrk, d_r_pSync);
    CUDA_CHECK(cudaMemcpy(target, d_avg_time, sizeof(double), cudaMemcpyDeviceToHost));
}
