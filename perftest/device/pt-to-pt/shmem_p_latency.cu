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

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#define THREADS 512
#define MAX_MSG_SIZE 64 * 1024 * 1024
#define UNROLL 8

__global__ void p_latency(volatile int *data_d, volatile int *flag_d, int len, int pe, int iter,
                          int skip) {
    long long int start, stop;
    double usec, time;
    int i, j, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();

        for (j = tid; j < len; j += THREADS) {
            nvshmem_int_p((int *)data_d + j, *(data_d + j), peer);
        }
        __syncthreads();
        if (!tid) {
            nvshmem_quiet();
        }
        __syncthreads();
    }
    stop = clock64();

    if (!tid) {
        time = (stop - start) / iter;
        usec = time * 1000 / clockrate;
        printf("%7lu \t %8.2f \n", len * sizeof(int), usec);
    }
}

int main(int c, char *v[]) {
    int mype, npes, size;
    int *flag_d = NULL, *data_d = NULL;

    int iter = 50;
    int skip = 5;
    int max_msg_size = MAX_MSG_SIZE;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    data_d = (int *)nvshmem_malloc(max_msg_size);
    flag_d = (int *)nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        printf("   size(bytes) \t latency(us)\n");
        fflush(stdout);
    }

    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        if (!mype) {
            int nelems;
            nelems = size / sizeof(int);

            p_latency<<<1, THREADS>>>(data_d, flag_d, nelems, mype, iter, skip);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        nvshmem_barrier_all();
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    if (flag_d) nvshmem_free(flag_d);

    finalize_wrapper();

    return 0;
}
