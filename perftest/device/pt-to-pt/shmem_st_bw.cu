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
#include <getopt.h>
#include "utils.h"

#define MAX_ITERS 10
#define MAX_SKIP 10
#define THREADS 1024
#define BLOCKS 4
#define MAX_MSG_SIZE 32 * 1024 * 1024
#define UNROLL 2

__global__ void bw(volatile double *data_d, volatile double *remote_d,
                   volatile unsigned int *counter_d, int len, int pe, int iter, int skip) {
    int u, i, j, tid, slice;
    unsigned int counter;
    long long int start = 0, stop = 0;
    double bw, time = 0;
    int threads = gridDim.x * blockDim.x;
    tid = blockIdx.x * blockDim.x + threadIdx.x;

    slice = UNROLL * threads;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) {
            nvshmem_quiet();
            start = clock64();
        }

        for (j = 0; j < len - slice; j += slice) {
            for (u = 0; u < UNROLL; ++u) {
                int idx = j + u * threads + tid;
                *(remote_d + idx) = *(data_d + idx);
            }
            __syncthreads();
        }

        for (u = 0; u < UNROLL; ++u) {
            int idx = j + u * threads + tid;
            if (idx < len) *(remote_d + idx) = *(data_d + idx);
        }

        // synchronizing across blocks
        __syncthreads();

        if (!threadIdx.x) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1)
                ;
        }

        __syncthreads();
    }

    // synchronizing across blocks
    __syncthreads();

    if (!threadIdx.x) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }

    __syncthreads();
    stop = clock64();
    time = (stop - start);

    if (!threadIdx.x && !blockIdx.x) {
        bw = ((float)iter * (float)len * sizeof(double) * clockrate) / ((time / 1000) * 1024 * 1024 * 1024);
        printf("%10lu \t\t %4.6f \n", len * sizeof(double), bw);
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    double *data_d = NULL, *remote_d;
    unsigned int *counter_d;
    int max_blocks = BLOCKS, max_threads = THREADS;

    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    int max_msg_size = MAX_MSG_SIZE;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    while (1) {
        int c;
        c = getopt(argc, argv, "c:t:h");
        if (c == -1) break;

        switch (c) {
            case 'c':
                max_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                max_threads = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf("-c [CTAs] -t [THREADS] \n");
                goto finalize;
        }
    }

    data_d = (double *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    remote_d = (double *)nvshmem_ptr((void *)data_d, !mype);
    if (remote_d == NULL) {
        fprintf(stderr, "peer memory not accessible for LD/ST \n");
        goto finalize;
    }

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        printf("Size(Bytes) \t\t BW(GB/sec)\n");
        fflush(stdout);
    }

    int size;
    if (mype == 0) {
        for (size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            int blocks = max_blocks, threads = max_threads;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            bw<<<blocks, threads>>>(data_d, remote_d, counter_d, size / sizeof(double), mype, iter,
                                    skip);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaDeviceSynchronize());

            nvshmem_barrier_all();
        }
    } else {
        for (size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            nvshmem_barrier_all();
        }
    }

finalize:

    if (data_d) nvshmem_free(data_d);

    finalize_wrapper();

    return 0;
}
