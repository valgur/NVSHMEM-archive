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
#define MAX_MSG_SIZE 64 * 1024
#define UNROLL 8

__global__ void pull(volatile int *data_d, int len, int pe, int iter, int skip) {
    long long int start, stop;
    double usec, time;
    int i, j, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();

        if (!pe) {
            for (j = 0; j < len; j += THREADS) {
                if (j + tid < len)
                    *(data_d + j + tid) = nvshmem_int_g((int *)data_d + j + tid, peer);
            }

            __syncthreads();

        }
    }
    stop = clock64();

    if ((pe == 0) && !tid) {
        time = (stop - start) / iter;
        usec = time * 1000 / clockrate;
        printf("%7lu \t %8.2f \n", len * sizeof(int), usec);
    }
}

int main(int c, char *v[]) {
    int mype, npes, size;
    int *data_d = NULL;

    int iter = 200;
    int skip = 20;
    int max_msg_size = MAX_MSG_SIZE;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    data_d = (int *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        printf("   size(bytes) \t latency(us)\n");
        fflush(stdout);
    }

    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        int nelems;
        nelems = size / sizeof(int);

        pull<<<1, THREADS>>>(data_d, nelems, mype, iter, skip);

        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
    }

finalize:

    if (data_d) nvshmem_free(data_d);

    finalize_wrapper();

    return 0;
}
