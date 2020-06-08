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

#define MAX_MSG_SIZE 1 * 1024 * 1024
#define UNROLL 8

__global__ void ping_pong(volatile int *data_d, uint64_t *flag_d,
                          int len, int pe, int iter, int skip, int *hflag) {
    long long int start, stop;
    double usec, time;
    int i, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();

        if (pe) {
            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));

            nvshmem_int_put_nbi((int *)data_d, (int *)data_d, len, peer);

            nvshmem_fence();

            nvshmem_uint64_atomic_inc(flag_d, peer);
        } else {
            nvshmem_int_put_nbi((int *)data_d, (int *)data_d, len, peer);

            nvshmem_fence();

            nvshmem_uint64_atomic_inc(flag_d, peer);

            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));
        }
    }
    stop = clock64();
    nvshmem_quiet();
    *hflag = 1;

    if ((pe == 0) && !tid) {
        time = (stop - start) / iter;
        usec = time * 1000 / clockrate;
        printf("%7lu \t %8.2f \n", len * sizeof(int), usec);
    }
}

int main(int c, char *v[]) {
    int mype, npes, size;
    uint64_t *flag_d = NULL;
    int *data_d = NULL;
    cudaStream_t stream;

    int iter = 500;
    int skip = 50;
    int max_msg_size = MAX_MSG_SIZE;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    data_d = (int *)nvshmem_malloc(max_msg_size);
    flag_d = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

    int *hflag, *hflag_d;
    CUDA_CHECK(cudaHostAlloc((void **)&hflag, sizeof(uint64_t), 0));
    *hflag = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&hflag_d, hflag, 0));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
        printf("   size(bytes) \t latency(us)\n");
        fflush(stdout);
    }

    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        int nelems, status = 0;
        nelems = size / sizeof(int);
        void *args[] = {&data_d, &flag_d, &nelems, &mype, &iter, &skip, &hflag_d};

        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        *hflag = 0;
        status = nvshmemx_collective_launch((const void *)ping_pong, 1, 1, args, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }

        while (*((volatile int *)hflag) != 1)
            ;

        nvshmem_barrier_all();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

finalize:

    if (data_d) nvshmem_free(data_d);
    if (flag_d) nvshmem_free(flag_d);

    finalize_wrapper();

    return 0;
}
