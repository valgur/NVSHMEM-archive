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

#include "coll_test.h"

int main(int c, char *v[]) {
    int status = 0;
    int mype, npes;
    size_t size = NVSHMEM_BARRIER_SYNC_SIZE * sizeof(long);
    char *buffer = NULL;
    int iters = BARRIER_MAX_ITERS;
    int skip = BARRIER_MAX_SKIP;
    struct timeval t_start, t_stop;
    float ms;
    int PE_start = 0;
    int logPE_stride = 0;
    int PE_size;
    long *pSync = NULL;
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    PE_size = npes;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    buffer = (char *)nvshmem_malloc(size);
    if (!buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }
    pSync = (long *)buffer;

    for (iters = 0; iters < BARRIER_MAX_ITERS + skip; iters++) {
        if (iters == skip) CUDA_CHECK(cudaEventRecord(start_event, stream));

        nvshmemx_barrier_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);

    }
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

    if (0 == mype) printf("%s\t\t%lf\n", "latency (us)", (ms / BARRIER_MAX_ITERS) * 1000);

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    nvshmem_free(buffer);

    finalize_wrapper();

out:
    return status;
}
