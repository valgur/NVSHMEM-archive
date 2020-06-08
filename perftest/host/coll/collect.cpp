/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
#define DATATYPE int

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    int i = 0;
    size_t size = (MAX_ELEMS * (MAX_NPES + 1) * sizeof(DATATYPE)) +
                  (NVSHMEM_COLLECT_SYNC_SIZE * sizeof(long));
    size_t alloc_size;
    int num_elems;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    long *d_pSync;
    long *h_pSync;
    char size_string[100];
    cudaStream_t stream;

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    assert(npes <= MAX_NPES);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    num_elems = MAX_ELEMS / 2;
    alloc_size =
        (num_elems * (npes + 1) * sizeof(DATATYPE)) + (NVSHMEM_COLLECT_SYNC_SIZE * sizeof(long));

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[num_elems];
    h_pSync = (long *)((DATATYPE *)&h_dest[num_elems * npes]);

    d_buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (DATATYPE *)d_buffer;
    d_dest = (DATATYPE *)&d_source[num_elems];
    d_pSync = (long *)((DATATYPE *)&d_dest[num_elems * npes]);

    for (i = 0; i < NVSHMEM_COLLECT_SYNC_SIZE; i++) {
        h_pSync[i] = NVSHMEM_SYNC_VALUE;
    }

    if (!mype) printf("# --------------\n");
    if (!mype) printf("# 32-bit operand\n");
    if (!mype) printf("# --------------\n");
    if (!mype) printf("+--------------+------------------+\n");
    if (!mype) printf("| size (bytes) |   latency (us)   |\n");
    if (!mype) printf("+--------------+------------------+\n");
    RUN_COLL(collect, COLLECT, int, 32, d_source, h_source, d_dest, h_dest, d_pSync, h_pSync, npes,
             -1, stream);

    if (!mype) printf("# --------------\n");
    if (!mype) printf("# 64-bit operand\n");
    if (!mype) printf("# --------------\n");
    if (!mype) printf("+--------------+------------------+\n");
    if (!mype) printf("| size (bytes) |   latency (us)   |\n");
    if (!mype) printf("+--------------+------------------+\n");
    RUN_COLL(collect, COLLECT, long, 64, d_source, h_source, d_dest, h_dest, d_pSync, h_pSync, npes,
             -1, stream);

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return 0;
}
