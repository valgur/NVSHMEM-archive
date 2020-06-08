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
#define LARGEST_DT int

#define RUN_RDXN(TYPE, OP, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride, \
                 PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream)                               \
    do {                                                                                          \
        int iters = MAX_ITERS;                                                                    \
        int skip = MAX_SKIP;                                                                      \
        struct timeval t_start, t_stop;                                                           \
        for (iters = 0; iters < MAX_ITERS + skip; iters++) {                                      \
            CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(TYPE) * num_elems),            \
                                       cudaMemcpyHostToDevice, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(TYPE) * num_elems),                \
                                       cudaMemcpyHostToDevice, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(d_pSync, h_pSync,                                          \
                                       (sizeof(TYPE) * NVSHMEM_REDUCE_SYNC_SIZE),                 \
                                       cudaMemcpyHostToDevice, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(d_pWrk, h_pWrk,                                            \
                                       (sizeof(TYPE) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE),          \
                                       cudaMemcpyHostToDevice, stream));                          \
                                                                                                  \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                            \
            nvshmemx_barrier_all_on_stream(stream);                                               \
                                                                                                  \
            if (iters >= skip) gettimeofday(&t_start, NULL);                                      \
                                                                                                  \
            nvshmemx_##TYPE##_##OP##_to_all_on_stream((TYPE *)d_dest, (const TYPE *)d_source,     \
                                                      num_elems, PE_start, logPE_stride, PE_size, \
                                                      (TYPE *)d_pWrk, d_pSync, stream);           \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                            \
                                                                                                  \
            if (iters >= skip) {                                                                  \
                gettimeofday(&t_stop, NULL);                                                      \
                latency += ((t_stop.tv_usec - t_start.tv_usec) +                                  \
                            (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));                           \
            }                                                                                     \
                                                                                                  \
            CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(TYPE) * num_elems),            \
                                       cudaMemcpyDeviceToHost, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(TYPE) * num_elems),                \
                                       cudaMemcpyDeviceToHost, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(h_pSync, d_pSync,                                          \
                                       (sizeof(TYPE) * NVSHMEM_REDUCE_SYNC_SIZE),                 \
                                       cudaMemcpyDeviceToHost, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(h_pWrk, d_pWrk,                                            \
                                       (sizeof(TYPE) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE),          \
                                       cudaMemcpyDeviceToHost, stream));                          \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                            \
        }                                                                                         \
    } while (0)

#define RUN_RDXN_ITERS(TYPE, d_source, h_source, d_dest, h_dest, num_elems, PE_start,              \
                       logPE_stride, PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream, mype)      \
    do {                                                                                           \
        double usec_sum = 0;                                                                       \
        double usec_prod = 0;                                                                      \
        double usec_and = 0;                                                                       \
        double usec_or = 0;                                                                        \
        double usec_xor = 0;                                                                       \
        double usec_min = 0;                                                                       \
        double usec_max = 0;                                                                       \
        double latency = 0;                                                                        \
        RUN_RDXN(TYPE, sum, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride, \
                 PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream);                               \
        usec_sum = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPE, prod, d_source, h_source, d_dest, h_dest, num_elems, PE_start,              \
                 logPE_stride, PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream);                 \
        usec_prod = latency / MAX_ITERS;                                                           \
        latency = 0;                                                                               \
        RUN_RDXN(TYPE, and, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride, \
                 PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream);                               \
        usec_and = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPE, or, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride,  \
                 PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream);                               \
        usec_or = latency / MAX_ITERS;                                                             \
        latency = 0;                                                                               \
        RUN_RDXN(TYPE, xor, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride, \
                 PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream);                               \
        usec_xor = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPE, min, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride, \
                 PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream);                               \
        usec_min = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPE, max, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride, \
                 PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream);                               \
        usec_max = latency / MAX_ITERS;                                                            \
        if (!mype)                                                                                 \
            printf("|%14.0lu|%14.2lf|%15.2lf|%14.2lf|%13.2lf|%14.2lf|%14.2lf|%14.2lf|\n",           \
                   num_elems * sizeof(TYPE), usec_sum, usec_prod, usec_and, usec_or, usec_xor,     \
                   usec_min, usec_max);                                                            \
    } while (0)

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    int i = 0;
    size_t size = ((MAX_ELEMS * 8) + NVSHMEM_REDUCE_SYNC_SIZE + NVSHMEM_REDUCE_MIN_WRKDATA_SIZE) *
                  sizeof(LARGEST_DT);
    size_t alloc_size;
    int num_elems;
    LARGEST_DT *h_buffer = NULL;
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *d_source, *d_dest;
    LARGEST_DT *h_source, *h_dest;
    long *d_pSync;
    long *h_pSync;
    LARGEST_DT *d_pWrk;
    LARGEST_DT *h_pWrk;
    int PE_start = 0;
    int PE_size;
    int logPE_stride = 0;
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
    CUDA_CHECK(cudaStreamCreate(&stream));

    PE_size = npes;
    logPE_stride = 0;
    PE_start = 0;

    num_elems = MAX_ELEMS / 2;
    alloc_size = ((num_elems * 2) + NVSHMEM_REDUCE_SYNC_SIZE + NVSHMEM_REDUCE_MIN_WRKDATA_SIZE) *
                 sizeof(long);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (LARGEST_DT *)h_buffer;
    h_dest = (LARGEST_DT *)&h_source[num_elems];
    h_pSync = (long *)&h_dest[num_elems];
    h_pWrk = (LARGEST_DT *)&h_pSync[NVSHMEM_REDUCE_SYNC_SIZE];

    d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (LARGEST_DT *)d_buffer;
    d_dest = (LARGEST_DT *)&d_source[num_elems];
    d_pSync = (long *)&d_dest[num_elems];
    d_pWrk = (LARGEST_DT *)&d_pSync[NVSHMEM_REDUCE_SYNC_SIZE];

    for (i = 0; i < NVSHMEM_REDUCE_SYNC_SIZE; i++) {
        h_pSync[i] = NVSHMEM_SYNC_VALUE;
    }

    if (!mype) printf("Latency of all operations of reduction API in us\n");
    if (!mype) printf("# ------------\n");
    if (!mype) printf("# int operand\n");
    if (!mype) printf("# ------------\n");
    if (!mype)
        printf(
            "+--------------+--------------+---------------+--------------+-------------+----------"
            "----+--------------+--------------+\n");
    if (!mype)
        printf(
            "| size (bytes) |   sum (us)   |   prod (us)   |   and (us)   |   or (us)   |   xor "
            "(us)   |   min (us)   |   max (us)   |\n");
    if (!mype)
        printf(
            "+--------------+--------------+---------------+--------------+-------------+----------"
            "----+--------------+--------------+\n");
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS(int, (int *)d_source, (int *)h_source, (int *)d_dest, (int *)h_dest,
                       num_elems, PE_start, logPE_stride, PE_size, d_pSync, h_pSync, (int *)d_pWrk,
                       (int *)h_pWrk, stream, mype);
    }

    if (!mype) printf("# ------------\n");
    if (!mype) printf("# long operand\n");
    if (!mype) printf("# ------------\n");
    if (!mype)
        printf(
            "+--------------+--------------+---------------+--------------+-------------+----------"
            "----+--------------+--------------+\n");
    if (!mype)
        printf(
            "| size (bytes) |   sum (us)   |   prod (us)   |   and (us)   |   or (us)   |   xor "
            "(us)   |   min (us)   |   max (us)   |\n");
    if (!mype)
        printf(
            "+--------------+--------------+---------------+--------------+-------------+----------"
            "----+--------------+--------------+\n");
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS(long, d_source, h_source, d_dest, h_dest, num_elems, PE_start, logPE_stride,
                       PE_size, d_pSync, h_pSync, d_pWrk, h_pWrk, stream, mype);
    }

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return status;
}
