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
#define PRINT_DATATYPE(a)       \
    do {                        \
        DEBUG_PRINT("%d\n", a); \
    } while (0)

__device__ double *pWrk;

#define CALL_ALLTOALL(BITS)                                                                        \
    __global__ void test_alltoall##BITS##_call_kern(void *dest, const void *source, size_t nelems, \
                                                    int PE_start, int logPE_stride, int PE_size,   \
                                                    long *pSync, int mype, double *d_time_avg) {   \
        int iter = MAX_ITERS;                                                                      \
        int skip = MAX_SKIP;                                                                       \
        long long int start = 0, stop = 0;                                                         \
        double time = 0;                                                                           \
        double thread_usec = 0, warp_usec = 0, block_usec = 0;                                     \
        int i;                                                                                     \
        double *dest_r, *source_r;                                                                 \
                                                                                                   \
        source_r = d_time_avg;                                                                     \
        dest_r = (double *)((double *)d_time_avg + 1);                                             \
                                                                                                   \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        time = 0;                                                                                  \
                                                                                                   \
        if (!blockIdx.x && !threadIdx.x && nelems < 512) {                                         \
            for (i = 0; i < (iter + skip); i++) {                                                  \
                nvshmem_barrier_all();                                                             \
                if (i > skip) start = clock64();                                                   \
                nvshmem_alltoall##BITS(dest, source, nelems, PE_start, logPE_stride, PE_size,      \
                                       pSync);                                                     \
                if (i > skip) stop = clock64();                                                    \
                time += (stop - start);                                                            \
            }                                                                                      \
            nvshmem_barrier_all();                                                                 \
            *source_r = time;                                                                      \
            nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size, pWrk,  \
                                      pSync);                                                      \
            time = *dest_r;                                                                        \
                                                                                                   \
            if (mype == 0) {                                                                       \
                time = time / iter;                                                                \
                time = time / PE_size;                                                             \
                thread_usec = time * 1000 / clockrate;                                             \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        __syncthreads();                                                                           \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        time = 0;                                                                                  \
                                                                                                   \
        if (!blockIdx.x && !(threadIdx.x / warpSize) && nelems < 4096) {                                 \
            for (i = 0; i < (iter + skip); i++) {                                                  \
                nvshmemx_barrier_all_warp();                                                       \
                if (i > skip) start = clock64();                                                   \
                nvshmemx_alltoall##BITS##_warp(dest, source, nelems, PE_start, logPE_stride,       \
                                               PE_size, pSync);                                    \
                if (i > skip) stop = clock64();                                                    \
                time += (stop - start);                                                            \
            }                                                                                      \
            nvshmemx_barrier_all_warp();                                                           \
            if (!threadIdx.x) {                                                                    \
                *source_r = time;                                                                  \
                nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size,    \
                                          pWrk, pSync);                                            \
                time = *dest_r;                                                                    \
                                                                                                   \
                if (mype == 0) {                                                                   \
                    time = time / iter;                                                            \
                    time = time / PE_size;                                                         \
                    warp_usec = time * 1000 / clockrate;                                           \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        __syncthreads();                                                                           \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        time = 0;                                                                                  \
                                                                                                   \
        if (!blockIdx.x) {                                                                         \
            for (i = 0; i < (iter + skip); i++) {                                                  \
                nvshmemx_barrier_all_block();                                                      \
                if (i > skip) start = clock64();                                                   \
                nvshmemx_alltoall##BITS##_block(dest, source, nelems, PE_start, logPE_stride,      \
                                                PE_size, pSync);                                   \
                if (i > skip) stop = clock64();                                                    \
                time += (stop - start);                                                            \
            }                                                                                      \
            nvshmemx_barrier_all_block();                                                          \
            if (!threadIdx.x) {                                                                    \
                *source_r = time;                                                                  \
                nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size,    \
                                          pWrk, pSync);                                            \
                time = *dest_r;                                                                    \
                                                                                                   \
                if (mype == 0) {                                                                   \
                    time = time / iter;                                                            \
                    time = time / PE_size;                                                         \
                    block_usec = time * 1000 / clockrate;                                          \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        if (!threadIdx.x && !blockIdx.x && !mype) {                                                \
            printf("|%14.0lu|%17.2lf|%15.2lf|%16.2lf|\n", nelems *BITS / 8, thread_usec, warp_usec, \
                   block_usec);                                                                    \
        }                                                                                          \
    }

CALL_ALLTOALL(32);
CALL_ALLTOALL(64);

int alltoall_calling_kernel(void *dest, const void *source, int mype, int PE_start,
                            int logPE_stride, int PE_size, long *pSync, cudaStream_t stream,
                            double *d_time_avg) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;
    int num_elems = 1;
    char *value = NULL;
    int max_elems = (MAX_ELEMS / 2);

    value = getenv("NVSHMEM_PERF_COLL_MAX_ELEMS");

    if (NULL != value) {
        max_elems = atoi(value);
        if (0 == max_elems) {
            fprintf(stderr, "Warning: min max elem size = 1\n");
            max_elems = 1;
        }
    }

    nvshmem_barrier_all();
    // if (!mype) printf("Transfer size in bytes and latency of thread/warp/block variants of
    // alltoall API in us\n");
    if (!mype) printf("# ------------\n");
    if (!mype) printf("# 32-bit operand\n");
    if (!mype) printf("# ------------\n");
    if (!mype) printf("+--------------+-----------------+---------------+----------------+\n");
    if (!mype) printf("| size (bytes) |   thread (us)   |   warp (us)   |   block (us)   |\n");
    if (!mype) printf("+--------------+-----------------+---------------+----------------+\n");
    for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {
        test_alltoall32_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(
            dest, source, num_elems, PE_start, logPE_stride, PE_size, pSync, mype, d_time_avg);
        cuda_check_error();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (!mype) printf("# ------------\n");
    if (!mype) printf("# 64-bit operand\n");
    if (!mype) printf("# ------------\n");
    if (!mype) printf("+--------------+-----------------+---------------+----------------+\n");
    if (!mype) printf("| size (bytes) |   thread (us)   |   warp (us)   |   block (us)   |\n");
    if (!mype) printf("+--------------+-----------------+---------------+----------------+\n");
    for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {
        test_alltoall64_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(
            dest, source, num_elems, PE_start, logPE_stride, PE_size, pSync, mype, d_time_avg);
        cuda_check_error();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    int i = 0;
    // size needs to hold psync array, source array (nelems) and dest array (nelems * npes)
    size_t size = ((MAX_ELEMS * (MAX_NPES)*2) * sizeof(DATATYPE)) +
                  (NVSHMEM_ALLTOALL_SYNC_SIZE * sizeof(long));
    size_t alloc_size;
    int num_elems;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    long *d_pSync;
    long *h_pSync;
    char size_string[100];
    double *d_time_avg;
    double *d_pWrk;
    cudaStream_t cstrm;

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

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    d_time_avg = (double *)nvshmem_malloc(sizeof(double) * 2);
    d_pWrk = (double *)nvshmem_malloc(sizeof(double) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE);
    CUDA_CHECK(
        cudaMemcpyToSymbol(pWrk, (void *)&d_pWrk, sizeof(double *), 0, cudaMemcpyHostToDevice));

    num_elems = MAX_ELEMS / 2;
    alloc_size = ((num_elems * (MAX_NPES)*2) * sizeof(DATATYPE)) +
                 (NVSHMEM_ALLTOALL_SYNC_SIZE * sizeof(long));

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[num_elems * npes];
    h_pSync = (long *)((DATATYPE *)&h_dest[num_elems * npes]);

    d_buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (DATATYPE *)d_buffer;
    d_dest = (DATATYPE *)&d_source[num_elems * npes];
    d_pSync = (long *)((DATATYPE *)&d_dest[num_elems * npes]);

    for (i = 0; i < NVSHMEM_ALLTOALL_SYNC_SIZE; i++) {
        h_pSync[i] = NVSHMEM_SYNC_VALUE;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_pSync, h_pSync, (sizeof(long) * NVSHMEM_ALLTOALL_SYNC_SIZE),
                               cudaMemcpyHostToDevice, cstrm));

    alltoall_calling_kernel(d_dest, d_source, mype, 0 /*PE_start*/, 0 /*logPE_stride*/, npes,
                            d_pSync, cstrm, d_time_avg);

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_pSync, d_pSync, (sizeof(long) * NVSHMEM_ALLTOALL_SYNC_SIZE),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);
    nvshmem_free(d_time_avg);
    nvshmem_free(d_pWrk);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
