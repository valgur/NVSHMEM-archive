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
#define LARGEST_DT long

#ifdef MAX_ITERS
#undef MAX_ITERS
#endif
#define MAX_ITERS 50

#define CALL_RDXN_BLOCK(DATATYPE, OP, usec)                                                \
    do {                                                                                   \
        double time;                                                                       \
        int i;                                                                             \
        int iter = MAX_ITERS;                                                              \
        int skip = MAX_SKIP;                                                               \
        long long int start = 0, stop = 0;                                                 \
                                                                                           \
        time = 0;                                                                          \
        for (i = 0; i < (iter + skip); i++) {                                              \
            nvshmemx_barrier_all_block();                                                  \
            if (i > skip) start = clock64();                                               \
            nvshmemx_##DATATYPE##_##OP##_to_all_block(dest, source, nelems, PE_start,      \
                                                      logPE_stride, PE_size, pWrk, pSync); \
            if (i > skip) stop = clock64();                                                \
            time += (stop - start);                                                        \
        }                                                                                  \
        nvshmemx_barrier_all_block();                                                      \
        if (!threadIdx.x && !mype) {                                                       \
            time = time / iter;                                                            \
            usec = time * 1000 / clockrate;                                                \
        }                                                                                  \
    } while (0)

#define CALL_RDXN_WARP(DATATYPE, OP, usec)                                                         \
    do {                                                                                           \
        double time;                                                                               \
        int i;                                                                                     \
        int iter = MAX_ITERS;                                                                      \
        int skip = MAX_SKIP;                                                                       \
        long long int start = 0, stop = 0;                                                         \
                                                                                                   \
        time = 0;                                                                                  \
        for (i = 0; i < (iter + skip); i++) {                                                      \
            nvshmemx_barrier_all_warp();                                                           \
            if (i > skip) start = clock64();                                                       \
            nvshmemx_##DATATYPE##_##OP##_to_all_warp(dest, source, nelems, PE_start, logPE_stride, \
                                                     PE_size, pWrk, pSync);                        \
            if (i > skip) stop = clock64();                                                        \
            time += (stop - start);                                                                \
        }                                                                                          \
        nvshmemx_barrier_all_warp();                                                               \
        if (!threadIdx.x && !mype) {                                                               \
            time = time / iter;                                                                    \
            usec = time * 1000 / clockrate;                                                        \
        }                                                                                          \
    } while (0)

#define CALL_RDXN_THREAD(DATATYPE, OP, usec)                                                 \
    do {                                                                                     \
        double time;                                                                         \
        int i;                                                                               \
        int iter = MAX_ITERS;                                                                \
        int skip = MAX_SKIP;                                                                 \
        long long int start = 0, stop = 0;                                                   \
                                                                                             \
        time = 0;                                                                            \
        for (i = 0; i < (iter + skip); i++) {                                                \
            nvshmem_barrier_all();                                                           \
            if (i > skip) start = clock64();                                                 \
            nvshmem_##DATATYPE##_##OP##_to_all(dest, source, nelems, PE_start, logPE_stride, \
                                               PE_size, pWrk, pSync);                        \
            if (i > skip) stop = clock64();                                                  \
            time += (stop - start);                                                          \
        }                                                                                    \
        nvshmem_barrier_all();                                                               \
        if (!threadIdx.x && !mype) {                                                         \
            time = time / iter;                                                              \
            usec = time * 1000 / clockrate;                                                  \
        }                                                                                    \
    } while (0)

#define DEFN_RDXN_BLOCK_FXNS(DATATYPE)                                                           \
    __global__ void test_##DATATYPE##_call_kern_block(                                           \
        DATATYPE *dest, const DATATYPE *source, size_t nelems, int PE_start, int logPE_stride,   \
        int PE_size, DATATYPE *pWrk, long *pSync, int mype, double *d_time_avg) {                \
        double usec_sum = 0;                                                                     \
        double usec_prod = 0;                                                                    \
        double usec_and = 0;                                                                     \
        double usec_or = 0;                                                                      \
        double usec_xor = 0;                                                                     \
        double usec_min = 0;                                                                     \
        double usec_max = 0;                                                                     \
                                                                                                 \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
        if (!blockIdx.x && nelems < 65536) {                                                     \
            CALL_RDXN_BLOCK(DATATYPE, sum, usec_sum);                                            \
            CALL_RDXN_BLOCK(DATATYPE, prod, usec_prod);                                          \
            CALL_RDXN_BLOCK(DATATYPE, and, usec_and);                                            \
            CALL_RDXN_BLOCK(DATATYPE, or, usec_or);                                              \
            CALL_RDXN_BLOCK(DATATYPE, xor, usec_xor);                                            \
            CALL_RDXN_BLOCK(DATATYPE, min, usec_min);                                            \
            CALL_RDXN_BLOCK(DATATYPE, max, usec_max);                                            \
                                                                                                 \
            if (!threadIdx.x && !mype) {                                                         \
                if (!mype)                                                                       \
                    printf("|%14.0lu|%14.2lf|%15.2lf|%14.2lf|%13.2lf|%14.2lf|%14.2lf|%14.2lf|\n", \
                           nelems * sizeof(DATATYPE), usec_sum, usec_prod, usec_and, usec_or,    \
                           usec_xor, usec_min, usec_max);                                        \
            }                                                                                    \
        }                                                                                        \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
    }

#define DEFN_RDXN_WARP_FXNS(DATATYPE)                                                             \
    __global__ void test_##DATATYPE##_call_kern_warp(                                             \
        DATATYPE *dest, const DATATYPE *source, size_t nelems, int PE_start, int logPE_stride,    \
        int PE_size, DATATYPE *pWrk, long *pSync, int mype, double *d_time_avg) {                 \
        double usec_sum = 0;                                                                      \
        double usec_prod = 0;                                                                     \
        double usec_and = 0;                                                                      \
        double usec_or = 0;                                                                       \
        double usec_xor = 0;                                                                      \
        double usec_min = 0;                                                                      \
        double usec_max = 0;                                                                      \
                                                                                                  \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                            \
        if (!blockIdx.x && !(threadIdx.x / warpSize) && nelems < 4096) {                          \
            CALL_RDXN_WARP(DATATYPE, sum, usec_sum);                                              \
            CALL_RDXN_WARP(DATATYPE, prod, usec_prod);                                            \
            CALL_RDXN_WARP(DATATYPE, and, usec_and);                                              \
            CALL_RDXN_WARP(DATATYPE, or, usec_or);                                                \
            CALL_RDXN_WARP(DATATYPE, xor, usec_xor);                                              \
            CALL_RDXN_WARP(DATATYPE, min, usec_min);                                              \
            CALL_RDXN_WARP(DATATYPE, max, usec_max);                                              \
                                                                                                  \
            if (!threadIdx.x && !mype) {                                                          \
                if (!mype)                                                                        \
                    printf("|%14.0lu|%14.2lf|%15.2lf|%14.2lf|%13.2lf|%14.2lf|%14.2lf|%14.2lf|\n",  \
                           nelems * sizeof(DATATYPE), usec_sum, usec_prod, usec_and, usec_or,     \
                           usec_xor, usec_min, usec_max);                                         \
            }                                                                                     \
        }                                                                                         \
        __syncthreads();                                                                          \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                            \
    }

#define DEFN_RDXN_THREAD_FXNS(DATATYPE)                                                          \
    __global__ void test_##DATATYPE##_call_kern_thread(                                          \
        DATATYPE *dest, const DATATYPE *source, size_t nelems, int PE_start, int logPE_stride,   \
        int PE_size, DATATYPE *pWrk, long *pSync, int mype, double *d_time_avg) {                \
        double usec_sum = 0;                                                                     \
        double usec_prod = 0;                                                                    \
        double usec_and = 0;                                                                     \
        double usec_or = 0;                                                                      \
        double usec_xor = 0;                                                                     \
        double usec_min = 0;                                                                     \
        double usec_max = 0;                                                                     \
                                                                                                 \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
        if (!blockIdx.x && !threadIdx.x && nelems < 512) {                                       \
            CALL_RDXN_THREAD(DATATYPE, sum, usec_sum);                                           \
            CALL_RDXN_THREAD(DATATYPE, prod, usec_prod);                                         \
            CALL_RDXN_THREAD(DATATYPE, and, usec_and);                                           \
            CALL_RDXN_THREAD(DATATYPE, or, usec_or);                                             \
            CALL_RDXN_THREAD(DATATYPE, xor, usec_xor);                                           \
            CALL_RDXN_THREAD(DATATYPE, min, usec_min);                                           \
            CALL_RDXN_THREAD(DATATYPE, max, usec_max);                                           \
                                                                                                 \
            if (!threadIdx.x && !mype) {                                                         \
                if (!mype)                                                                       \
                    printf("|%14.0lu|%14.2lf|%15.2lf|%14.2lf|%13.2lf|%14.2lf|%14.2lf|%14.2lf|\n", \
                           nelems * sizeof(DATATYPE), usec_sum, usec_prod, usec_and, usec_or,    \
                           usec_xor, usec_min, usec_max);                                        \
            }                                                                                    \
        }                                                                                        \
        __syncthreads();                                                                         \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
    }

DEFN_RDXN_THREAD_FXNS(int);
DEFN_RDXN_THREAD_FXNS(long);
DEFN_RDXN_WARP_FXNS(int);
DEFN_RDXN_WARP_FXNS(long);
DEFN_RDXN_BLOCK_FXNS(int);
DEFN_RDXN_BLOCK_FXNS(long);

#define RUN_ITERS(DATATYPE, GROUP)                                                                 \
    do {                                                                                           \
        nvshmem_barrier_all();                                                                     \
        if (!mype) printf("# ------------\n");                                                     \
        if (!mype) printf("# " #DATATYPE " operand\n");                                            \
        if (!mype) printf("# ------------\n");                                                     \
        if (!mype)                                                                                 \
            printf(                                                                                \
                "+--------------+--------------+---------------+--------------+-------------+----" \
                "----------+--------------+--------------+\n");                                    \
        if (!mype)                                                                                 \
            printf(                                                                                \
                "| size (bytes) |   sum (us)   |   prod (us)   |   and (us)   |   or (us)   |   "  \
                "xor (us)   |   min (us)   |   max (us)   |\n");                                   \
        if (!mype)                                                                                 \
            printf(                                                                                \
                "+--------------+--------------+---------------+--------------+-------------+----" \
                "----------+--------------+--------------+\n");                                    \
        for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {                               \
            nvshmem_barrier_all();                                                                 \
            test_##DATATYPE##_call_kern_##GROUP<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(    \
                (DATATYPE *)dest, (const DATATYPE *)source, num_elems, PE_start, logPE_stride,     \
                PE_size, (DATATYPE *)pWrk, pSync, mype, d_time_avg);                               \
            cuda_check_error();                                                                    \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                             \
        }                                                                                          \
    } while (0)

int rdxn_calling_kernel(void *dest, const void *source, int mype, int PE_start, int logPE_stride,
                        int PE_size, void *pWrk, long *pSync, cudaStream_t stream,
                        double *d_time_avg, run_opt_t run_options) {
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

    // if (!mype) printf("Transfer size in bytes and latency of thread/warp/block variants of all
    // operations of reduction API in us\n");
    if (run_options.run_thread) {
        if (!mype) printf("# ------------\n");
        if (!mype) printf("# thread-coll\n");
        if (!mype) printf("# ------------\n");

        RUN_ITERS(int, thread);
        RUN_ITERS(long, thread);
    }

    if (run_options.run_warp) {
        if (!mype) printf("# ------------\n");
        if (!mype) printf("# warp-coll\n");
        if (!mype) printf("# ------------\n");

        RUN_ITERS(int, warp);
        RUN_ITERS(long, warp);
    }

    if (run_options.run_block) {
        if (!mype) printf("# ------------\n");
        if (!mype) printf("# block-coll\n");
        if (!mype) printf("# ------------\n");

        RUN_ITERS(int, block);
        RUN_ITERS(long, block);
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    int i = 0;
    size_t size = 0;
    size_t alloc_size;
    int num_elems;
    int *h_buffer = NULL;
    int *d_source, *d_dest;
    int *h_source, *h_dest;
    long *d_pSync;
    long *h_pSync;
    int *d_pWrk;
    int *h_pWrk;
    char size_string[100];
    int PE_start = 0;
    int PE_size;
    int logPE_stride = 0;
    double *d_time_avg;
    cudaStream_t cstrm;
    run_opt_t run_options;

    PROCESS_OPTS(run_options);

    size = page_size_roundoff((MAX_ELEMS) * sizeof(LARGEST_DT));   // send buf
    size += page_size_roundoff((MAX_ELEMS) * sizeof(LARGEST_DT));  // recv buf
    size += page_size_roundoff(NVSHMEM_REDUCE_SYNC_SIZE * sizeof(LARGEST_DT));
    size += page_size_roundoff(NVSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(LARGEST_DT));

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

    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    d_time_avg = (double *)nvshmem_align(getpagesize(), sizeof(double) * 2);

    PE_size = npes;
    logPE_stride = 0;
    PE_start = 0;

    num_elems = MAX_ELEMS / 2;
    alloc_size = ((num_elems * 2) + NVSHMEM_REDUCE_SYNC_SIZE + NVSHMEM_REDUCE_MIN_WRKDATA_SIZE) *
                 sizeof(long);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (int *)h_buffer;
    h_dest = (int *)&h_source[num_elems];
    h_pSync = (long *)&h_dest[num_elems];
    h_pWrk = (int *)&h_pSync[NVSHMEM_REDUCE_SYNC_SIZE];

    d_source = (int *)nvshmem_align(getpagesize(), num_elems * sizeof(LARGEST_DT));
    d_dest = (int *)nvshmem_align(getpagesize(), num_elems * sizeof(LARGEST_DT));
    d_pSync = (long *)nvshmem_align(getpagesize(), NVSHMEM_REDUCE_SYNC_SIZE * sizeof(long));
    d_pWrk =
        (int *)nvshmem_align(getpagesize(), NVSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(LARGEST_DT));

    for (i = 0; i < NVSHMEM_REDUCE_SYNC_SIZE; i++) {
        h_pSync[i] = NVSHMEM_SYNC_VALUE;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_pSync, h_pSync, (sizeof(long) * NVSHMEM_REDUCE_SYNC_SIZE),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_pWrk, h_pWrk,
                               (sizeof(LARGEST_DT) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE),
                               cudaMemcpyHostToDevice, cstrm));

    rdxn_calling_kernel(d_dest, d_source, mype, PE_start, logPE_stride, PE_size, d_pWrk, d_pSync,
                        cstrm, d_time_avg, run_options);

    DEBUG_PRINT("last error = %s\n", cudaGetErrorString(cudaGetLastError()));

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_pSync, d_pSync, (sizeof(long) * NVSHMEM_REDUCE_SYNC_SIZE),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_pWrk, d_pWrk,
                               (sizeof(LARGEST_DT) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_source);
    nvshmem_free(d_dest);
    nvshmem_free(d_pSync);
    nvshmem_free(d_pWrk);
    nvshmem_free(d_time_avg);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
