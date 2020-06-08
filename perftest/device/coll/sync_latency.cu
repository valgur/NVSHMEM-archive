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

__device__ double *pWrk;

__global__ void test_sync_call_kern(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                    int mype, double *d_time_avg) {
    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    long long int start = 0, stop = 0;
    double thread_usec, warp_usec, block_usec, time = 0;
    int i;
    double *dest_r, *source_r;

    source_r = d_time_avg;
    dest_r = (double *)((double *)d_time_avg + 1);

    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !threadIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmem_sync(PE_start, logPE_stride, PE_size, pSync);
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size, pWrk,
                                      pSync);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                thread_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !(threadIdx.x / warpSize)) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_sync_warp(PE_start, logPE_stride, PE_size, pSync);
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size, pWrk,
                                      pSync);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                warp_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_sync_block(PE_start, logPE_stride, PE_size, pSync);
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size, pWrk,
                                      pSync);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                block_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x && !threadIdx.x) nvshmem_barrier_all();

    if (!threadIdx.x && !blockIdx.x && !mype) {
        printf("|%17.2lf|%15.2lf|%16.2lf|\n", thread_usec, warp_usec, block_usec);
    }
}

__global__ void test_sync_all_call_kern(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                        int mype, double *d_time_avg) {
    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    long long int start = 0, stop = 0;
    double thread_usec, warp_usec, block_usec, time = 0;
    int i;
    double *dest_r, *source_r;

    source_r = d_time_avg;
    dest_r = (double *)((double *)d_time_avg + 1);

    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !threadIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmem_sync_all();
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size, pWrk,
                                      pSync);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                thread_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !(threadIdx.x / warpSize)) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_sync_all_warp();
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size, pWrk,
                                      pSync);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                warp_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_sync_all_block();
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_to_all(dest_r, source_r, 1, PE_start, logPE_stride, PE_size, pWrk,
                                      pSync);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                block_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x && !threadIdx.x) nvshmem_barrier_all();

    if (!threadIdx.x && !blockIdx.x && !mype) {
        printf("|%17.2lf|%15.2lf|%16.2lf|\n", thread_usec, warp_usec, block_usec);
    }
}

int sync_calling_kernel(int PE_start, int logPE_stride, int PE_size, long *pSync,
                        cudaStream_t stream, int mype, double *d_time_avg) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;

    if (!mype) printf("Latency of thread/warp/block variants of sync and sync_all API in us\n");
    if (!mype) printf("#sync latency\n");
    if (!mype) printf("+-----------------+---------------+----------------+\n");
    if (!mype) printf("|   thread (us)   |   warp (us)   |   block (us)   |\n");
    if (!mype) printf("+-----------------+---------------+----------------+\n");

    nvshmem_barrier_all();
    test_sync_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(
        PE_start, logPE_stride, PE_size, pSync, mype, d_time_avg);
    cuda_check_error();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!mype) printf("+-----------------+---------------+----------------+\n");

    if (!mype) printf("#sync_all latency\n");
    if (!mype) printf("+-----------------+---------------+----------------+\n");
    if (!mype) printf("|   thread (us)   |   warp (us)   |   block (us)   |\n");
    if (!mype) printf("+-----------------+---------------+----------------+\n");

    nvshmem_barrier_all();
    test_sync_all_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(
        PE_start, logPE_stride, PE_size, pSync, mype, d_time_avg);
    cuda_check_error();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!mype) printf("+-----------------+---------------+----------------+\n");

    return status;
}

int main(int argc, char **argv) {
    int mype, npes;
    int i = 0;
    size_t alloc_size;
    long *buffer = NULL;
    long *d_pSync;
    long *h_pSync;
    int PE_start = 0;
    int logPE_stride = 0;
    double *d_time_avg;
    double *d_pWrk;
    cudaStream_t cstrm;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    d_time_avg = (double *)nvshmem_malloc(sizeof(double) * 2);
    d_pWrk = (double *)nvshmem_malloc(sizeof(double) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE);
    CUDA_CHECK(
        cudaMemcpyToSymbol(pWrk, (void *)&d_pWrk, sizeof(double *), 0, cudaMemcpyHostToDevice));

    alloc_size = (NVSHMEM_BARRIER_SYNC_SIZE) * sizeof(long);
    CUDA_CHECK(cudaHostAlloc(&h_pSync, alloc_size, cudaHostAllocDefault));

    buffer = (long *)nvshmem_malloc(alloc_size);
    if (!buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        goto out;
    }

    // fprintf(stderr, "pgm base = %p\n", buffer);
    d_pSync = buffer;

    for (i = 0; i < NVSHMEM_BARRIER_SYNC_SIZE; i++) {
        h_pSync[i] = NVSHMEM_SYNC_VALUE;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_pSync, h_pSync, (sizeof(long) * NVSHMEM_BARRIER_SYNC_SIZE),
                               cudaMemcpyHostToDevice, cstrm));

    sync_calling_kernel(PE_start, logPE_stride, npes, d_pSync, cstrm, mype, d_time_avg);

    CUDA_CHECK(cudaMemcpyAsync(h_pSync, d_pSync, (sizeof(long) * NVSHMEM_BARRIER_SYNC_SIZE),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_pSync));
    nvshmem_free(buffer);
    nvshmem_free(d_time_avg);
    nvshmem_free(d_pWrk);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
