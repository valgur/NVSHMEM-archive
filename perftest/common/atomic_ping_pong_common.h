/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _ATOMIC_PING_PONG_COMMON_H_
#define _ATOMIC_PING_PONG_COMMON_H_

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#define DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR)               \
__global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter, int skip,        \
                                              double *lat_result) {                            \
    long long int start, stop;                                                                 \
    double time;                                                                               \
    int i, tid, peer;                                                                          \
                                                                                               \
    assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );   \
    peer = !pe;                                                                                \
    tid = threadIdx.x;                                                                         \
                                                                                               \
    for (i = 0; i < (iter + skip); i++) {                                                      \
        if (i == skip) start = clock64();                                                      \
        if (pe) {                                                                              \
            nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);            \
            nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, peer);                                  \
        } else {                                                                               \
            nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, peer);                                  \
            nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);            \
        }                                                                                      \
    }                                                                                          \
    stop = clock64();                                                                          \
    nvshmem_quiet();                                                                           \
                                                                                               \
    if ((pe == 0) && !tid) {                                                                   \
        time = (stop - start) / iter;                                                          \
        *lat_result = time * 1000 / clockrate;                                                 \
    }                                                                                          \
}

#define DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR, SET_EXPR)    \
__global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter, int skip,        \
                                              double *lat_result, TYPE value, TYPE cmp) {      \
    long long int start, stop;                                                                 \
    double time;                                                                               \
    int i, tid, peer;                                                                          \
                                                                                               \
    assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );   \
    peer = !pe;                                                                                \
    tid = threadIdx.x;                                                                         \
                                                                                               \
    for (i = 0; i < (iter + skip); i++) {                                                      \
        if (i == skip) start = clock64();                                                      \
                                                                                               \
        if (pe) {                                                                              \
            nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);            \
            nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, SET_EXPR, peer);                        \
        } else {                                                                               \
            nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, SET_EXPR, peer);                        \
            nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);            \
        }                                                                                      \
    }                                                                                          \
    stop = clock64();                                                                          \
    nvshmem_quiet();                                                                           \
                                                                                               \
    if ((pe == 0) && !tid) {                                                                   \
        time = (stop - start) / iter;                                                          \
        *lat_result = time * 1000 / clockrate;                                                 \
    }                                                                                          \
}

#define DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR, SET_EXPR)    \
__global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter, int skip,        \
                                              double *lat_result, TYPE value, TYPE cmp) {      \
    long long int start, stop;                                                                 \
    double time;                                                                               \
    int i, tid, peer;                                                                          \
                                                                                               \
    assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );   \
    peer = !pe;                                                                                \
    tid = threadIdx.x;                                                                         \
                                                                                               \
    for (i = 0; i < (iter + skip); i++) {                                                      \
        if (i == skip) start = clock64();                                                      \
                                                                                               \
        if (pe) {                                                                              \
            nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, SET_EXPR);                \
            nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, COMPARE_EXPR, SET_EXPR, peer);          \
        } else {                                                                               \
            nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, COMPARE_EXPR, SET_EXPR, peer);          \
            nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, SET_EXPR);                \
        }                                                                                      \
    }                                                                                          \
    stop = clock64();                                                                          \
    nvshmem_quiet();                                                                           \
                                                                                               \
    if ((pe == 0) && !tid) {                                                                   \
        time = (stop - start) / iter;                                                          \
        *lat_result = time * 1000 / clockrate;                                                 \
    }                                                                                          \
}

#define MAIN_SETUP(c, v, mype, npes, flag_d, stream, h_size_arr,                               \
                   h_tables, h_lat)                                                            \
do {                                                                                           \
                                                                                               \
    init_wrapper(&c, &v);                                                                      \
                                                                                               \
    mype = nvshmem_my_pe();                                                                    \
    npes = nvshmem_n_pes();                                                                    \
                                                                                               \
    if (npes != 2) {                                                                           \
        fprintf(stderr, "This test requires exactly two processes \n");                        \
        finalize_wrapper();                                                                    \
        exit(-1);                                                                              \
    }                                                                                          \
                                                                                               \
    alloc_tables(&h_tables, 2, 1);                                                             \
    h_size_arr = (uint64_t *)h_tables[0];                                                      \
    h_lat = (double *)h_tables[1];                                                             \
                                                                                               \
    flag_d = nvshmem_malloc(sizeof(uint64_t));                                                 \
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));                                       \
                                                                                               \
                                                                                               \
    CUDA_CHECK(cudaStreamCreate(&stream));                                                     \
                                                                                               \
    nvshmem_barrier_all();                                                                     \
                                                                                               \
    CUDA_CHECK(cudaDeviceSynchronize());                                                       \
                                                                                               \
    if (mype == 0) {                                                                           \
        printf("Note: This test measures full round-trip latency\n");                          \
    }                                                                                          \
} while(0)

#define MAIN_LOOP_NO_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip,                       \
                         h_lat, h_size_arr)                                                    \
do {                                                                                           \
    int size = sizeof(TYPE);                                                                   \
    double *cur_lat;                                                                           \
                                                                                               \
    int status = 0;                                                                            \
    h_size_arr[0] = size;                                                                      \
    cur_lat = &h_lat[0];                                                                       \
    void *args[] = {&flag_d, &mype, &iter, &skip, &cur_lat};                                   \
                                                                                               \
    CUDA_CHECK(cudaDeviceSynchronize());                                                       \
    nvshmem_barrier_all();                                                                     \
                                                                                               \
    status = nvshmemx_collective_launch((const void *)ping_pong_##TYPE_NAME##_##AMO,           \
                                        1, 1, args, 0, stream);                                \
    if (status != NVSHMEMX_SUCCESS) {                                                          \
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);                      \
        exit(-1);                                                                              \
    }                                                                                          \
                                                                                               \
    cudaStreamSynchronize(stream);                                                             \
                                                                                               \
    nvshmem_barrier_all();                                                                     \
                                                                                               \
    if (mype == 0) {                                                                           \
        print_table("shmem_at_" #TYPE "_" #AMO "_ping_lat", "None", "size (Bytes)",            \
        "latency", "us", '-', h_size_arr, h_lat, 1);                                           \
    }                                                                                          \
                                                                                               \
    CUDA_CHECK(cudaDeviceSynchronize());                                                       \
                                                                                               \
} while (0)

#define MAIN_LOOP_WITH_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip, h_lat,              \
                            h_size_arr, value, compare)                                        \
do {                                                                                           \
    int size = sizeof(TYPE);                                                                   \
    double *cur_lat;                                                                           \
                                                                                               \
    int status = 0;                                                                            \
    h_size_arr[0] = size;                                                                      \
    cur_lat = &h_lat[0];                                                                       \
    void *args[] = {&flag_d, &mype, &iter, &skip, &cur_lat, &value, &compare};                 \
                                                                                               \
    CUDA_CHECK(cudaDeviceSynchronize());                                                       \
    nvshmem_barrier_all();                                                                     \
                                                                                               \
    status = nvshmemx_collective_launch((const void *)ping_pong_##TYPE_NAME##_##AMO,           \
                                        1, 1, args, 0, stream);                                \
    if (status != NVSHMEMX_SUCCESS) {                                                          \
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);                      \
        exit(-1);                                                                              \
    }                                                                                          \
                                                                                               \
    cudaStreamSynchronize(stream);                                                             \
                                                                                               \
    nvshmem_barrier_all();                                                                     \
                                                                                               \
    if (mype == 0) {                                                                           \
        print_table("shmem_at_" #TYPE "_" #AMO "_ping_lat", "None", "size (Bytes)",            \
        "latency", "us", '-', h_size_arr, h_lat, 1);                                           \
    }                                                                                          \
                                                                                               \
    CUDA_CHECK(cudaDeviceSynchronize());                                                       \
                                                                                               \
} while (0)

#define RUN_TEST_WITH_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip,                      \
                          h_lat, h_size_arr, val, cmp, flag_init)                              \
do {                                                                                           \
    TYPE compare, value, flag_init_var;                                                        \
                                                                                               \
    compare = cmp;                                                                             \
    value = val;                                                                               \
    flag_init_var = flag_init;                                                                 \
    CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));      \
    MAIN_LOOP_WITH_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip,                         \
                       h_lat, h_size_arr, value, compare);                                     \
} while(0);

#define RUN_TEST_WITHOUT_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip,                   \
                             h_lat, h_size_arr, flag_init)                                     \
do {                                                                                           \
    TYPE flag_init_var = flag_init;                                                            \
    CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));      \
    MAIN_LOOP_NO_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip,                           \
                     h_lat, h_size_arr);                                                       \
} while(0);

#define MAIN_CLEANUP(flag_d, stream, h_tables, num_entries)                                    \
do {                                                                                           \
     if (flag_d) nvshmem_free(flag_d);                                                         \
     cudaStreamDestroy(stream);                                                                \
     free_tables(h_tables, 2);                                                                 \
     finalize_wrapper();                                                                       \
} while(0);

#endif /* _ATOMIC_PING_PONG_COMMON_H_ */
