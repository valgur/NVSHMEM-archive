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

#ifndef COLL_TEST_H
#define COLL_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

#define MAX_SKIP 16
#define MAX_ITERS 128
#define MAX_NPES 128
#define BARRIER_MAX_ITERS 1000
#define BARRIER_MAX_SKIP 10

#define alltoall_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define alltoall_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define collect_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define collect_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define broadcast_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define broadcast_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define call_shmem_broadcast(BITS, d_dest, d_source, num_elems, root, PE_start, logPE_stride,    \
                             npes, d_pSync)                                                      \
    do {                                                                                         \
        nvshmem_broadcast##BITS(d_dest, d_source, num_elems, root, PE_start, logPE_stride, npes, \
                                d_pSync);                                                        \
    } while (0)

#define call_shmem_collect(BITS, d_dest, d_source, num_elems, root, PE_start, logPE_stride, npes,  \
                           d_pSync)                                                                \
    do {                                                                                           \
        nvshmem_collect##BITS(d_dest, d_source, num_elems, PE_start, logPE_stride, npes, d_pSync); \
    } while (0)

#define call_shmem_alltoall(BITS, d_dest, d_source, num_elems, root, PE_start, logPE_stride, npes, \
                            d_pSync)                                                               \
    do {                                                                                           \
        nvshmem_alltoall##BITS(d_dest, d_source, num_elems, PE_start, logPE_stride, npes,          \
                               d_pSync);                                                           \
    } while (0)

#define call_shmem_broadcast_on_stream(BITS, d_dest, d_source, num_elems, root, PE_start, \
                                       logPE_stride, npes, d_pSync, stream)               \
    do {                                                                                  \
        nvshmemx_broadcast##BITS##_on_stream(d_dest, d_source, num_elems, root, PE_start, \
                                             logPE_stride, npes, d_pSync, stream);        \
    } while (0)

#define call_shmem_collect_on_stream(BITS, d_dest, d_source, num_elems, root, PE_start,         \
                                     logPE_stride, npes, d_pSync, stream)                       \
    do {                                                                                        \
        nvshmemx_collect##BITS##_on_stream(d_dest, d_source, num_elems, PE_start, logPE_stride, \
                                           npes, d_pSync, stream);                              \
    } while (0)

#define call_shmem_alltoall_on_stream(BITS, d_dest, d_source, num_elems, root, PE_start,         \
                                      logPE_stride, npes, d_pSync, stream)                       \
    do {                                                                                         \
        nvshmemx_alltoall##BITS##_on_stream(d_dest, d_source, num_elems, PE_start, logPE_stride, \
                                            npes, d_pSync, stream);                              \
    } while (0)

#define RUN_COLL(coll, COLL, DATATYPE, BITS, d_source, h_source, d_dest, h_dest, d_pSync, h_pSync, \
                 npes, root, stream)                                                               \
    do {                                                                                           \
        for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {                         \
            int iters = 0;                                                                         \
            double latency = 0;                                                                    \
            int skip = MAX_SKIP;                                                                   \
            struct timeval t_start, t_stop;                                                        \
            int PE_start = 0;                                                                      \
            int logPE_stride = 0;                                                                  \
                                                                                                   \
            for (iters = 0; iters < MAX_ITERS + skip; iters++) {                                   \
                CUDA_CHECK(cudaMemcpyAsync(d_source, h_source,                                     \
                                           coll##_src_size(DATATYPE, num_elems, npes),             \
                                           cudaMemcpyHostToDevice, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest,                                         \
                                           coll##_dest_size(DATATYPE, num_elems, npes),            \
                                           cudaMemcpyHostToDevice, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(d_pSync, h_pSync,                                       \
                                           (sizeof(long) * NVSHMEM_##COLL##_SYNC_SIZE),            \
                                           cudaMemcpyHostToDevice, stream));                       \
                                                                                                   \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                         \
                                                                                                   \
                nvshmem_barrier_all();                                                             \
                                                                                                   \
                if (iters >= skip) gettimeofday(&t_start, NULL);                                   \
                                                                                                   \
                call_shmem_##coll(BITS, d_dest, d_source, num_elems, root, PE_start, logPE_stride, \
                                  npes, d_pSync);                                                  \
                                                                                                   \
                if (iters >= skip) {                                                               \
                    gettimeofday(&t_stop, NULL);                                                   \
                    latency += ((t_stop.tv_usec - t_start.tv_usec) +                               \
                                (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));                        \
                }                                                                                  \
                                                                                                   \
                CUDA_CHECK(cudaMemcpyAsync(h_source, d_source,                                     \
                                           coll##_src_size(DATATYPE, num_elems, npes),             \
                                           cudaMemcpyDeviceToHost, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest,                                         \
                                           coll##_dest_size(DATATYPE, num_elems, npes),            \
                                           cudaMemcpyDeviceToHost, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(h_pSync, d_pSync,                                       \
                                           (sizeof(long) * NVSHMEM_##COLL##_SYNC_SIZE),            \
                                           cudaMemcpyDeviceToHost, stream));                       \
                                                                                                   \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                         \
            }                                                                                      \
                                                                                                   \
            nvshmem_barrier_all();                                                                 \
                                                                                                   \
            if (!mype)                                                                             \
                printf("|%14.0lu|%18.2lf|\n", num_elems * sizeof(DATATYPE), (latency / MAX_ITERS)); \
        }                                                                                          \
    } while (0)

#define RUN_COLL_ON_STREAM(coll, COLL, DATATYPE, BITS, d_source, h_source, d_dest, h_dest,         \
                           d_pSync, h_pSync, npes, root, stream)                                   \
    do {                                                                                           \
        for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {                         \
            int iters = 0;                                                                         \
            double latency = 0;                                                                    \
            int skip = MAX_SKIP;                                                                   \
            struct timeval t_start, t_stop;                                                        \
            int PE_start = 0;                                                                      \
            int logPE_stride = 0;                                                                  \
                                                                                                   \
            for (iters = 0; iters < MAX_ITERS + skip; iters++) {                                   \
                CUDA_CHECK(cudaMemcpyAsync(d_source, h_source,                                     \
                                           coll##_src_size(DATATYPE, num_elems, npes),             \
                                           cudaMemcpyHostToDevice, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest,                                         \
                                           coll##_dest_size(DATATYPE, num_elems, npes),            \
                                           cudaMemcpyHostToDevice, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(d_pSync, h_pSync,                                       \
                                           (sizeof(long) * NVSHMEM_##COLL##_SYNC_SIZE),            \
                                           cudaMemcpyHostToDevice, stream));                       \
                                                                                                   \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                         \
                nvshmem_barrier_all();                                                             \
                                                                                                   \
                if (iters >= skip) gettimeofday(&t_start, NULL);                                   \
                                                                                                   \
                call_shmem_##coll##_on_stream(BITS, d_dest, d_source, num_elems, root, PE_start,   \
                                              logPE_stride, npes, d_pSync, stream);                \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                         \
                                                                                                   \
                if (iters >= skip) {                                                               \
                    gettimeofday(&t_stop, NULL);                                                   \
                    latency += ((t_stop.tv_usec - t_start.tv_usec) +                               \
                                (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));                        \
                }                                                                                  \
                                                                                                   \
                CUDA_CHECK(cudaMemcpyAsync(h_source, d_source,                                     \
                                           coll##_src_size(DATATYPE, num_elems, npes),             \
                                           cudaMemcpyDeviceToHost, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest,                                         \
                                           coll##_dest_size(DATATYPE, num_elems, npes),            \
                                           cudaMemcpyDeviceToHost, stream));                       \
                CUDA_CHECK(cudaMemcpyAsync(h_pSync, d_pSync,                                       \
                                           (sizeof(long) * NVSHMEM_##COLL##_SYNC_SIZE),            \
                                           cudaMemcpyDeviceToHost, stream));                       \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                         \
            }                                                                                      \
                                                                                                   \
            nvshmem_barrier_all();                                                                 \
                                                                                                   \
            if (!mype)                                                                             \
                printf("|%14.0lu|%18.2lf|\n", num_elems * sizeof(DATATYPE), (latency / MAX_ITERS)); \
        }                                                                                          \
    } while (0)

#endif /*COLL_TEST_H*/
