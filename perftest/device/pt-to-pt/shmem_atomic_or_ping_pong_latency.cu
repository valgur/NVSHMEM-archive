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


#include "atomic_ping_pong_common.h"

/* should get flag set to 0b1, 0b11, 0b111, etc. */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, or, (cmp >> (iter - (i + 1))), (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, or, (cmp >> (iter - (i + 1))), (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, or, (cmp >> (iter - (i + 1))), (value << i));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, or, (cmp >> (iter - (i + 1))), (value << i)); */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, or, (cmp >> (iter - (i + 1))), (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, or, (cmp >> (iter - (i + 1))), (value << i));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, or, (cmp >> (iter - (i + 1))), (value << i)); */

int main(int c, char *v[]) {
    int mype, npes;
    void *flag_d = NULL;
    cudaStream_t stream;

    /* In order to get a good latency read, we can't reset flag_d
     * so the best we can do is start at all 0 bits and flip one at
     * a time.
     */
    int iter = 64;
    int skip = 0;

    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    /* TODO: Figure out a good way to do this with signed types. The bit shifts we do don't mix with signed types. */
    MAIN_SETUP(c, v, mype, npes, flag_d, stream, h_size_arr, h_tables, h_lat);
    RUN_TEST_WITH_ARG(unsigned long, ulong, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0);
    RUN_TEST_WITH_ARG(unsigned long long, ulonglong, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0);
    RUN_TEST_WITH_ARG(uint64_t, uint64, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0);
    /* RUN_TEST_WITH_ARG(int64_t, int64, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0); */

    iter = 32;
    /* RUN_TEST_WITH_ARG(int64_t, int64, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0); */
    RUN_TEST_WITH_ARG(uint32_t, uint32, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1, 0xFFFFFFFF, 0);
    RUN_TEST_WITH_ARG(unsigned int, uint, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1, 0xFFFFFFFF, 0);

    MAIN_CLEANUP(flag_d, stream, h_tables, 2);
    return 0;
}
