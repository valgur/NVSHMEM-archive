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

DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned int, uint, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned long, ulong, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned long long, ulonglong, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(int32_t, int32, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(uint32_t, uint32, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(uint64_t, uint64, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(int, int, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(long, long, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(size_t, size, inc, (i + 1));
 
int main(int c, char *v[]) {
    int mype, npes;
    void *flag_d = NULL;
    cudaStream_t stream;

    int iter = 500;
    int skip = 50;

    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    MAIN_SETUP(c, v, mype, npes, flag_d, stream, h_size_arr, h_tables, h_lat);
    RUN_TEST_WITHOUT_ARG(unsigned int, uint, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
    RUN_TEST_WITHOUT_ARG(unsigned long, ulong, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
    RUN_TEST_WITHOUT_ARG(unsigned long long, ulonglong, inc, flag_d, mype, iter, skip, h_lat, h_size_arr,  0);
    RUN_TEST_WITHOUT_ARG(int32_t, int32, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
    RUN_TEST_WITHOUT_ARG(uint32_t, uint32, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
    RUN_TEST_WITHOUT_ARG(uint64_t, uint64, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
    RUN_TEST_WITHOUT_ARG(int, int, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
    RUN_TEST_WITHOUT_ARG(long, long, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
    RUN_TEST_WITHOUT_ARG(size_t, size, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);

    MAIN_CLEANUP(flag_d, stream, h_tables, 2);
    return 0;
}