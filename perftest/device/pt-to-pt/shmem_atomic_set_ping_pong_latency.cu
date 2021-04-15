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

 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int, int, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(long, long, set, i, i);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(size_t, size, set, i, i);
 
 int main(int c, char *v[]) {
     cudaStream_t stream;
 
     double *h_lat;
     uint64_t *h_size_arr;
     void *flag_d = NULL;
     void **h_tables;
 
     int iter = 500;
     int skip = 50;
     int mype, npes;
 
     MAIN_SETUP(c, v, mype, npes, flag_d, stream, h_size_arr, h_tables, h_lat);
     RUN_TEST_WITH_ARG(unsigned int, uint, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(unsigned long, ulong, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(unsigned long long, ulonglong, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(int32_t, int32, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(uint32_t, uint32, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(uint64_t, uint64, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(int, int, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(long, long, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
     RUN_TEST_WITH_ARG(size_t, size, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0, 551);
 
     MAIN_CLEANUP(flag_d, stream, h_tables, 2);
     return 0;
 }
