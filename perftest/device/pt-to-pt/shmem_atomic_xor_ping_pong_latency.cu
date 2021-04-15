/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure xor
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */


 #include "atomic_ping_pong_common.h"

 /* alternate between 1 and 0 */
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, xor, i % 2, 1);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, xor, i % 2, 1);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, xor, i % 2, 1);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, xor, i % 2, 1);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, xor, i % 2, 1);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, xor, i % 2, 1);
 DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, xor, i % 2, 1);
 
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
     RUN_TEST_WITH_ARG(unsigned long, ulong, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
     RUN_TEST_WITH_ARG(unsigned long long, ulonglong, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
     RUN_TEST_WITH_ARG(uint64_t, uint64, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
     RUN_TEST_WITH_ARG(int64_t, int64, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
     RUN_TEST_WITH_ARG(int64_t, int64, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
     RUN_TEST_WITH_ARG(uint32_t, uint32, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
     RUN_TEST_WITH_ARG(unsigned int, uint, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
 
     MAIN_CLEANUP(flag_d, stream, h_tables, 2);
     return 0;
 }