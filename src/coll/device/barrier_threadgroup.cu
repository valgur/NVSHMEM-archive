/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_util.h"
#include "gpu_coll.h"

extern __constant__ int barrier_tg_dissem_kval_d;

__global__ void barrier_on_stream_kernel_warp(int PE_start, int logPE_stride, int PE_size, long *pSync);
__global__ void barrier_on_stream_kernel_block(int PE_start, int logPE_stride, int PE_size, long *pSync);
__global__ void barrier_all_on_stream_kernel_warp();
__global__ void barrier_all_on_stream_kernel_block();
__global__ void sync_on_stream_kernel_warp(int PE_start, int logPE_stride, int PE_size, long *pSync);
__global__ void sync_on_stream_kernel_block(int PE_start, int logPE_stride, int PE_size, long *pSync);
__global__ void sync_all_on_stream_kernel_warp();
__global__ void sync_all_on_stream_kernel_block();


#ifdef __CUDA_ARCH__

template<int k, int logk>
__device__ static inline void sync_dissem_pow2_warp(int PE_start, int logPE_stride, int PE_size, volatile long *pSync, volatile long *counter,
                                   int is_barrier_all)
    {                                                                                   
        int my_idx_in_active_set = (nvshmemi_mype_d - PE_start) >> logPE_stride;                     
        volatile long *sync_arr = NULL;                                                    
        int shift;                                                                         
        int to_nbr_idx, to_nbr;                                                            
        int from_nbr_idx, from_nbr;                                                        
        int temp = PE_size - 1; /* used to calculate number of phases */
        int myIdx = nvshmemi_thread_id_in_warp();
        int groupSize = nvshmemi_warp_size();
        if (!is_barrier_all)                                                                
            sync_arr = (volatile long *)pSync + (counter[0] & 1) * nvshmemi_npes_d;           
        else                                                                               
            sync_arr = (volatile long *) pSync;                                            
        int pow_k = 1;                                                                     
        int phase_num = 0;                  
        while (temp) {                                            
            /* notify neighbors */                                                         
            for (int j = myIdx; j <= k - 1; j += groupSize) {                              
                shift = j << phase_num;                                                         
                if (shift >= PE_size) break;
                                              
                to_nbr_idx = my_idx_in_active_set + shift;                     
                if (to_nbr_idx >= PE_size) to_nbr_idx = to_nbr_idx - PE_size;
                to_nbr = PE_start + to_nbr_idx << logPE_stride;                                   
                                                                                           
                nvshmemx_long_signal                                         
                (((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr); 
            }                                                                              
                                                                                           
            /* wait for neighbors notification */                                          
            for (int j = myIdx; j <= k - 1; j += groupSize) {                              
                shift = j << phase_num;                                                         
                if (shift >= PE_size) break;                                              
                                                                                          
                from_nbr_idx = my_idx_in_active_set - shift;                               
                if (from_nbr_idx < 0) from_nbr_idx = PE_size + from_nbr_idx;               
                from_nbr = PE_start + from_nbr_idx << logPE_stride;                               
                                                                                           
                nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER_WARP);
                if (!is_barrier_all)                                                       
                    *(sync_arr + from_nbr) = NVSHMEM_SYNC_VALUE;       
            }                                                                              
            pow_k <<= logk;
            temp >>= logk;                                                                  
            phase_num++;
        }
        /*** NOTE: Ideallly counter[0] should be incremented here but 
        it is done by the calling function to reduce the number of times
        the threadgroup synchronization is called ***/                                                                               
    }

template __device__ void sync_dissem_pow2_warp<2, 1>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_warp<4, 2>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_warp<8, 3>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_warp<16, 4>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_warp<32, 5>(int, int, int , volatile long *, volatile long *, int);

__device__ static inline void sync_dissem_warp(int PE_start, int logPE_stride, int PE_size, volatile long *pSync, volatile long *counter,
                                   int is_barrier_all)                   
    {                                                                                   
        int stride = 1 << logPE_stride;                                                    
        int num_phases = 0;                                                                
        int k = min(barrier_tg_dissem_kval_d, PE_size); /* radix for the dissemination algorithm */               
        int my_idx_in_active_set = (nvshmemi_mype_d - PE_start) / stride;                     
        volatile long *sync_arr = NULL;                                                    
        int shift;                                                                         
        int to_nbr_idx, to_nbr;                                                            
        int from_nbr_idx, from_nbr;                                                        
        int temp = PE_size - 1; /* used to calculate number of phases */                   
        while (temp) {                                                                     
            num_phases++;                                                                  
            temp /= k;                                                                     
        }                                                                                  
        int myIdx = nvshmemi_thread_id_in_warp();
        int groupSize = nvshmemi_warp_size();
        if (!is_barrier_all)                                                                
            sync_arr = (volatile long *)pSync + (counter[0] & 1) * nvshmemi_npes_d;           
        else                                                                               
            sync_arr = (volatile long *) pSync;                                            
        int pow_k = 1;                                                                     
        for (int i = 0; i < num_phases; i++) {                                             
            /* notify neighbors */                                                         
            for (int j = myIdx; j <= k - 1; j += groupSize) {                              
                shift = j * pow_k;                                                         
                if (shift >= PE_size) break;                                               
                to_nbr_idx = (my_idx_in_active_set + shift) % PE_size;                     
                to_nbr = PE_start + to_nbr_idx * stride;                                   
                                                                                           
                nvshmemx_long_signal                                         
                (((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr); 
            }                                                                              
                                                                                           
            /* wait for neighbors notification */                                          
            for (int j = myIdx; j <= k - 1; j += groupSize) {                              
                shift = j * pow_k;                                                         
                if (shift >= PE_size) break;                                              
                                                                                          
                from_nbr_idx = my_idx_in_active_set - shift;                               
                if (from_nbr_idx < 0) from_nbr_idx = PE_size + from_nbr_idx;               
                from_nbr = PE_start + from_nbr_idx * stride;                               
                                                                                           
                nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER_WARP);
                if (!is_barrier_all)                                                       
                    *(sync_arr + from_nbr) = NVSHMEM_SYNC_VALUE;       
            }                                                                              
            pow_k *= k;                                                                    
        }                                                                                  
        /*** NOTE: Ideallly counter[0] should be incremented here but 
        it is done by the calling function to reduce the number of times
        the threadgroup synchronization is called ***/                                                                               
    }

template<int k, int logk>
__device__ static inline void sync_dissem_pow2_block(int PE_start, int logPE_stride, int PE_size,
                                            volatile long *pSync, volatile long *counter, int is_barrier_all) {
    int my_idx_in_active_set = (nvshmemi_mype_d - PE_start) >> logPE_stride;
    volatile long *sync_arr = NULL;
    int shift;
    int to_nbr_idx, to_nbr;
    int from_nbr_idx, from_nbr;
    int temp = PE_size - 1; /* used to calculate number of phases */
    int myIdx = nvshmemi_thread_id_in_block();
    int groupSize = nvshmemi_block_size();
    if (!is_barrier_all)
        sync_arr = (volatile long *)pSync + (counter[0] & 1) * nvshmemi_npes_d;
    else
        sync_arr = (volatile long *)pSync;
    int pow_k = 1;
    int phase_num = 0;
    while (temp) {
        /* notify neighbors */
        for (int j = myIdx; j <= k - 1; j += groupSize) {
            shift = j << phase_num;
            if (shift >= PE_size) break;
            to_nbr_idx = my_idx_in_active_set + shift;
            if (to_nbr_idx >= PE_size) to_nbr_idx = to_nbr_idx - PE_size;
            to_nbr = PE_start + to_nbr_idx << logPE_stride;

            nvshmemx_long_signal(((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr);
        }

        /* wait for neighbors notification */
        for (int j = myIdx; j <= k - 1; j += groupSize) {
            shift = j << phase_num;
            if (shift >= PE_size) break;

            from_nbr_idx = my_idx_in_active_set - shift;
            if (from_nbr_idx < 0) from_nbr_idx = PE_size + from_nbr_idx;
            from_nbr = PE_start + from_nbr_idx << logPE_stride;

            nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER_THREADBLOCK);
            if (!is_barrier_all) *(sync_arr + from_nbr) = NVSHMEM_SYNC_VALUE;
        }
        pow_k <<= logk;
        temp >>= logk;
        phase_num++;
    }
    /*** NOTE: Ideallly counter[0] should be incremented here but 
    it is done by the calling function to reduce the number of times
    the threadgroup synchronization is called ***/                                                                               
}
template __device__ void sync_dissem_pow2_block<2, 1>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_block<4, 2>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_block<8, 3>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_block<16, 4>(int, int, int , volatile long *, volatile long *, int);
template __device__ void sync_dissem_pow2_block<32, 5>(int, int, int , volatile long *, volatile long *, int);


__device__ static inline void sync_dissem_block(int PE_start, int logPE_stride, int PE_size,
                                           volatile long *pSync, volatile long *counter, int is_barrier_all) {
    int stride = 1 << logPE_stride;
    int num_phases = 0;
    int k = min(barrier_tg_dissem_kval_d, PE_size); /* radix for the dissemination algorithm */
    int my_idx_in_active_set = (nvshmemi_mype_d - PE_start) / stride;
    volatile long *sync_arr = NULL;
    int shift;
    int to_nbr_idx, to_nbr;
    int from_nbr_idx, from_nbr;
    int temp = PE_size - 1; /* used to calculate number of phases */
    while (temp) {
        num_phases++;
        temp /= k;
    }
    int myIdx = nvshmemi_thread_id_in_block();
    int groupSize = nvshmemi_block_size();
    if (!is_barrier_all)
        sync_arr = (volatile long *)pSync + (counter[0] & 1) * nvshmemi_npes_d;
    else
        sync_arr = (volatile long *)pSync;
    int pow_k = 1;
    for (int i = 0; i < num_phases; i++) {
        /* notify neighbors */
        for (int j = myIdx; j <= k - 1; j += groupSize) {
            shift = j * pow_k;
            if (shift >= PE_size) break;
            to_nbr_idx = (my_idx_in_active_set + shift) % PE_size;
            to_nbr = PE_start + to_nbr_idx * stride;

            nvshmemx_long_signal(((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr);
        }

        /* wait for neighbors notification */
        for (int j = myIdx; j <= k - 1; j += groupSize) {
            shift = j * pow_k;
            if (shift >= PE_size) break;

            from_nbr_idx = my_idx_in_active_set - shift;
            if (from_nbr_idx < 0) from_nbr_idx = PE_size + from_nbr_idx;
            from_nbr = PE_start + from_nbr_idx * stride;

            nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER_THREADBLOCK);
            if (!is_barrier_all) *(sync_arr + from_nbr) = NVSHMEM_SYNC_VALUE;
        }
        pow_k *= k;
    }
    /*** NOTE: Ideallly counter[0] should be incremented here but 
    it is done by the calling function to reduce the number of times
    the threadgroup synchronization is called ***/                                                                               
}

#define NVSHMEMI_BARRIER_THREADGROUP_ALGO(SCOPE)                                                                      \
    __device__ inline void nvshmemi_sync_algo_##SCOPE(int PE_start, int logPE_stride, int PE_size, volatile long *pSync,    \
                                                           volatile long *counter, bool is_barrier_all) {             \
        int k = min(barrier_tg_dissem_kval_d, PE_size);                                                               \
        k = max(k, 2);                                                                                                \
        switch(k) {                                                                                                   \
            case 2:                                                                                                   \
                sync_dissem_pow2_##SCOPE<2, 1>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);      \
                break;                                                                                                \
            case 4:                                                                                                   \
                sync_dissem_pow2_##SCOPE<4, 2>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);      \
                break;                                                                                                \
            case 8:                                                                                                   \
                sync_dissem_pow2_##SCOPE<8, 3>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);      \
                break;                                                                                                \
            case 16:                                                                                                  \
                sync_dissem_pow2_##SCOPE<16, 4>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);     \
                break;                                                                                                \
            case 32:                                                                                                  \
                sync_dissem_pow2_##SCOPE<32, 5>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);     \
                break;                                                                                                \
            default:                                                                                                  \
                sync_dissem_##SCOPE(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);                 \
                break;                                                                                                \
        }                                                                                                             \
    }
NVSHMEMI_BARRIER_THREADGROUP_ALGO(warp)
NVSHMEMI_BARRIER_THREADGROUP_ALGO(block) 


__device__ void nvshmemx_barrier_warp(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int myIdx = nvshmemi_thread_id_in_warp();

    __syncwarp(); 
    if (!myIdx) nvshmem_quiet();                           
    __syncwarp(); 

    nvshmemi_sync_algo_warp(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);

    __syncwarp();
    if (!myIdx) {
        gpu_icounter_barrier_d[0] += 1; /* Ideally this should be part of the algo, but doing
                                            it here to optimize syncwarp calls */
                                          
        if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)
            nvshmemi_proxy_enforce_consistency_at_target_no_membar();
    }                                                                            
    __syncwarp(); 
}


__device__ void nvshmemx_barrier_block(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int myidx = nvshmemi_thread_id_in_block();

    __syncthreads(); 
    if (!myidx) nvshmem_quiet();                           
    __syncthreads(); 

    nvshmemi_sync_algo_block(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);

    __syncthreads();
    if (!myidx) {
        gpu_icounter_barrier_d[0] += 1;/* Ideally this should be part of the algo, but doing
                                           it here to optimize syncthreads calls */
        if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)
            nvshmemi_proxy_enforce_consistency_at_target_no_membar();
    }                                                                                  
    __syncthreads(); 
}


__device__ void nvshmemx_barrier_all_warp() {
    int myIdx = nvshmemi_thread_id_in_warp();

    __syncwarp(); 
    if (!myIdx) nvshmem_quiet();                           
    __syncwarp(); 

    nvshmemi_sync_algo_warp(0, 0, nvshmemi_npes_d, (long *)gpu_ipsync_d, gpu_icounter_d, 1);

    __syncwarp();
    if (!myIdx) {
        gpu_icounter_d[0] += 1;/* Ideally this should be part of the algo, but doing
                                   it here to optimize syncwarp calls */
        if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)
            nvshmemi_proxy_enforce_consistency_at_target_no_membar();
    }                                                                                  
    __syncwarp(); 
}


__device__ void nvshmemx_barrier_all_block() {
    int myidx = nvshmemi_thread_id_in_block();

    __syncthreads(); 
    if (!myidx) nvshmem_quiet();                           
    __syncthreads(); 

    nvshmemi_sync_algo_block(0, 0, nvshmemi_npes_d, (long *)gpu_ipsync_d, gpu_icounter_d, 1);

    __syncthreads();
    if (!myidx) {
        gpu_icounter_d[0] += 1;/* Ideally this should be part of the algo, but doing
                                   it here to optimize syncthreads calls */
        if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)
            nvshmemi_proxy_enforce_consistency_at_target_no_membar();
    }                                                                                  
    __syncthreads(); 
}
__device__ void nvshmemx_sync_warp(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int myidx = nvshmemi_thread_id_in_warp();

    __syncwarp(); 

    nvshmemi_sync_algo_warp(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);

    __syncwarp();
    if (!myidx)
        gpu_icounter_barrier_d[0] += 1;
    __syncwarp();
}


__device__ void nvshmemx_sync_block(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int myidx = nvshmemi_thread_id_in_block();

    __syncthreads(); 

    nvshmemi_sync_algo_block(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);

    __syncthreads(); 
    if (!myidx)
        gpu_icounter_barrier_d[0] += 1;
    __syncthreads();
}


__device__ void nvshmemx_sync_all_warp() {
    int myidx = nvshmemi_thread_id_in_warp();

    __syncwarp();

    nvshmemi_sync_algo_warp(0, 0, nvshmemi_npes_d, (long *)gpu_ipsync_d, gpu_icounter_d, 1);

    __syncwarp(); 
    if (!myidx)
        gpu_icounter_d[0] += 1;
    __syncwarp();
}


__device__ void nvshmemx_sync_all_block() {
    int myidx = nvshmemi_thread_id_in_block();
    
    __syncthreads();

    nvshmemi_sync_algo_block(0, 0, nvshmemi_npes_d, (long *)gpu_ipsync_d, gpu_icounter_d, 1);

    __syncthreads(); 
    if (!myidx)
        gpu_icounter_d[0] += 1;
    __syncthreads();
}


#define BARRIER_ON_STREAM_KERNEL(SCOPE)                                                                           \
    __global__ void barrier_on_stream_kernel_##SCOPE (int PE_start, int logPE_stride, int PE_size, long *pSync) { \
        int myidx = nvshmemi_thread_id_in_##SCOPE();                                                              \
                                                                                                                  \
        if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY) { 					          \
            if (!myidx)                                                                                           \
                nvshmemi_proxy_quiet_no_membar();                                                                 \
            NVSHMEMI_SYNC_##SCOPE();                                                                              \
        }                                                                                                         \
                                                                                                                  \
        nvshmemi_sync_algo_##SCOPE(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);            \
                                                                                                                  \
        NVSHMEMI_SYNC_##SCOPE();                                                                                  \
        if (!myidx) {                                                                                             \
            gpu_icounter_barrier_d[0] += 1;                                                                       \
            if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)						  \
                nvshmemi_proxy_enforce_consistency_at_target_no_membar();                                         \
        }                                                                                                         \
    }

BARRIER_ON_STREAM_KERNEL(warp);
BARRIER_ON_STREAM_KERNEL(block);

#define BARRIER_ALL_ON_STREAM_KERNEL(SCOPE)                                                     \
    __global__ void barrier_all_on_stream_kernel_##SCOPE () {                                   \
        int myidx = nvshmemi_thread_id_in_##SCOPE();                                            \
                                                                                                \
        if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY) {				\
            if (!myidx)                                                                         \
                nvshmemi_proxy_quiet_no_membar();                                               \
            NVSHMEMI_SYNC_##SCOPE();                                                            \
        }                                                                                       \
        /*printf("running barrier algo\n");   */                                                    \
        nvshmemi_sync_algo_##SCOPE(0, 0, nvshmemi_npes_d, gpu_ipsync_d, gpu_icounter_d, 1);     \
       /* printf("returning barrier algo\n");   */                                                  \
                                                                                                \
        NVSHMEMI_SYNC_##SCOPE();                                                                \
        if (!myidx) {                                                                           \
            gpu_icounter_d[0] += 1;                                                             \
            if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)				\
                nvshmemi_proxy_enforce_consistency_at_target_no_membar();                       \
        }                                                                                       \
    }

BARRIER_ALL_ON_STREAM_KERNEL(warp)
BARRIER_ALL_ON_STREAM_KERNEL(block)


#define SYNC_ON_STREAM_KERNEL(SCOPE)                                                                                \
    __global__ void sync_on_stream_kernel_##SCOPE(int PE_start, int logPE_stride, int PE_size, long *pSync) {       \
        int myidx = nvshmemi_thread_id_in_##SCOPE();                                                                \
        nvshmemi_sync_algo_##SCOPE(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);              \
        NVSHMEMI_SYNC_##SCOPE();                                                                                    \
        if (!myidx)                                                                                                 \
            gpu_icounter_barrier_d[0] += 1;                                                                         \
    }

SYNC_ON_STREAM_KERNEL(warp)
SYNC_ON_STREAM_KERNEL(block)

#define SYNC_ALL_ON_STREAM_KERNEL(SCOPE)                                                                            \
    __global__ void sync_all_on_stream_kernel_##SCOPE() {                                                           \
        int myidx = nvshmemi_thread_id_in_##SCOPE();                                                                \
        nvshmemi_sync_algo_##SCOPE(0, 0, nvshmemi_npes_d, gpu_ipsync_d, gpu_icounter_d, 1);                         \
        NVSHMEMI_SYNC_##SCOPE();                                                                                    \
        if (!myidx)                                                                                                 \
            gpu_icounter_d[0] += 1;                                                                                 \
    }

SYNC_ALL_ON_STREAM_KERNEL(warp)
SYNC_ALL_ON_STREAM_KERNEL(block)

#endif


extern "C" int call_barrier_on_stream_kern(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                           cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        num_threads_per_block = PE_size - 1; // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    if (num_threads_per_block <= 32) {
        barrier_on_stream_kernel_warp<<<num_blocks, 32, 0, stream>>>(
            PE_start, logPE_stride, PE_size, pSync);
    }
    else {
        barrier_on_stream_kernel_block<<<num_blocks, num_threads_per_block, 0, stream>>>(
            PE_start, logPE_stride, PE_size, pSync);
    }

    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int call_barrier_all_on_stream_kern(cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        num_threads_per_block = nvshmem_state->npes - 1; // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }
    
    if (num_threads_per_block <= 32) {
        TRACE(NVSHMEM_COLL, "In call_barrier_all_on_stream_kern - launching barrier_all_on_stream_kernel_warp with %d threads", num_threads_per_block);
        barrier_all_on_stream_kernel_warp<<<num_blocks, 32, 0, stream>>>();
    }
    else {
        TRACE(NVSHMEM_COLL, "In call_barrier_all_on_stream_kern - launching barrier_all_on_stream_kernel_block with %d threads", num_threads_per_block);
        barrier_all_on_stream_kernel_block<<<num_blocks, num_threads_per_block, 0, stream>>>();
    }
        
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}


extern "C" int call_sync_on_stream_kern(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                        cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        num_threads_per_block = PE_size - 1; // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    if (num_threads_per_block <= 32) {
        sync_on_stream_kernel_warp<<<num_blocks, 32, 0, stream>>>(PE_start, logPE_stride,
                                                                  PE_size, pSync);
    } else {
        sync_on_stream_kernel_block<<<num_blocks, num_threads_per_block, 0, stream>>>(PE_start, logPE_stride,
                                                                                      PE_size, pSync);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int call_sync_all_on_stream_kern(cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        num_threads_per_block = nvshmem_state->npes - 1; // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }
    if (num_threads_per_block <= 32) {
        sync_all_on_stream_kernel_warp<<<num_blocks, 32, 0, stream>>>();
    } else {
        sync_all_on_stream_kernel_block<<<num_blocks, num_threads_per_block, 0, stream>>>();
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}
