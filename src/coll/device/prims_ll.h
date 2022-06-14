#ifndef PRIMS_LL_H
#define PRIMS_LL_H

#include "nvshmem_internal.h"
#include "nvshmemi_util.h"

#ifdef __CUDA_ARCH__

template<typename T, threadgroup_t SCOPE>
__device__ void nvshmemi_recvLL(T *dest, uint64_t *src, size_t nelems, uint32_t flag) {
    //Assumptions: sizeof(T) >= 4 bytes, num_subelems is a multiple of 2
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));

    uint32_t flag1, flag2, data1, data2;
    for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
        do {
              asm("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1),
                  "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src[i]));
        } while((flag1 != flag) || (flag2 != flag));
        //printf("received: %d %d\n", data1, data2);
        *(uint32_t *)((char *)dest + i * sizeof(uint32_t)) = data1;
        *(uint32_t *)((char *)dest + (i + 1) * sizeof(uint32_t)) = data2;
    }
}

template<typename T, threadgroup_t SCOPE>
__device__ void nvshmemi_packLL(uint64_t *dest, const T *source, size_t nelems, uint32_t ll_flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));
    for (int i = myIdx; i < num_subelems; i += groupSize) {
        size_t dst_offset = 2 * i * sizeof(uint32_t);
        size_t src_offset = i * sizeof(uint32_t);
        *(uint32_t *)((char *)dest + dst_offset) =  *(uint32_t *)((char *)source + src_offset);
        *(uint32_t *)((char *)dest + dst_offset + sizeof(uint32_t)) = ll_flag;
    }
}

/* TODO:Can use these functions for LL load and store */
/*__device__ void storeLL(void *dst, uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(dst), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
}*/

/*__device__ uint64_t readLLFinish(int offset, ncclLLFifoLine(&line)[MaxRecv], int i) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    int spins = 0;
    while (line[i].flag1 != flag || line[i].flag2 != flag) {
      asm("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(line[i].data1), "=r"(line[i].flag1), "=r"(line[i].data2), "=r"(line[i].flag2) : "l"(&src->i4));
      if (checkAbort(spins, 0)) break;
    }
    uint64_t val64 = line[i].data1 + (((uint64_t)line[i].data2) << 32);
    return val64;
}*/
#endif
#endif
