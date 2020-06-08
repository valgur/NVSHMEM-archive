#ifndef NVSHMEM_COLL_SHORTHAND_H
#define NVSHMEM_COLL_SHORTHAND_H

#define PSYN long *pSync
#define PR int PE_root
#define PS int PE_start
#define PL int logPE_stride
#define PZ int PE_size
#define NR int nreduce
#define NE size_t nelems
#define CS cudaStream_t stream
#define SC int scope
#define VD void *dest
#define VS const void *source

#define SRC_DST(TYPE) TYPE *dest, const TYPE *source

#define SRC_DST_R(TYPE, TYPE2) TYPE TYPE2 *dest, const TYPE TYPE2 *source

#ifdef NVSHMEM_COMPLEX_SUPPORT

#define SRC_DST_C(TYPE) TYPE complex *dest, const TYPE complex *source

#define PWRK_C(TYPE) TYPE complex *pWrk

#endif

#define PWRK(TYPE) TYPE *pWrk

#define PWRK_R(TYPE, TYPE2) TYPE TYPE2 *pWrk

#endif
