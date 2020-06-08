#ifndef NVSHMEMI_RDXN_COMMON_CPU_H
#define NVSHMEMI_RDXN_COMMON_CPU_H 1
#include <cuda.h>
#include <cuda_runtime.h>
typedef enum rdxn_op {
    rd_and = 0,
    rd_max,
    rd_min,
    rd_sum,
    rd_prod,
    rd_or,
    rd_xor,
    rd_op_null
} rdxn_op_t;

typedef enum rdxn_op_dt {
    rd_dt_short = 0,
    rd_dt_int,
    rd_dt_long,
    rd_dt_float,
    rd_dt_double,
    rd_dt_long_long,
    rd_dt_long_double,
    rd_dt_float_complex,
    rd_dt_double_complex,
    rd_dt_null
} rdxn_op_dt_t;

typedef struct rdxn_opr {
    void *dest;
    const void *source;
    int nreduce;
    int PE_start;
    int logPE_stride;
    int PE_size;
    void *pWrk;
    long *pSync;
    int op_size;
    rdxn_op_t op_type;
    rdxn_op_dt_t op_dt_type;
    cudaStream_t stream;
} rdxn_opr_t;

typedef void (*rdxn_fxn_ptr_t)(void *x, void *y, void *z, int nelems);
typedef void (*rdxn_comb_fxn_ptr_t)(int src_offset, int dest_offset, rdxn_opr_t rdx_op);

#if __cplusplus
extern "C" {
#endif
void nvshmemi_simple_kernel();
int nvshmemi_rdxn_fxn_ptrs_init();
int nvshmemi_rdxn_cpu_op_kernel(void *x, void *y, void *z, rdxn_opr_t *rdx_op);
int nvshmemxi_rdxn_cpu_op_kernel(void *x, void *y, void *z, rdxn_opr_t *rdx_op);
int nvshmemi_rdxn_cpu_op_comb_kernel(int src_offset, int dest_offset, rdxn_opr_t *rdx_op);

#define CALL_RDXN_ON_STREAM_KERN(TYPE, OP)                                                        \
    extern "C" void call_rdxn_##TYPE##_##OP##_on_stream_kern(                                     \
        TYPE *dest, const TYPE *source, int nreduce, int PE_start, int logPE_stride, int PE_size, \
        TYPE *pWrk, long *pSync, cudaStream_t stream);

CALL_RDXN_ON_STREAM_KERN(int, and);
CALL_RDXN_ON_STREAM_KERN(long, and);
CALL_RDXN_ON_STREAM_KERN(short, and);
CALL_RDXN_ON_STREAM_KERN(double, max);
CALL_RDXN_ON_STREAM_KERN(float, max);
CALL_RDXN_ON_STREAM_KERN(int, max);
CALL_RDXN_ON_STREAM_KERN(long, max);
CALL_RDXN_ON_STREAM_KERN(short, max);
CALL_RDXN_ON_STREAM_KERN(double, min);
CALL_RDXN_ON_STREAM_KERN(float, min);
CALL_RDXN_ON_STREAM_KERN(int, min);
CALL_RDXN_ON_STREAM_KERN(long, min);
CALL_RDXN_ON_STREAM_KERN(short, min);
CALL_RDXN_ON_STREAM_KERN(double, sum);
CALL_RDXN_ON_STREAM_KERN(float, sum);
CALL_RDXN_ON_STREAM_KERN(int, sum);
CALL_RDXN_ON_STREAM_KERN(long, sum);
CALL_RDXN_ON_STREAM_KERN(short, sum);
CALL_RDXN_ON_STREAM_KERN(double, prod);
CALL_RDXN_ON_STREAM_KERN(float, prod);
CALL_RDXN_ON_STREAM_KERN(int, prod);
CALL_RDXN_ON_STREAM_KERN(long, prod);
CALL_RDXN_ON_STREAM_KERN(short, prod);
CALL_RDXN_ON_STREAM_KERN(int, or);
CALL_RDXN_ON_STREAM_KERN(long, or);
CALL_RDXN_ON_STREAM_KERN(short, or);
CALL_RDXN_ON_STREAM_KERN(int, xor);
CALL_RDXN_ON_STREAM_KERN(long, xor);
CALL_RDXN_ON_STREAM_KERN(short, xor);

#if __cplusplus
}
#endif

#define ASSGN_OP_TYPE(TYPE)                           \
    do {                                              \
        if (NULL != strstr(#TYPE, "short")) {         \
            rd_op.op_dt_type = rd_dt_short;           \
        } else if (NULL != strstr(#TYPE, "int")) {    \
            rd_op.op_dt_type = rd_dt_int;             \
        } else if (NULL != strstr(#TYPE, "long")) {   \
            rd_op.op_dt_type = rd_dt_long;            \
        } else if (NULL != strstr(#TYPE, "float")) {  \
            rd_op.op_dt_type = rd_dt_float;           \
        } else if (NULL != strstr(#TYPE, "double")) { \
            rd_op.op_dt_type = rd_dt_double;          \
        } else {                                      \
            rd_op.op_dt_type = rd_dt_null;            \
        }                                             \
    } while (0)

#define ASSGN_OP_TYPE2(TYPE, TYPE2)                                                        \
    do {                                                                                   \
        if (NULL != strstr(#TYPE, "long") && NULL != strstr(#TYPE2, "long")) {             \
            rd_op.op_dt_type = rd_dt_long_long;                                            \
        } else if (NULL != strstr(#TYPE, "long") && NULL != strstr(#TYPE2, "double")) {    \
            rd_op.op_dt_type = rd_dt_long_double;                                          \
        } else if (NULL != strstr(#TYPE, "double") && NULL != strstr(#TYPE2, "complex")) { \
            rd_op.op_dt_type = rd_dt_double_complex;                                       \
        } else if (NULL != strstr(#TYPE, "float") && NULL != strstr(#TYPE2, "complex")) {  \
            rd_op.op_dt_type = rd_dt_float_complex;                                        \
        } else {                                                                           \
            rd_op.op_dt_type = rd_dt_null;                                                 \
        }                                                                                  \
    } while (0)

#endif
