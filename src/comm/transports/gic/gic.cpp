/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "nvshmemi_gic.h"
#include "transport_common.h"
#include "nvshmemx_error.h"
#include "topo.h"

#include "infiniband/verbs.h"
#include "mlx5_ifc.h"
#include "mlx5_prm.h"
#include "infiniband/mlx5dv.h"

#include <string>
#include <cctype>
#include <algorithm>

#include <stddef.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <map>
#include <vector>
#include <deque>
#include <dlfcn.h>
#include <asm/types.h>
#include <netinet/in.h>
#include <endian.h>
#ifdef NVSHMEM_X86_64
#include <immintrin.h>
#endif

#define MAX_NUM_HCAS 16
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32

#define GIC_DC_ACCESS_KEY 0x5623CEAF

#define GIC_MLX5_QPC_ATOMIC_MODE_UP_TO_64BIT 0x3
#define GIC_DBSIZE 8
#define GIC_SRQ_TYPE_VALUE 0x1

#define GIC_LOG_MAX_MSG_SIZE 30    // 30 is max allowed on IB QPs

#define GIC_GPAGE_BITS 16
#define GIC_GPAGE_SIZE (1ULL << GIC_GPAGE_BITS)
#define GIC_GPAGE_OFF  (GIC_GPAGE_SIZE - 1)
#define GIC_GPAGE_MASK (~(GIC_GPAGE_OFF))

#define GIC_ROUND_UP(V,SIZE) (((V)+(SIZE)-1)/(SIZE)*(SIZE))

#define GIC_ROUND_UP_POW2(_n)                               \
    ({                                                      \
         typeof(_n) pow2 = 0;                               \
         assert((_n) >= 1);                                 \
         for (pow2 = 1; pow2 < (_n); pow2 <<= 1);           \
         pow2;                                              \
    })

#define GIC_ROUND_UP_POW2_OR_0(_n)                          \
    ( ((_n) == 0) ? 0 : GIC_ROUND_UP_POW2(_n) )

#define GIC_ROUND_DOWN_POW2_OR_0(_n)                        \
    ({                                                      \
        typeof(_n) pow2 = GIC_ROUND_UP_POW2_OR_0(_n);       \
        ( ((_n) < pow2) ? pow2 / 2 : pow2 );                \
    })

template <typename T>
inline T GIC_ILOG2(T _n) {
    return (T)ceil(log2((double)_n));
}

#define GIC_ILOG2_OR0(_n)                                   \
    ( ((_n) == 0) ? 0 : GIC_ILOG2(_n) )

enum {
    GIC_MLX5_QPC_ST_DCI = 0x5
};

enum {
    GIC_MLX5_UMEM_VALID_DISABLE = 0x0,
    GIC_MLX5_UMEM_VALID_ENABLE = 0x1,
};

typedef enum {
    GIC_NIC_MAPPING_MEMTYPE_REQUEST_AUTO = 0,
    GIC_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM,
    GIC_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM,
} gic_nic_mapping_memtype_reqeust_t;

typedef enum {
    GIC_MEM_TYPE_HOST = 0,
    GIC_MEM_TYPE_GPU = 1,
    GIC_MEM_TYPE_NIC = 2,
} gic_mem_type_t;

struct gic_mem_object {
    gic_mem_type_t         mem_type;
    struct {
        void                   *cpu_ptr;
        CUdeviceptr             gpu_ptr;
        size_t                  size;
    } base;
    struct {
        void                   *cpu_ptr;
        CUdeviceptr             gpu_ptr;
        size_t                  size;
    } aligned;
    union {
        struct mlx5dv_devx_umem *umem;
        struct mlx5dv_devx_uar  *uar;
    };
    bool has_cpu_mapping:1;
    bool has_gpu_mapping:1;
    bool has_nic_mapping:1;
};

typedef enum {
    GIC_QP_TYPE_DCI = 1,
    GIC_QP_TYPE_DCT = 2
} gic_qp_type_t;

struct gic_cq {
    struct mlx5dv_devx_obj         *devx_cq;
    uint32_t                        cqn;
    uint32_t                        num_cqe;
    struct gic_mem_object     *cq_mobject;
    struct gic_mem_object     *dbr_mobject;
    struct mlx5dv_devx_uar         *uar;
};

struct gic_ep {
    gic_qp_type_t              qp_type;

    union {
        struct mlx5dv_devx_obj     *devx_qp;
        struct ibv_qp              *ib_qp;
    };
    uint32_t                        qpn;
    int                             portid;

    size_t                          sq_cnt;
    off_t                           sq_buf_offset;
    size_t                          rq_cnt;
    off_t                           rq_buf_offset;

    struct gic_mem_object     *wq_mobject;
    struct gic_mem_object     *dbr_mobject;
    struct gic_mem_object     *bf_mobject;

    struct gic_cq             *send_cq;    // Valid only on DCI

    uint8_t                         sl;
    uint16_t                        lid;
    uint64_t                        spn;
    uint64_t                        iid;
    uint32_t                        user_index;
};

struct gic_mem_handle {
    uint32_t lkey;
    uint32_t rkey;
    ibv_mr *mr;
};

struct gic_device {
    struct ibv_device *dev;
    struct ibv_pd *pd;  /* protection domain */
    struct ibv_context *context;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr[MAX_NUM_PORTS];
    union ibv_gid gid[MAX_NUM_PORTS];
    struct {
        int num_eps;
        struct gic_ep **eps;
        nvshmemi_gic_device_dct_t *dct_handles;
        struct ibv_pd *pd;  /* parent domain */
        struct ibv_srq *srq;
        struct ibv_cq *send_cq;
        struct ibv_cq *recv_cq;
    } dct;
    struct {
        int num_eps;
        int num_eps_per_sm;
        struct gic_ep **eps;
        struct ibv_srq *srq;
        struct ibv_cq *recv_cq;
        int pdn;
        int srqn;
        int rcqn;
        struct {
            struct gic_mem_object *mem_object;
            struct gic_mem_handle *mem_handle;
            size_t size_per_ep;
        } internal_buf;
    } dci;
    bool support_nic_buf_on_gpumem;
    bool support_nic_buf_on_hostmem;
};

typedef struct {
    void   *devices;
    int    *dev_ids;
    int    *port_ids;
    int     n_dev_ids;
    int     selected_dev_id;
} nvshmemt_gic_state_t;

struct gic_device_mhandle_cache {
    nvshmemi_gic_device_mhandle_t mhandle;
    CUdeviceptr dev_ptr;
};

// CPU cannot dereference this ptr
static nvshmemi_gic_device_state_t *gic_device_state_d = NULL;

// CPU cannot dereference next and rkeys
static std::vector<struct gic_device_mhandle_cache> gic_device_local_mhandles;
static std::vector<struct gic_device_mhandle_cache> gic_device_remote_mhandles;

/* transport constants */
int gic_qp_depth;
int gic_srq_depth;

/* ibv state */
static struct nvshmemt_ibv_function_table ftable;
static void *ibv_handle;

static gic_mem_type_t gic_nic_buf_location;

int nvshmemt_gic_progress(nvshmem_transport_t t) {
    /* TODO: Implement me. Here we need to check for errors from the device */
    return 0;
}

int nvshmemt_gic_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id,
                            int transport_count, int npes, int mype) {
    for (int i = 0; i < npes; ++i) {
        INFO(NVSHMEM_TRANSPORT, "[%d] mem_handle %d : %p", mype, transport_id,
             &mem_handles[i * transport_count + transport_id]);
        struct gic_mem_handle *mem_handle =
            (struct gic_mem_handle *)&mem_handles[i * transport_count + transport_id];
        INFO(NVSHMEM_TRANSPORT, "[%d] lkey %x rkey %x mr %p", mype, mem_handle->lkey,
             mem_handle->rkey, mem_handle->mr);
    }
    return 0;
}

// TODO:
// 1. Check that nvidia.ko supports IO mapping
// 2. Check that nvidia.ko supports persistent vidmem
// 3. Check that nvidia_peermem / nv_peer_mem supports persistent mapping
// 4. Filter mlx5-capable NICs
// 5. Filter only line-layer IB
int nvshmemt_gic_get_device_count(int *ndev, nvshmem_transport_t t) {
    int status = 0;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_gic_state_t *gic_state = (nvshmemt_gic_state_t *)transport->state;

    *ndev = gic_state->n_dev_ids;

    return status;
}

int nvshmemt_gic_get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_gic_state_t *gic_state = (nvshmemt_gic_state_t *)transport->state;
    int dev_id = gic_state->dev_ids[dev];
    const char *ib_name =
        (const char *)((struct gic_device *)gic_state->devices)[dev_id].dev->name;

    status = nvshmemt_ib_iface_get_mlx_path(ib_name, pci_path);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmemt_ib_iface_get_mlx_path failed \n");

out:
    return status;
}

/* TODO: Do we need a new type of access characteristics for gpu-verbs? */
int nvshmemt_gic_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                 nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
              NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS;

    return status;
}

int nvshmemt_gic_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                 nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t length,
                                 nvshmem_transport_t t, bool local_only) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_gic_state_t *gic_state = (nvshmemt_gic_state_t *)transport->state;
    struct gic_device *device =
        ((struct gic_device *)gic_state->devices + gic_state->dev_ids[gic_state->selected_dev_id]);
    struct gic_mem_handle *handle = (struct gic_mem_handle *)mem_handle;

    struct ibv_mr *mr = NULL;

    struct gic_device_mhandle_cache device_mhandle_cache;
    nvshmemi_gic_device_mhandle_t *device_mhandle_h = &device_mhandle_cache.mhandle;
    nvshmemi_gic_device_mhandle_t *device_mhandle_d = NULL;

    CUdeviceptr mhandle_gpu_ptr;

    assert(sizeof(struct gic_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    status = cuMemAlloc((CUdeviceptr *)&device_mhandle_d, sizeof(*device_mhandle_d));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cuMemAlloc failed.\n");

    mr = ftable.reg_mr(device->pd, buf, length,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_ATOMIC);
    NULL_ERROR_JMP(mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "mem registration failed \n");

    handle->lkey = mr->lkey;
    handle->rkey = mr->rkey;
    handle->mr = mr;
    INFO(NVSHMEM_TRANSPORT, "ibv_reg_mr handle %p handle->mr %p", handle, handle->mr);

    device_mhandle_h->lkey = htobe32(mr->lkey);
    device_mhandle_h->start = (uint64_t)buf;
    device_mhandle_h->end = (uint64_t)buf + length - 1;
    device_mhandle_h->next = NULL;

    status = cuMemcpyHtoDAsync((CUdeviceptr)device_mhandle_d,
                               (const void *)device_mhandle_h,
                               sizeof(*device_mhandle_d), 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Copying device_mhandle to GPU memory failed.\n");

    device_mhandle_cache.dev_ptr = (CUdeviceptr)device_mhandle_d;

    if (gic_device_local_mhandles.empty()) {
        assert(gic_device_state_d != NULL);

        mhandle_gpu_ptr = (CUdeviceptr)gic_device_state_d + offsetof(nvshmemi_gic_device_state_t, local_mhandle_head);
    } else {
        struct gic_device_mhandle_cache *last_mhandle_cache = &gic_device_local_mhandles.back();
        mhandle_gpu_ptr = last_mhandle_cache->dev_ptr + offsetof(nvshmemi_gic_device_mhandle_t, next);
        last_mhandle_cache->mhandle.next = device_mhandle_d;
    }
    status = cuMemcpyHtoDAsync(mhandle_gpu_ptr,
                               (const void *)&device_mhandle_d,
                               sizeof(device_mhandle_d), 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Setting local_mhandle_head in GPU memory failed.\n");

    status = cuStreamSynchronize(nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "stream synchronize failed.\n");

    gic_device_local_mhandles.emplace_back(device_mhandle_cache);

out:
    if (status) {
        if (device_mhandle_d) cuMemFree((CUdeviceptr)device_mhandle_d);
        if (mr) ftable.dereg_mr(mr);
    }
    return status;
}

static int gic_mobject_nic_map(struct gic_mem_object *mobject, struct ibv_context *context, uint32_t access) {
    int status = 0;
    void *addr;
    struct mlx5dv_devx_umem *umem = NULL;

    assert(mobject);
    assert(!mobject->has_nic_mapping);
    assert(context);

    if (mobject->mem_type == GIC_MEM_TYPE_GPU) {
        addr = (void *)mobject->aligned.gpu_ptr;
    } else if (mobject->mem_type == GIC_MEM_TYPE_HOST) {
        addr = mobject->aligned.cpu_ptr;
    } else {
        assert(0);
    }

    umem = mlx5dv_devx_umem_reg(context, addr, mobject->aligned.size, access);
    if (!umem) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    mobject->umem = umem;
    mobject->has_nic_mapping = true;

out:
    return status;
}

static void gic_mobject_nic_unmap(struct gic_mem_object *mobject) {
    int status = 0;

    assert(mobject);
    assert(mobject->has_nic_mapping);
    assert(mobject->mem_type != GIC_MEM_TYPE_NIC);
    assert(mobject->umem);

    status = mlx5dv_devx_umem_dereg(mobject->umem);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_umem_dereg failed.\n");

    mobject->has_nic_mapping = false;
    mobject->umem = NULL;

out:
    return;
}

static int gic_gpu_mem_alloc(struct gic_mem_object **pmobject, size_t size, size_t alignment, bool host_mapping) {
    // TODO: Support host mapping through gdrcopy or dmabuf
    assert(!host_mapping);

    int status = 0;

    int attr_val;

    CUdeviceptr ptr = 0;
    CUdeviceptr aligned_ptr;
    size_t bufsize = size;

    struct gic_mem_object *mobject = (struct gic_mem_object *)calloc(1, sizeof(struct gic_mem_object));
    NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate a new mobject.\n");

    if (alignment > 0)
        bufsize = size + alignment - 1;

    status = cuMemAlloc(&ptr, bufsize);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemAlloc failed.\n");

    attr_val = 1;
    status = cuPointerSetAttribute(&attr_val, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuPointerSetAttribute failed.\n");

    status = cuMemsetD8(ptr, 0, bufsize);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemsetD8 failed.\n");

    if (alignment > 0) {
        aligned_ptr = (ptr + alignment - 1) & (~(alignment - 1));
    } else {
        aligned_ptr = ptr;
    }

    mobject->mem_type = GIC_MEM_TYPE_GPU;

    mobject->base.gpu_ptr = ptr;
    mobject->base.size = bufsize;

    mobject->aligned.gpu_ptr = aligned_ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = false;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = false;

    *pmobject = mobject;

out:
    if (status) {
        if (ptr) {
            CUresult _status = cuMemFree(ptr);
            CUDA_DRIVER_ERROR_STRING(_status);
        }

        if (mobject) free(mobject);
    }
    return status;
}

static void gic_gpu_mem_free(struct gic_mem_object *mobject) {
    CUresult status;

    if (!mobject)
        return;

    assert(mobject->mem_type == GIC_MEM_TYPE_GPU);

    status = cuMemFree(mobject->base.gpu_ptr);
    CUDA_DRIVER_ERROR_STRING(status);

    free(mobject);
}

static int gic_host_mem_alloc(struct gic_mem_object **pmobject, size_t size, size_t alignment, bool gpu_mapping) {
    int status;

    void *ptr = NULL;

    bool did_host_reg = false;
    CUdeviceptr gpu_ptr;

    struct gic_mem_object *mobject = (struct gic_mem_object *)calloc(1, sizeof(struct gic_mem_object));
    NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate a new mobject.\n");

    status = posix_memalign(&ptr, alignment, size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "posix_memalign failed.\n");

    memset(ptr, 0, size);

    if (gpu_mapping) {
        status = cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemsetD8 failed.\n");
        did_host_reg = true;

        status = cuMemHostGetDevicePointer(&gpu_ptr, ptr, 0);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemHostGetDevicePointer failed.\n");

        mobject->base.gpu_ptr = gpu_ptr;
        mobject->aligned.gpu_ptr = gpu_ptr;
        mobject->has_gpu_mapping = true;
    }

    mobject->base.cpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            CUresult _status = cuMemHostUnregister(ptr);
            CUDA_DRIVER_ERROR_STRING(_status);
        }
        if (ptr) free(ptr);
        if (mobject) free(mobject);
    }
    return status;
}

static void gic_host_mem_free(struct gic_mem_object *mobject) {
    CUresult status;

    if (!mobject)
        return;

    assert(mobject->mem_type == GIC_MEM_TYPE_HOST);

    if (mobject->has_gpu_mapping) {
        status = cuMemHostUnregister(mobject->base.cpu_ptr);
        CUDA_DRIVER_ERROR_STRING(status);
    }

    free(mobject->base.cpu_ptr);

    free(mobject);
}

static int gic_nic_mem_gpu_map(struct gic_mem_object **pmobject, struct mlx5dv_devx_uar *uar, size_t size) {
    int status = 0;
    bool did_host_reg = false;

    CUdeviceptr ptr = 0;

    struct gic_mem_object *mobject = (struct gic_mem_object *)calloc(1, sizeof(struct gic_mem_object));
    NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate a new mobject.\n");

    status = cuMemHostRegister(uar->reg_addr, size, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemHostRegister failed.\n");
    did_host_reg = true;

    status = cuMemHostGetDevicePointer(&ptr, uar->reg_addr, 0);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemHostGetDevicePointer failed.\n");

    mobject->mem_type = GIC_MEM_TYPE_NIC;

    mobject->base.cpu_ptr = uar->reg_addr;
    mobject->base.gpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = uar->reg_addr;
    mobject->aligned.gpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->uar = uar;

    mobject->has_cpu_mapping = true;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            CUresult _status = cuMemHostUnregister(uar->reg_addr);
            CUDA_DRIVER_ERROR_STRING(_status);
        }
        if (mobject) free(mobject);
    }
    return status;
}

static void gic_nic_mem_gpu_unmap(struct gic_mem_object *mobject) {
    CUresult status;

    if (!mobject)
        return;

    assert(mobject->mem_type == GIC_MEM_TYPE_NIC);

    status = cuMemHostUnregister(mobject->uar->reg_addr);
    CUDA_DRIVER_ERROR_STRING(status);

    free(mobject);
}

static inline int gic_nic_control_alloc(struct gic_mem_object **pmobject, size_t size, size_t alignment) {
    assert(gic_nic_buf_location == GIC_MEM_TYPE_GPU || gic_nic_buf_location == GIC_MEM_TYPE_HOST);
    if (gic_nic_buf_location == GIC_MEM_TYPE_GPU)
        return gic_gpu_mem_alloc(pmobject, size, alignment, false);
    else
        return gic_host_mem_alloc(pmobject, size, alignment, true);
}

static inline void gic_nic_control_free(struct gic_mem_object *mobject) {
    assert(gic_nic_buf_location == GIC_MEM_TYPE_GPU || gic_nic_buf_location == GIC_MEM_TYPE_HOST);
    if (gic_nic_buf_location == GIC_MEM_TYPE_GPU)
        gic_gpu_mem_free(mobject);
    else
        gic_host_mem_free(mobject);
}

static int gic_create_cq(struct gic_cq **pgcq, const struct gic_device *device, int ncqes) {
    int status = 0;

    struct gic_cq *gcq = NULL;

    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;

    void *cq_context;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_cq_in)] = {0,};
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_cq_out)] = {0,};

    struct gic_mem_object *cq_mobject = NULL;
    struct mlx5dv_devx_umem *cq_umem = NULL;
    int num_cqe = GIC_ROUND_UP_POW2_OR_0(ncqes);
    size_t cq_buf_size = num_cqe * NVSHMEMI_GIC_CQE_SIZE;

    struct gic_mem_object *dbr_mobject = NULL;
    struct mlx5dv_devx_umem *dbr_umem = NULL;

    struct mlx5dv_devx_uar *uar = NULL;

    uint32_t eqn;

    gcq = (struct gic_cq *)calloc(1, sizeof(struct gic_cq));
    NULL_ERROR_JMP(gcq, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate mem for cq.\n");

    // Allocate and map CQ buffer
    status = gic_nic_control_alloc(&cq_mobject, cq_buf_size, GIC_GPAGE_SIZE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate cq buf.\n");

    status = cuMemsetD8(cq_mobject->base.gpu_ptr, 0xff, cq_mobject->base.size);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemsetD8 failed.\n");

    status = gic_mobject_nic_map(cq_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register cq buf.\n");
    cq_umem = cq_mobject->umem;

    // Allocate and map DBR
    status = gic_nic_control_alloc(&dbr_mobject, GIC_DBSIZE, GIC_GPAGE_SIZE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate dbr buf for qpair.\n");

    status = gic_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register dbr buf for qpair.\n");
    dbr_umem = dbr_mobject->umem;

    // Query the first EQ
    status = mlx5dv_devx_query_eqn(context, 0, &eqn);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_query_eqn failed.\n");

    // CQ needs UAR but GIC never uses it.
    // So, we don't map this UAR to GPU space.
    uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
    NULL_ERROR_JMP(uar, status, ENOMEM, out, "cannot allocate mlx5dv_devx_uar\n");

    DEVX_SET(create_cq_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_CQ);
    DEVX_SET(create_cq_in, cmd_in, cq_umem_id, cq_umem->umem_id); // CQ buffer
    DEVX_SET(create_cq_in, cmd_in, cq_umem_valid, GIC_MLX5_UMEM_VALID_ENABLE); // Enable cq_umem_id
    DEVX_SET(create_cq_in, cmd_in, cq_umem_offset, 0x0); 

    cq_context = DEVX_ADDR_OF(create_cq_in, cmd_in, cq_context);
    DEVX_SET(cqc, cq_context, dbr_umem_valid, GIC_MLX5_UMEM_VALID_ENABLE);
    DEVX_SET(cqc, cq_context, cqe_sz, MLX5_CQE_SIZE_64B);
    DEVX_SET(cqc, cq_context, dbr_umem_id, dbr_umem->umem_id);
    DEVX_SET(cqc, cq_context, log_cq_size, GIC_ILOG2_OR0(num_cqe));
    DEVX_SET(cqc, cq_context, uar_page, uar->page_id);
    DEVX_SET(cqc, cq_context, c_eqn, eqn);
    DEVX_SET(cqc, cq_context, log_page_size, GIC_GPAGE_BITS - MLX5_ADAPTER_PAGE_SHIFT);
    DEVX_SET64(cqc, cq_context, dbr_addr, 0x0); // DBR offset

    gcq->devx_cq = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NULL_ERROR_JMP(gcq->devx_cq, status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create CQ.\n");

    gcq->cqn = DEVX_GET(create_cq_out, cmd_out, cqn);
    gcq->num_cqe = num_cqe;
    gcq->cq_mobject = cq_mobject;
    gcq->dbr_mobject = dbr_mobject;
    gcq->uar = uar;

    *pgcq = gcq;

out:
    if (status) {
        if (uar) mlx5dv_devx_free_uar(uar);
        if (dbr_umem) gic_mobject_nic_unmap(dbr_mobject);
        if (dbr_mobject) gic_nic_control_free(dbr_mobject);
        if (cq_umem) gic_mobject_nic_unmap(cq_mobject);
        if (cq_mobject) gic_nic_control_free(cq_mobject);
        if (gcq) free(gcq);
    }
    return status;
}

static void gic_destroy_cq(struct gic_cq *gcq) {
}

static void gic_get_device_cq(nvshmemi_gic_device_cq_t *dev_cq, const struct gic_cq *cq) {
    dev_cq->lock = 0;
    dev_cq->cqn = cq->cqn;
    dev_cq->ncqes = cq->num_cqe;

    assert(cq->cq_mobject->has_gpu_mapping);
    dev_cq->cqe = (void *)cq->cq_mobject->aligned.gpu_ptr;

    assert(cq->dbr_mobject->has_gpu_mapping);
    dev_cq->dbrec = (__be32 *)cq->dbr_mobject->aligned.gpu_ptr;
}


static int gic_dci_rst2init(struct gic_ep *ep, const struct gic_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {0,};
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {0,};

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == GIC_QP_TYPE_DCI);

    DEVX_SET(rst2init_qp_in, cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
    DEVX_SET(rst2init_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(rst2init_qp_in, cmd_in, qpc);
    DEVX_SET64(qpc, qpc, dc_access_key, GIC_DC_ACCESS_KEY);
    DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, portid);

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND)
        DEVX_SET(qpc, qpc, primary_address_path.pkey_index, 0);

    DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qpc, counter_set_id, 0x0);    // Not connected to a counter set

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, 
        "Error in mlx5dv_devx_obj_modify for RST2INIT_QP with syndrome %x\n", DEVX_GET(rst2init_qp_out, cmd_out, syndrome));

    ep->portid = portid;

out:
    return status;
}

static int gic_dci_init2rtr(struct gic_ep *ep, const struct gic_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {0,};
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {0,};

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == GIC_QP_TYPE_DCI);

    DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, mtu, port_attr->active_mtu);
    DEVX_SET(qpc, qpc, log_msg_max, GIC_LOG_MAX_MSG_SIZE);

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DEVX_SET(qpc, qpc, primary_address_path.sl, nvshmemi_options.IB_SL);
        ep->sl = nvshmemi_options.IB_SL;
    }

    if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
        DEVX_SET(qpc, qpc, primary_address_path.tclass, nvshmemi_options.IB_TRAFFIC_CLASS);
        DEVX_SET(qpc, qpc, primary_address_path.eth_prio, nvshmemi_options.IB_SL);
        DEVX_SET(qpc, qpc, primary_address_path.dscp, nvshmemi_options.IB_TRAFFIC_CLASS >> 2);
    }

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, 
        "Error in mlx5dv_devx_obj_modify for INIT2RTR_QP with syndrome %x\n", DEVX_GET(init2rtr_qp_out, cmd_out, syndrome));

out:
    return status;
}

static int gic_dci_rtr2rts(struct gic_ep *ep, const struct gic_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {0,};
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {0,};

    void *qpc;

    assert(ep->qp_type == GIC_QP_TYPE_DCI);

    DEVX_SET(rtr2rts_qp_in, cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
    DEVX_SET(rtr2rts_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(rtr2rts_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, log_ack_req_freq, 0x0);  // Ack every packet
    DEVX_SET(qpc, qpc, log_sra_max, GIC_ILOG2_OR0(device->device_attr.max_qp_rd_atom));
    DEVX_SET(qpc, qpc, next_send_psn, 0x0);
    DEVX_SET(qpc, qpc, retry_count, 7);
    DEVX_SET(qpc, qpc, rnr_retry, 7);
    DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 20);

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, 
        "Error in mlx5dv_devx_obj_modify for RTR2RTS_QP with syndrome %x\n", DEVX_GET(rtr2rts_qp_out, cmd_out, syndrome));

out:
    return status;
}

static int gic_create_dci_shared_objects(struct gic_device *device) {
    int status = 0;

    struct ibv_context *context = device->context;
    struct ibv_pd *pd = device->pd;

    struct ibv_srq *srq = NULL;
    struct ibv_srq_init_attr srq_init_attr;

    struct ibv_cq *recv_cq = NULL;

    mlx5dv_obj dv_obj;
    struct mlx5dv_pd dvpd;
    struct mlx5dv_cq dvscq;
    struct mlx5dv_cq dvrcq;
    struct mlx5dv_srq dvsrq;

    int pdn = 0;
    int srqn = 0;
    int rcqn = 0;

    struct gic_mem_object *internal_buf_mobject = NULL;
    struct gic_mem_handle *internal_buf_mhandle = NULL;
    struct ibv_mr *internal_buf_mr = NULL;
    size_t internal_buf_size_per_ep = 0;

    int warp_size;

    // Initialization
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));
    memset(&dvpd, 0, sizeof(dvpd));
    memset(&dvscq, 0, sizeof(dvscq));
    memset(&dvrcq, 0, sizeof(dvrcq));
    memset(&dvsrq, 0, sizeof(dvsrq));

    // Query pdn
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.pd.in = pd;
    dv_obj.pd.out = &dvpd;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv PD initialization failed.\n");

    pdn = dvpd.pdn;

    // Create srq on host memory.
    srq_init_attr.attr.max_wr = gic_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    srq = ftable.create_srq(pd, &srq_init_attr);
    NULL_ERROR_JMP(srq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_srq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    dvsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;
    dv_obj.srq.in = srq;
    dv_obj.srq.out = &dvsrq;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_SRQ);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv SRQ initialization failed.\n");

    srqn = dvsrq.srqn;
    EQ_ERROR_JMP(srqn, 0, NVSHMEMX_ERROR_INTERNAL, out,
                 "Unable to allocate SRQ for your device. "
                 "This may occur if your ofed is older than version 5.0.\n");

    // Create recv_cq on host memory.
    recv_cq = ftable.create_cq(context, gic_srq_depth, NULL, NULL, 0);
    NULL_ERROR_JMP(recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_cq for recv_cq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.cq.in = recv_cq;
    dv_obj.cq.out = &dvrcq;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_CQ);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv RCQ initialization failed.\n");

    rcqn = dvrcq.cqn;

    status = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, nvshmemi_state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGetAttribute querying warp size failed.\n");

    internal_buf_size_per_ep = sizeof(uint64_t) + NVSHMEMI_GIC_MAX_INLINE_SIZE * warp_size;
    status = gic_gpu_mem_alloc(&internal_buf_mobject, internal_buf_size_per_ep * device->dci.num_eps, GIC_GPAGE_SIZE, false);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate internal buffer.\n");

    internal_buf_mhandle = (struct gic_mem_handle *)calloc(1, sizeof(*internal_buf_mhandle));
    NULL_ERROR_JMP(internal_buf_mhandle, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate internal_buf_mhandle.\n");

    internal_buf_mr = ftable.reg_mr(device->pd, (void *)internal_buf_mobject->aligned.gpu_ptr, 
        internal_buf_mobject->aligned.size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC
    );
    NULL_ERROR_JMP(internal_buf_mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "mem registration for internal buffer failed.\n");

    internal_buf_mhandle->mr = internal_buf_mr;
    internal_buf_mhandle->lkey = internal_buf_mr->lkey;
    internal_buf_mhandle->rkey = internal_buf_mr->rkey;

    // Output
    device->dci.srq = srq;
    device->dci.recv_cq = recv_cq;
    device->dci.pdn = pdn;
    device->dci.srqn = srqn;
    device->dci.rcqn = rcqn;
    device->dci.internal_buf.mem_object = internal_buf_mobject;
    device->dci.internal_buf.mem_handle = internal_buf_mhandle;
    device->dci.internal_buf.size_per_ep = internal_buf_size_per_ep;

out:
    if (status) {
        if (internal_buf_mr) ftable.dereg_mr(internal_buf_mr);
        if (internal_buf_mhandle) free(internal_buf_mhandle);
        if (internal_buf_mobject) gic_gpu_mem_free(internal_buf_mobject);
        if (recv_cq) ftable.destroy_cq(recv_cq);
        if (srq) ftable.destroy_srq(srq);
    }
    return status;
}

static int gic_create_dci(struct gic_ep **ep_ptr, struct gic_device *device, int portid, uint32_t dci_idx) {
    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;
    struct gic_ep *ep = NULL;

    void *qp_context;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {0,};
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {0,};

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {0,};
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {0,};
    void *cap;

    size_t bf_reg_size;
    uint8_t log_bf_reg_size;
    struct mlx5dv_devx_uar *bf_uar = NULL;
    struct gic_mem_object *bf_mobject = NULL;

    size_t wq_buf_size;
    struct gic_mem_object *wq_mobject = NULL;
    struct mlx5dv_devx_umem *wq_umem = NULL;

    struct gic_mem_object *dbr_mobject = NULL;
    struct mlx5dv_devx_umem *dbr_umem = NULL;

    int cqe_version = 0;
    int amo_endianness_mode;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    struct gic_cq *send_cq = NULL;

    size_t num_wqebb = GIC_ROUND_UP_POW2_OR_0(NVSHMEMI_GIC_MAX_WQEBB_PER_WQE * gic_qp_depth);

    int status = 0;

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE |
        (MLX5_CAP_GENERAL << 1) |
        HCA_CAP_OPMOD_GET_CUR
    );

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out, sizeof(cmd_cap_out));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
    log_bf_reg_size = DEVX_GET(cmd_hca_cap, cap, log_bf_reg_size);

    cqe_version = DEVX_GET(cmd_hca_cap, cap, cqe_version);
    if (cqe_version != 1) {
        ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out, "hca_cap.cqe_version != 1 is not supported.\n");
    }

    DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE |
        (MLX5_CAP_ATOMIC << 1) |
        HCA_CAP_OPMOD_GET_MAX
    );
    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out, sizeof(cmd_cap_out));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_general_cmd for atomic caps failed.\n");

    /* Report whether we need to do atomic endianness conversions on 8 byte operands. */
    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.atomic_caps);
    amo_endianness_mode = DEVX_GET(atomic_caps, cap, atomic_req_8B_endianness_mode);
    if (amo_endianness_mode) {
        nvshmemi_state->atomic_host_endian_min_size = 8;
    } else {
        nvshmemi_state->atomic_host_endian_min_size = UINT32_MAX;
    }

    // Create send_cq on GPU memory.
    status = gic_create_cq(&send_cq, device, gic_qp_depth);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_create_cq failed.\n");

    ep = (struct gic_ep *)calloc(1, sizeof(struct gic_ep));
    NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate mem for ep.\n");

    // The size of 1st + 2nd half (as when we use alternating DB)
    bf_reg_size = 1LLU << log_bf_reg_size;

    // Allocate UAR. This will be used as a DB/BF register).
    bf_uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_BF);
    NULL_ERROR_JMP(bf_uar, status, ENOMEM, out, "cannot allocate mlx5dv_devx_uar\n");

    // Map the UAR to GPU
    status = gic_nic_mem_gpu_map(&bf_mobject, bf_uar, bf_reg_size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_nic_mem_gpu_map failed.\n");

    // Allocate WQ buffer.
    wq_buf_size = num_wqebb * MLX5_SEND_WQE_BB; // num_wqebb is always a power of 2
    status = gic_nic_control_alloc(&wq_mobject, wq_buf_size, GIC_GPAGE_SIZE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate wq buf for qpair.\n");

    status = gic_mobject_nic_map(wq_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register wq buf for qpair.\n");
    wq_umem = wq_mobject->umem;

    // Allocate Doorbell Register buffer.
    status = gic_nic_control_alloc(&dbr_mobject, GIC_DBSIZE, GIC_GPAGE_SIZE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate dbr buf for qpair.\n");

    status = gic_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register dbr buf for qpair.\n");
    dbr_umem = dbr_mobject->umem;

    DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
    DEVX_SET(create_qp_in, cmd_in, wq_umem_id, wq_umem->umem_id); // WQ buffer
    DEVX_SET(create_qp_in, cmd_in, wq_umem_valid, GIC_MLX5_UMEM_VALID_ENABLE); // Enable wq_umem_id

    qp_context = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qp_context, st, GIC_MLX5_QPC_ST_DCI);
    DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qp_context, pd, device->dci.pdn);
    DEVX_SET(qpc, qp_context, uar_page, bf_uar->page_id);     // BF register
    DEVX_SET(qpc, qp_context, rq_type, GIC_SRQ_TYPE_VALUE); // Shared Receive Queue
    DEVX_SET(qpc, qp_context, srqn_rmpn_xrqn, device->dci.srqn);
    DEVX_SET(qpc, qp_context, cqn_snd, send_cq->cqn);
    DEVX_SET(qpc, qp_context, cqn_rcv, device->dci.rcqn);
    DEVX_SET(qpc, qp_context, log_sq_size, GIC_ILOG2_OR0(num_wqebb));
    DEVX_SET(qpc, qp_context, log_rq_size, 0);
    DEVX_SET(qpc, qp_context, cs_req, 0);  // Disable CS Request
    DEVX_SET(qpc, qp_context, cs_res, 0);  // Disable CS Response
    DEVX_SET(qpc, qp_context, dbr_umem_valid, GIC_MLX5_UMEM_VALID_ENABLE); // Enable dbr_umem_id
    DEVX_SET64(qpc, qp_context, dbr_addr, 0); // Offset 0 of dbr_umem_id (behavior changed because of dbr_umem_valid)
    DEVX_SET(qpc, qp_context, dbr_umem_id, dbr_umem->umem_id); // DBR buffer
    DEVX_SET(qpc, qp_context, user_index, dci_idx);
    DEVX_SET(qpc, qp_context, page_offset, 0);

    ep->devx_qp = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NULL_ERROR_JMP(ep->devx_qp, status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create DCI for EP.\n");

    ep->qpn = DEVX_GET(create_qp_out, cmd_out, qpn);

    ep->sq_cnt = num_wqebb;
    ep->sq_buf_offset = 0;

    ep->rq_cnt = 0;
    ep->rq_buf_offset = 0;

    ep->wq_mobject = wq_mobject;
    ep->dbr_mobject = dbr_mobject;
    ep->bf_mobject = bf_mobject;

    ep->send_cq = send_cq;

    ep->qp_type = GIC_QP_TYPE_DCI;

    ep->lid = port_attr->lid;
    ep->user_index = dci_idx;

    *ep_ptr = ep;

out: 
    if (status) {
        if (dbr_umem) gic_mobject_nic_unmap(dbr_mobject);
        if (dbr_mobject) gic_nic_control_free(dbr_mobject);
        if (wq_umem) gic_mobject_nic_unmap(wq_mobject);
        if (wq_mobject) gic_nic_control_free(wq_mobject);
        if (bf_mobject) gic_nic_mem_gpu_unmap(bf_mobject);
        if (bf_uar) mlx5dv_devx_free_uar(bf_uar);
        if (send_cq) gic_destroy_cq(send_cq);
        if (ep) free(ep);
    }

    return status;
}

static int gic_create_dct_shared_objects(struct gic_device *device) {
    int status = 0;

    struct ibv_context *context = device->context;

    struct ibv_pd *pd = NULL;
    struct ibv_parent_domain_init_attr pd_init_attr;

    struct ibv_srq *srq = NULL;
    struct ibv_srq_init_attr srq_init_attr;

    struct ibv_cq *send_cq = NULL;
    struct ibv_cq *recv_cq = NULL;

    memset(&pd_init_attr, 0, sizeof(pd_init_attr));
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));

    pd_init_attr.pd = device->pd;
    pd = ibv_alloc_parent_domain(context, &pd_init_attr);
    NULL_ERROR_JMP(pd, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_alloc_parent_domain failed.\n");

    srq_init_attr.attr.max_wr = gic_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    srq = ftable.create_srq(pd, &srq_init_attr);
    NULL_ERROR_JMP(srq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_srq failed.\n");

    send_cq = ftable.create_cq(context, gic_srq_depth, NULL, NULL, 0);
    NULL_ERROR_JMP(send_cq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_cq for send_cq failed.\n");

    recv_cq = ftable.create_cq(context, gic_srq_depth, NULL, NULL, 0);
    NULL_ERROR_JMP(recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_cq for recv_cq failed.\n");
    
    device->dct.pd = pd;
    device->dct.srq = srq;
    device->dct.send_cq = send_cq;
    device->dct.recv_cq = recv_cq;

out:
    if (status) {
        if (recv_cq) ftable.destroy_cq(recv_cq);
        if (send_cq) ftable.destroy_cq(send_cq);
        if (srq) ftable.destroy_srq(srq);
    }
    return status;
}

static int gic_create_dct(struct gic_ep **ep_ptr, const struct gic_device *device, int portid) {
    int status = 0;

    struct gic_ep *ep = NULL;
    struct ibv_qp *ib_qp = NULL;

    struct ibv_qp_init_attr_ex ib_qp_attr_ex;
    struct mlx5dv_qp_init_attr dv_init_attr;
    struct ibv_qp_attr ib_qp_attr;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    uint64_t spn = 0;
    uint64_t iid = 0;

    memset(&ib_qp_attr_ex, 0, sizeof(ib_qp_attr_ex));
    memset(&dv_init_attr, 0, sizeof(dv_init_attr));

    ep = (struct gic_ep *)calloc(1, sizeof(struct gic_ep));
    NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate mem for ep.\n");

    dv_init_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_init_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    dv_init_attr.dc_init_attr.dct_access_key = GIC_DC_ACCESS_KEY;

    ib_qp_attr_ex.pd = device->dct.pd;
    ib_qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD;
    ib_qp_attr_ex.qp_type = IBV_QPT_DRIVER;
    ib_qp_attr_ex.srq = device->dct.srq;
    ib_qp_attr_ex.send_cq = device->dct.send_cq;
    ib_qp_attr_ex.recv_cq = device->dct.recv_cq;

    ib_qp_attr_ex.cap.max_send_wr = nvshmemi_options.QP_DEPTH;
    ib_qp_attr_ex.cap.max_recv_wr = nvshmemi_options.QP_DEPTH;
    ib_qp_attr_ex.cap.max_send_sge = 1;
    ib_qp_attr_ex.cap.max_recv_sge = 1;
    ib_qp_attr_ex.cap.max_inline_data = NVSHMEMI_GIC_MAX_INLINE_SIZE;

    ib_qp = mlx5dv_create_qp(device->context, &ib_qp_attr_ex, &dv_init_attr);
    NULL_ERROR_JMP(ib_qp, status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_create_qp failed.\n");

    // RST2INIT
    memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
    ib_qp_attr.qp_state        = IBV_QPS_INIT;
    ib_qp_attr.pkey_index      = 0;
    ib_qp_attr.port_num        = portid;
    ib_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | 
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    
    status = ftable.modify_qp(ib_qp, &ib_qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp rst2init for dct failed.\n");

    // INIT2RTR
    memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
    ib_qp_attr.qp_state                     = IBV_QPS_RTR;
    ib_qp_attr.path_mtu                     = port_attr->active_mtu;
    if (port_attr->lid == 0) {
        spn                                     = device->gid[portid - 1].global.subnet_prefix;
        iid                                     = device->gid[portid - 1].global.interface_id;
        ib_qp_attr.ah_attr.is_global            = 1;
        ib_qp_attr.ah_attr.grh.dgid.global.subnet_prefix    = spn;
        ib_qp_attr.ah_attr.grh.dgid.global.interface_id     = iid;
        ib_qp_attr.ah_attr.grh.flow_label       = 0;
        ib_qp_attr.ah_attr.grh.sgid_index       = nvshmemi_options.IB_GID_INDEX;
        ib_qp_attr.ah_attr.grh.hop_limit        = 255;
        ib_qp_attr.ah_attr.grh.traffic_class    = nvshmemi_options.IB_TRAFFIC_CLASS;
    } else {
        ib_qp_attr.ah_attr.dlid = port_attr->lid;
        ib_qp_attr.ah_attr.is_global = 0;
    }
    ib_qp_attr.ah_attr.sl                   = nvshmemi_options.IB_SL;
    ib_qp_attr.ah_attr.src_path_bits        = 0;
    ib_qp_attr.ah_attr.port_num             = portid;
    ib_qp_attr.min_rnr_timer                = 12;

    status = ftable.modify_qp(ib_qp, &ib_qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_MIN_RNR_TIMER);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp init2rtr for dct failed.\n");

    ep->qp_type = GIC_QP_TYPE_DCT;

    ep->ib_qp = ib_qp;
    ep->qpn = ib_qp->qp_num;
    ep->portid = portid;
    ep->lid = port_attr->lid;
    ep->spn = spn;
    ep->iid = iid;

    *ep_ptr = ep;

out:
    if (status) {
        if (ib_qp) {
            int _status = ftable.destroy_qp(ib_qp);
            if (_status) ERROR_PRINT("ibv_destroy_qp for dct failed.\n");
        }
        if (ep) free(ep);
    }
    return status;
}

static int gic_get_dct_handle(nvshmemi_gic_device_dct_t *dct_handle, const struct gic_ep *ep) {
    assert(ep->qp_type == GIC_QP_TYPE_DCT);

    dct_handle->qpn = ep->qpn;
    dct_handle->access_key = GIC_DC_ACCESS_KEY;
    dct_handle->lid = ep->lid;

    return 0;
}

static int gic_destroy_ep(struct gic_ep *ep_ptr, nvshmemt_gic_state_t *gic_state) {
    // TODO: Implement me
    return 0;
}

static void gic_get_device_dci(nvshmemi_gic_device_dci_t *dev_dci, struct gic_device *device, const struct gic_ep *ep) {
    struct mlx5_wqe_ctrl_seg *ctrl_seg;
    nvshmemi_gic_mlx5_wqe_half_av_t *half_av_seg;

    assert(ep->qp_type == GIC_QP_TYPE_DCI);

    dev_dci->qpn = ep->qpn;

    assert(ep->wq_mobject->has_gpu_mapping);
    dev_dci->tx_wq.wqe = (void *)ep->wq_mobject->aligned.gpu_ptr;

    assert(ep->dbr_mobject->has_gpu_mapping);
    dev_dci->tx_wq.dbrec = (__be32 *)ep->dbr_mobject->aligned.gpu_ptr;

    assert(ep->bf_mobject->has_gpu_mapping);
    dev_dci->tx_wq.bf = (void *)ep->bf_mobject->aligned.gpu_ptr;

    dev_dci->tx_wq.nwqes = ep->sq_cnt;

    for (int i = NVSHMEMI_GIC_DS_MIN; i <= NVSHMEMI_GIC_DS_MAX; ++i) {
        ctrl_seg = &dev_dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(i)];
        memset(ctrl_seg, 0, sizeof(*ctrl_seg));
        ctrl_seg->qpn_ds = htobe32(ep->qpn << 8 | (i & 0x3F));
        ctrl_seg->signature = 0;
        // We don't use DCS in this implementation. But it will be used in the future.
        ctrl_seg->dci_stream_channel_id = 0;
        ctrl_seg->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
        ctrl_seg->imm = 0;
    }

    half_av_seg = &dev_dci->half_av_seg_template;
    memset(half_av_seg, 0, sizeof(*half_av_seg));
    half_av_seg->stat_rate_sl = ep->sl & 0x0F;
    half_av_seg->fl_mlid = ep->lid & 0x7F;

    dev_dci->internal_buf.lkey = htobe32(device->dci.internal_buf.mem_handle->lkey);
    dev_dci->internal_buf.rkey = htobe32(device->dci.internal_buf.mem_handle->rkey);
    dev_dci->internal_buf.buf = (void *)((uintptr_t)device->dci.internal_buf.mem_object->aligned.gpu_ptr + (device->dci.internal_buf.size_per_ep * ep->user_index));

    dev_dci->lock = 0;
}

static void gic_get_device_dct(nvshmemi_gic_device_dct_t *dev_dct, const nvshmemi_gic_device_dct_t *dct_handle) {
    dev_dct->qpn = htobe32(dct_handle->qpn);
    dev_dct->access_key = htobe64(dct_handle->access_key);
    dev_dct->lid = htobe16(dct_handle->lid);
}

static int gic_setup_gpu_state(struct gic_device *device) {
    int status = 0;

    int num_dct_handles = device->dct.num_eps * nvshmemi_state->npes;

    nvshmemi_device_state_t *device_state = NULL;

    nvshmemi_gic_device_state_t gic_device_state_h;

    nvshmemi_gic_device_dci_t *dci_d = NULL;
    nvshmemi_gic_device_dci_t *dci_h = NULL;

    nvshmemi_gic_device_dct_t *dct_d = NULL;
    nvshmemi_gic_device_dct_t *dct_h = NULL;

    nvshmemi_gic_device_cq_t *cq_d = NULL;
    nvshmemi_gic_device_cq_t *cq_h = NULL;

    assert(gic_device_state_d == NULL);

    // Setup DCT table
    status = cuMemAlloc((CUdeviceptr *)&dct_d, num_dct_handles * sizeof(*dct_d));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "dct_d allocation failed.\n");

    dct_h = (nvshmemi_gic_device_dct_t *)calloc(num_dct_handles, sizeof(*dct_h));
    NULL_ERROR_JMP(dct_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "dct_h allocation failed.\n");

    for (int i = 0; i < num_dct_handles; ++i) {
        gic_get_device_dct(&dct_h[i], &device->dct.dct_handles[i]);
    }

    status = cuMemcpyHtoDAsync((CUdeviceptr)dct_d,
                               (const void *)dct_h,
                               sizeof(*dct_d) * num_dct_handles, 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Copying dct_h to dct_d failed.\n");

    // Get GPU DCIs and send CQs
    status = cuMemAlloc((CUdeviceptr *)&dci_d, device->dci.num_eps * sizeof(*dci_d));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "dci_d allocation failed.\n");

    dci_h = (nvshmemi_gic_device_dci_t *)calloc(device->dci.num_eps, sizeof(*dci_h));
    NULL_ERROR_JMP(dci_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "dci_h allocation failed.\n");

    status = cuMemAlloc((CUdeviceptr *)&cq_d, device->dci.num_eps * sizeof(*cq_d));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cq_d allocation failed.\n");

    cq_h = (nvshmemi_gic_device_cq_t *)calloc(device->dci.num_eps, sizeof(*cq_h));
    NULL_ERROR_JMP(cq_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cq_h allocation failed.\n");

    for (int i = 0; i < device->dci.num_eps; ++i) {
        uintptr_t tx_wq_d_addr = (uintptr_t)(&dci_d[i]) + offsetof(nvshmemi_gic_device_dci_t, tx_wq);

        gic_get_device_dci(&dci_h[i], device, device->dci.eps[i]);
        dci_h[i].tx_wq.cq = &cq_d[i];

        gic_get_device_cq(&cq_h[i], device->dci.eps[i]->send_cq);
        cq_h[i].cons_head = (uint64_t *)(tx_wq_d_addr + offsetof(nvshmemi_gic_device_wq_t, cons_head));
        cq_h[i].cons_tail = (uint64_t *)(tx_wq_d_addr + offsetof(nvshmemi_gic_device_wq_t, cons_tail));
    }

    status = cuMemcpyHtoDAsync((CUdeviceptr)dci_d,
                               (const void *)dci_h,
                               sizeof(*dci_h) * device->dci.num_eps, 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Copying dci_h to dci_d failed.\n");

    status = cuMemcpyHtoDAsync((CUdeviceptr)cq_d,
                               (const void *)cq_h,
                               sizeof(*cq_h) * device->dci.num_eps, 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Copying cq_h to cq_d failed.\n");

    // Post the device state
    status = cuMemAlloc((CUdeviceptr *)&gic_device_state_d, sizeof(*gic_device_state_d));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "gic_device_state_d allocation failed.\n");

    gic_device_state_h.dcis = dci_d;
    gic_device_state_h.dcts = dct_d;
    gic_device_state_h.cqs = cq_d;
    gic_device_state_h.local_mhandle_head = NULL;
    gic_device_state_h.remote_mhandle_head = NULL;
    gic_device_state_h.ndcis = device->dci.num_eps;
    gic_device_state_h.ndcis_per_sm = device->dci.num_eps_per_sm;
    gic_device_state_h.ndcts_per_pe = device->dct.num_eps;

    assert(gic_nic_buf_location == GIC_MEM_TYPE_GPU || gic_nic_buf_location == GIC_MEM_TYPE_HOST);
    gic_device_state_h.nic_buf_on_gpumem = (gic_nic_buf_location == GIC_MEM_TYPE_GPU);

    status = cuMemcpyHtoDAsync((CUdeviceptr)gic_device_state_d,
                               (const void *)&gic_device_state_h,
                               sizeof(gic_device_state_h), 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Copying gic_device_state to GPU memory failed.\n");

    status = cuStreamSynchronize(nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "stream synchronize failed.\n");

    nvshmemx_get_device_state(&device_state);
    device_state->gic_state = (void *)gic_device_state_d;
    nvshmemi_set_device_state(device_state);

out:
    if (status) {
        if (gic_device_state_d) {
            cuMemFree((CUdeviceptr)gic_device_state_d);
            gic_device_state_d = NULL;
        }
        if (dci_d) cuMemFree((CUdeviceptr)dci_d);
        if (dct_d) cuMemFree((CUdeviceptr)dct_d);
        if (cq_d) cuMemFree((CUdeviceptr)cq_d);
    }
    if (dci_h) free(dci_h);
    if (dct_h) free(dct_h);
    if (cq_h) free(cq_h);
    return status;
}

int nvshmemt_gic_connect_endpoints(nvshmem_transport_t t) {
    int status = 0;

    nvshmemi_gic_device_dct_t *local_dct_handles = NULL;

    nvshmemt_gic_state_t *gic_state = (nvshmemt_gic_state_t *)t->state;
    struct gic_device *device = ((struct gic_device *)gic_state->devices + gic_state->dev_ids[gic_state->selected_dev_id]);
    int portid = gic_state->port_ids[gic_state->selected_dev_id];
    int warp_size;
    int max_threads_per_block;

    // Creating DCT.
    device->dct.num_eps = nvshmemi_options.IB_GPUINITIATED_NUM_DCT;
    if (device->dct.num_eps <= 0) {
        ERROR_JMP(status, EINVAL, out, "NVSHMEM_IB_GPUINITIATED_NUM_DCT must be greater than 0.\n");
    } else if (device->dct.num_eps < 2) {
        WARN_PRINT("Setting NVSHMEM_IB_GPUINITIATED_NUM_DCT lower than 2 may impact performance.\n");
    }

    status = gic_create_dct_shared_objects(device);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_create_dct_shared_objects failed.\n");

    local_dct_handles = (nvshmemi_gic_device_dct_t *)calloc(device->dct.num_eps, sizeof(*local_dct_handles));
    NULL_ERROR_JMP(local_dct_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "allocation of local_dct_handles failed.\n");

    device->dct.dct_handles = (nvshmemi_gic_device_dct_t *)calloc(device->dct.num_eps * nvshmemi_state->npes, 
                                                                sizeof(*device->dct.dct_handles));
    NULL_ERROR_JMP(device->dct.dct_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "allocation of dct_handles failed.\n");

    device->dct.eps = (struct gic_ep **)calloc(device->dct.num_eps, sizeof(*device->dct.eps));
    NULL_ERROR_JMP(device->dct.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "allocation of dct.eps failed.\n");

    for (int i = 0; i < device->dct.num_eps; ++i) {
        status = gic_create_dct(&device->dct.eps[i], device, portid);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_create_dct failed on DCT #%d.\n", i);

        status = gic_get_dct_handle(&local_dct_handles[i], device->dct.eps[i]);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_get_dct_handle failed on DCT #%d.\n", i);
    }

    // Exchange DCT info with other PEs.
    status = nvshmemi_boot_handle.allgather(
        (void *)local_dct_handles, (void *)device->dct.dct_handles,
        sizeof(*local_dct_handles) * device->dct.num_eps, &nvshmemi_boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of dct handles failed.\n");

    // Creating DCI.
    device->dci.num_eps_per_sm = nvshmemi_options.IB_GPUINITIATED_NUM_DCI_PER_SM;
    if (device->dci.num_eps_per_sm <= 0) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "NVSHMEM_IB_GPUINITIATED_NUM_DCI_PER_SM cannot be lower than 1.\n");
    }
    status = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, nvshmemi_state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGetAttribute querying warp size failed.\n");
    status = cuDeviceGetAttribute(&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, nvshmemi_state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGetAttribute querying max threads per block failed.\n");
    if (device->dci.num_eps_per_sm > max_threads_per_block / warp_size) {
        WARN_PRINT("Setting NVSHMEM_IB_GPUINITIATED_NUM_DCI_PER_SM greater than number of warps per block will waste resources unnecessarily.\n");
    }

    device->dci.num_eps = nvshmemi_options.IB_GPUINITIATED_NUM_DCI;
    if (device->dci.num_eps <= 0) {
        int mpc = 0;
        status = cuDeviceGetAttribute(&mpc, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nvshmemi_state->cudevice);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGetAttribute querying multiprocessor count failed.\n");
        device->dci.num_eps = mpc * device->dci.num_eps_per_sm;
    }
    assert(device->dci.num_eps > 0);

    status = gic_create_dci_shared_objects(device);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_create_dci_shared_objects failed.\n");

    device->dci.eps = (struct gic_ep **)calloc(device->dci.num_eps, sizeof(*device->dci.eps));
    NULL_ERROR_JMP(device->dci.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "allocation of dci.eps failed.\n");

    for (int i = 0; i < device->dci.num_eps; ++i) {
        status = gic_create_dci(&device->dci.eps[i], device, portid, i);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_create_dci failed on DCI #%d.\n", i);
    }

    // Transition DCI to RTS.
    for (int i = 0; i < device->dci.num_eps; ++i) {
        status = gic_dci_rst2init(device->dci.eps[i], device, portid);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_dci_rst2init failed on DCI #%d.\n", i);

        status = gic_dci_init2rtr(device->dci.eps[i], device, portid);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_dci_init2rtr failed on DCI #%d.\n", i);

        status = gic_dci_rtr2rts(device->dci.eps[i], device, portid);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_dci_rtr2rts failed on DCI #%d.\n", i);
    }

    // Setup QPs / CQs on GPU.
    status = gic_setup_gpu_state(device);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_setup_gpu_state failed.\n");

out:
    if (status) {
        // TODO: Implement cleanup
    }
    return status;
}

int nvshmemt_gic_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    int status = 0;
    struct gic_mem_handle *handle = (struct gic_mem_handle *)mem_handle;

    INFO(NVSHMEM_TRANSPORT, "ibv_dereg_mr handle %p handle->mr %p", handle, handle->mr);
    status = ftable.dereg_mr((struct ibv_mr *)handle->mr);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");

out:
    return status;
}

int nvshmemt_gic_finalize(nvshmem_transport_t transport) {
    int status = 0;

    nvshmemt_ibv_ftable_fini(&ibv_handle);

    if (transport->state) {
        free(transport->state);
    }

    free(transport);
    return status;
}

int nvshmemt_gic_add_device_remote_mem_handles(nvshmem_transport_t t, int transport_id, nvshmem_mem_handle_t *mem_handles, uint64_t heap_offset, size_t size) {
    int status = 0;

    struct gic_device_mhandle_cache device_mhandle_cache;
    nvshmemi_gic_device_mhandle_t *device_mhandle_h = &device_mhandle_cache.mhandle;
    nvshmemi_gic_device_mhandle_t *device_mhandle_d = NULL;

    CUdeviceptr mhandle_gpu_ptr;

    __be32  *rkeys_h = NULL;
    __be32  *rkeys_d = NULL;

    assert(sizeof(struct gic_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    status = cuMemAlloc((CUdeviceptr *)&device_mhandle_d, sizeof(*device_mhandle_d));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cuMemAlloc failed.\n");

    rkeys_h = (__be32 *)malloc(sizeof(*rkeys_h) * nvshmemi_state->npes);
    NULL_ERROR_JMP(rkeys_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "Unable to allocate rkeys_h.\n");

    status = cuMemAlloc((CUdeviceptr *)&rkeys_d, sizeof(*rkeys_d) * nvshmemi_state->npes);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cuMemAlloc for rkeys_d failed.\n");

    for (int i = 0; i < nvshmemi_state->npes; ++i) {
        // sizeof(struct gic_mem_handle) <= sizeof(nvshmem_mem_handle_t)
        // So, we calculate the pointer with nvshmem_mem_handle_t and convert to gic_mem_handle later.
        struct gic_mem_handle *gmhandle = (struct gic_mem_handle *)&mem_handles[i * NVSHMEM_TRANSPORT_COUNT + transport_id];
        rkeys_h[i] = htobe32(gmhandle->rkey);
    }

    status = cuMemcpyHtoDAsync((CUdeviceptr)rkeys_d,
                               (const void *)rkeys_h,
                               sizeof(*rkeys_h) * nvshmemi_state->npes, 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Copying rkeys_h to rkeys_d failed.\n");

    device_mhandle_h->rkeys = rkeys_d;
    device_mhandle_h->start = heap_offset;
    device_mhandle_h->end = heap_offset + size - 1;
    device_mhandle_h->next = NULL;

    status = cuMemcpyHtoDAsync((CUdeviceptr)device_mhandle_d,
                               (const void *)device_mhandle_h,
                               sizeof(*device_mhandle_d), 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Copying device_mhandle to GPU memory failed.\n");

    device_mhandle_cache.dev_ptr = (CUdeviceptr)device_mhandle_d;

    if (gic_device_remote_mhandles.empty()) {
        assert(gic_device_state_d != NULL);

        mhandle_gpu_ptr = (CUdeviceptr)gic_device_state_d + offsetof(nvshmemi_gic_device_state_t, remote_mhandle_head);
    } else {
        struct gic_device_mhandle_cache *last_mhandle_cache = &gic_device_remote_mhandles.back();
        mhandle_gpu_ptr = last_mhandle_cache->dev_ptr + offsetof(nvshmemi_gic_device_mhandle_t, next);
        last_mhandle_cache->mhandle.next = device_mhandle_d;
    }
    status = cuMemcpyHtoDAsync(mhandle_gpu_ptr,
                               (const void *)&device_mhandle_d,
                               sizeof(device_mhandle_d), 
                               nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "Setting local_mhandle_head in GPU memory failed.\n");

    status = cuStreamSynchronize(nvshmemi_state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "stream synchronize failed.\n");

    gic_device_remote_mhandles.emplace_back(device_mhandle_cache);

out:
    if (status) {
        if (device_mhandle_d) cuMemFree((CUdeviceptr)device_mhandle_d);
        if (rkeys_d) cuMemFree((CUdeviceptr)rkeys_d);
    }
    if (rkeys_h) free(rkeys_h);
    return status;
}

static gic_nic_mapping_memtype_reqeust_t gic_parse_nic_mapping_memtype_request(const char *str) {
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](auto &c) { c = ::tolower(c); });

    if (req == "gpumem")
        return GIC_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM;
    else if (req == "hostmem")
        return GIC_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM;
    else
        return GIC_NIC_MAPPING_MEMTYPE_REQUEST_AUTO;
}

static int gic_check_nic_mapping_memtypes(struct gic_device *device, gic_nic_mapping_memtype_reqeust_t request_memtype) {
    int status = 0;

    bool try_gpumem = ((request_memtype == GIC_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) || (request_memtype == GIC_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM));
    bool try_hostmem = ((request_memtype == GIC_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) || (request_memtype == GIC_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM));

    bool can_use_gpumem = false;
    bool can_use_hostmem = false;

    struct gic_mem_object *mobject = NULL;

    if (try_gpumem) {
        status = gic_gpu_mem_alloc(&mobject, GIC_DBSIZE, GIC_GPAGE_SIZE, false);
        if (status)
            goto out_try_gpumem;

        status = gic_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE);
        if (status)
            goto out_try_gpumem;

        can_use_gpumem = true;

out_try_gpumem:
        if (mobject) {
            if (mobject->has_nic_mapping) gic_mobject_nic_unmap(mobject);
            gic_gpu_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    if (try_hostmem) {
        status = gic_host_mem_alloc(&mobject, GIC_DBSIZE, GIC_GPAGE_SIZE, true);
        if (status)
            goto out_try_hostmem;

        status = gic_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE);
        if (status)
            goto out_try_hostmem;

        can_use_hostmem = true;

out_try_hostmem:
        if (mobject) {
            if (mobject->has_nic_mapping) gic_mobject_nic_unmap(mobject);
            gic_host_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    device->support_nic_buf_on_gpumem = can_use_gpumem;
    device->support_nic_buf_on_hostmem = can_use_hostmem;

    if (!can_use_gpumem && !can_use_hostmem)
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
    
    return 0;
}

static int gic_check_gpu_mapping_nic_uar(struct gic_device *device) {
    int status = 0;
    size_t bf_reg_size;
    uint8_t log_bf_reg_size;
    struct mlx5dv_devx_uar *uar = NULL;
    struct gic_mem_object *mobject = NULL;

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {0,};
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {0,};
    void *cap;

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE |
        (MLX5_CAP_GENERAL << 1) |
        HCA_CAP_OPMOD_GET_CUR
    );

    status = mlx5dv_devx_general_cmd(device->context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out, sizeof(cmd_cap_out));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
    log_bf_reg_size = DEVX_GET(cmd_hca_cap, cap, log_bf_reg_size);

    bf_reg_size = 1LLU << log_bf_reg_size;

    uar = mlx5dv_devx_alloc_uar(device->context, MLX5DV_UAR_ALLOC_TYPE_BF);
    NULL_ERROR_JMP(uar, status, ENOMEM, out, "mlx5dv_devx_alloc_uar failed.\n");

    status = gic_nic_mem_gpu_map(&mobject, uar, bf_reg_size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gic_nic_mem_gpu_map failed.\n");

out:
    if (mobject) gic_nic_mem_gpu_unmap(mobject);
    if (uar) mlx5dv_devx_free_uar(uar);
    return status;
}

int nvshmemt_gic_init(nvshmem_transport_t *t) {
    struct nvshmemt_hca_info hca_list[MAX_NUM_HCAS];
    struct nvshmemt_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];

    int status = 0;
    int exclude_list = 0;
    int hca_list_count = 0;
    int pe_hca_map_count = 0;
    int user_selection = 0;
    int transport_skipped;
    int offset = 0;
    int num_devices = 0;

    struct nvshmem_transport *transport = NULL;
    nvshmemt_gic_state_t *gic_state;
    struct gic_device *device;
    struct ibv_device **dev_list = NULL;

    bool nic_buf_on_gpumem = true;
    bool nic_buf_on_hostmem = true;

    gic_nic_mapping_memtype_reqeust_t nic_mapping_memtype_request;

    transport_skipped = !nvshmemi_options.IB_ENABLE_GPUINITIATED;
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "GPU-initiated communication is disabled by user through environment "
                           "in favor of the %s transport.", nvshmemi_options.REMOTE_TRANSPORT);
        status = NVSHMEMI_ERROR_SKIPPED;
        goto out;
    }
    else {
        INFO(NVSHMEM_INIT, "GPU-initiated communication is enabled by user through environment.",
             nvshmemi_options.REMOTE_TRANSPORT);
    }

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "Unable to allocate transport stuct for gic transport.\n");
    memset(transport, 0, sizeof(struct nvshmem_transport));

    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable)) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                  "Unable to dlopen libibverbs. Skipping devx transport.");
    }

    gic_srq_depth = nvshmemi_options.SRQ_DEPTH;
    gic_qp_depth = GIC_ROUND_UP_POW2_OR_0(nvshmemi_options.QP_DEPTH);
    if (gic_qp_depth <= 0) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "QP depth is not a positive number.\n");
    }
    else if (gic_qp_depth < NVSHMEMI_GIC_MIN_NUM_BATCH_SIZE) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "QP depth must be at least %d.\n", NVSHMEMI_GIC_MIN_NUM_BATCH_SIZE);
    }

    gic_state = (nvshmemt_gic_state_t *)calloc(1, sizeof(nvshmemt_gic_state_t));
    NULL_ERROR_JMP(gic_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "p2p state allocation failed \n");

    dev_list = ftable.get_device_list(&num_devices);
    NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out, "get_device_list failed \n");

    gic_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct gic_device));
    NULL_ERROR_JMP(gic_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "get_device_list failed \n");

    gic_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NULL_ERROR_JMP(gic_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "malloc failed \n");

    gic_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NULL_ERROR_JMP(gic_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "malloc failed \n");
    if (nvshmemi_options.HCA_LIST_provided) {
        user_selection = 1;
        exclude_list = (nvshmemi_options.HCA_LIST[0] == '^');
        hca_list_count = nvshmemt_parse_hca_list(nvshmemi_options.HCA_LIST, hca_list, MAX_NUM_HCAS);
    }

    if (nvshmemi_options.HCA_PE_MAPPING_provided) {
        if (hca_list_count) {
            WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            user_selection = 1;
            pe_hca_map_count = nvshmemt_parse_hca_list(nvshmemi_options.HCA_PE_MAPPING, pe_hca_mapping,
                                                       MAX_NUM_PES_PER_NODE);
        }
    }

    nic_mapping_memtype_request = gic_parse_nic_mapping_memtype_request(nvshmemi_options.IB_GPUINITIATED_FORCE_NIC_BUF_MEMTYPE);

    INFO(NVSHMEM_INIT,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - ");
    for (int i = 0; i < num_devices; i++) {
        device = (struct gic_device *)gic_state->devices + i;
        device->dev = dev_list[i];

        device->context = ftable.open_device(device->dev);
        if (!device->context) {
            INFO(NVSHMEM_INIT, "open_device failed for IB device at index %d", i);
            continue;
        }

        const char *name = ftable.get_device_name(device->dev);
        NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_get_device_name failed \n");
        if (!strstr(name, "mlx5")) {
            WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...\n", name);
            continue;
        }

        status = ftable.query_device(device->context, &device->device_attr);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        if (device->device_attr.vendor_id != MELLANOX_VENDOR_ID ||
            device->device_attr.vendor_part_id < MELLANOX_MIN_DEVICE_ID) {
                WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...\n", name);
                continue;
            }

        status = gic_check_gpu_mapping_nic_uar(device);
        if (status) {
            WARN_PRINT("GPU cannot map UAR of device %s. Skipping...\n", name);
            continue;
        }

        status = gic_check_nic_mapping_memtypes(device, nic_mapping_memtype_request);
        if (status) {
            WARN_PRINT("device %s cannot allocate buffer on the specified memory type. Skipping...\n", name);
            continue;
        }

        INFO(NVSHMEM_INIT,
             "Enumerated IB devices in the system - device id=%d (of %d), name=%s, num_ports=%d", i,
             num_devices, name, device->device_attr.phys_port_cnt);
        int device_used = 0;
        for (int p = 1; p <= device->device_attr.phys_port_cnt; p++) {
            int allowed_device = 1;
            int replicate_count = 1;
            if (hca_list_count) {
                // filter out based on user hca list
                allowed_device = exclude_list;
                for (int j = 0; j < hca_list_count; j++) {
                    if (!strcmp(hca_list[j].name, name)) {
                        if (hca_list[j].port == -1 || hca_list[j].port == p) {
                            hca_list[j].found = 1;
                            allowed_device = !exclude_list;
                        }
                    }
                }
            } else if (pe_hca_map_count) {
                // filter devices based on user hca-pe mapping
                allowed_device = 0;
                for (int j = 0; j < pe_hca_map_count; j++) {
                    if (!strcmp(pe_hca_mapping[j].name, name)) {
                        if (pe_hca_mapping[j].port == -1 || pe_hca_mapping[j].port == p) {
                            allowed_device = 1;
                            pe_hca_mapping[j].found = 1;
                            replicate_count = pe_hca_mapping[j].count;
                        }
                    }
                }
            }

            if (!allowed_device) {
                continue;
            } else {
                status = ftable.query_port(device->context, p, &device->port_attr[p - 1]);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_port_query failed \n");

                // GIC supports IB link layer for now. RoCE is not supported.
                if ((device->port_attr[p - 1].state != IBV_PORT_ACTIVE) 
                    || (device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_INFINIBAND)) { 

                    if (user_selection) {
                        WARN_PRINT(
                            "found inactive port or port with non-IB link layer protocol, "
                            "skipping...\n");
                    }
                    continue;
                }

                status = ftable.query_gid(device->context, p, nvshmemi_options.IB_GID_INDEX, &device->gid[p - 1]);
                NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out, "query_gid failed \n");

                device->pd = ftable.alloc_pd(device->context);
                NULL_ERROR_JMP(device->pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                               "ibv_alloc_pd failed \n");

                for (int k = 0; k < replicate_count; k++) {
                    gic_state->dev_ids[offset] = i;
                    gic_state->port_ids[offset] = p;
                    offset++;
                }

                device_used = 1;
            }
        }

        if (!device_used) {
            status = ftable.close_device(device->context);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_port_query failed \n");
        }
    }
    INFO(NVSHMEM_INIT, "End - Enumerating IB devices in the system");

    gic_state->n_dev_ids = offset;
    INFO(NVSHMEM_INIT,
         "Begin - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))  - ");
    for (int i = 0; i < gic_state->n_dev_ids; i++) {
        INFO(NVSHMEM_INIT,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d",
             i, gic_state->n_dev_ids, gic_state->dev_ids[i], gic_state->port_ids[i]);
        
        device = (struct gic_device *)gic_state->devices + gic_state->dev_ids[i];
        nic_buf_on_gpumem &= device->support_nic_buf_on_gpumem;
        nic_buf_on_hostmem &= device->support_nic_buf_on_hostmem;
    }
    INFO(NVSHMEM_INIT,
         "End - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))");

    if (!gic_state->n_dev_ids) {
        INFO(NVSHMEM_INIT, "no active IB device that supports GPU-initiated communication is found, exiting...");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    assert(nic_buf_on_gpumem || nic_buf_on_hostmem);
    if (nic_buf_on_gpumem) {
        gic_nic_buf_location = GIC_MEM_TYPE_GPU;
        INFO(NVSHMEM_INIT, "NIC buffer will be on GPU memory.");
    } else {
        gic_nic_buf_location = GIC_MEM_TYPE_HOST;
        INFO(NVSHMEM_INIT, "NIC buffer will be on host memory.");
    }

    // print devices that were not found
    if (hca_list_count) {
        for (int j = 0; j < hca_list_count; j++) {
            if (hca_list[j].found != 1) {
                WARN_PRINT("cound not find user specified HCA name: %s port: %d, skipping\n",
                           hca_list[j].name, hca_list[j].port);
            }
        }
    } else if (pe_hca_map_count) {
        // filter devices based on user hca-pe mapping
        for (int j = 0; j < pe_hca_map_count; j++) {
            if (pe_hca_mapping[j].found != 1) {
                WARN_PRINT("cound not find user specified HCA name: %s port: %d, skipping\n",
                           pe_hca_mapping[j].name, pe_hca_mapping[j].port);
            }
        }
    }

    transport->host_ops.get_device_count = nvshmemt_gic_get_device_count;
    transport->host_ops.get_pci_path = nvshmemt_gic_get_pci_path;
    transport->host_ops.can_reach_peer = nvshmemt_gic_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_gic_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_gic_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_gic_release_mem_handle;
    transport->host_ops.show_info = nvshmemt_gic_show_info;
    transport->host_ops.progress = nvshmemt_gic_progress;
    transport->host_ops.finalize = nvshmemt_gic_finalize;
    transport->host_ops.rma = NULL;
    transport->host_ops.amo = NULL;
    transport->host_ops.fence = NULL;
    transport->host_ops.quiet = NULL;
    transport->host_ops.enforce_cst = NULL;
    transport->host_ops.add_device_remote_mem_handles = nvshmemt_gic_add_device_remote_mem_handles;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->state = (void *)gic_state;
    transport->is_successfully_initialized = true;

    *t = transport;

out:
    // TODO: Implement cleanup
    return status;
}
