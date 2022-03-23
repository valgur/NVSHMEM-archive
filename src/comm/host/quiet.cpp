/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"

void nvshmemi_call_proxy_quiet_entrypoint(cudaStream_t cstrm);
#ifdef __cplusplus
extern "C" {
#endif
void nvshmemx_quiet_on_stream(cudaStream_t cstrm);
#ifdef __cplusplus
}
#endif

void nvshmem_quiet(void) {
    NVTX_FUNC_RANGE_IN_GROUP(MEMORDER);
    NVSHMEMI_CHECK_INIT_STATUS();

    int status = 0;

    int tbitmap = nvshmemi_state->transport_bitmap;
    if (nvshmemi_state->npes_node > 1) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            CUstream custrm = nvshmemi_state->custreams[s];
            status = cuStreamSynchronize(custrm);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_quiet() failed \n");
        }
        nvshmemi_state->used_internal_streams = 0;
    }

    for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
        if (tbitmap & 1) {
                struct nvshmem_transport *tcurr =
                    ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
                for (int k = 0; k < nvshmemi_state->npes; k++) {
                    if (nvshmemi_state->quiet[j]) {
                        status = nvshmemi_state->quiet[j](tcurr, k, 0);
                    }
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_quiet() failed \n");
                }
        }
        tbitmap >>= 1;
    }
out:
    return;
}

void nvshmemi_quiesce_internal_streams(cudaStream_t cstrm) {
    int status = 0;
    if (nvshmemi_state->npes_node > 1 && nvshmemi_state->used_internal_streams) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            CUstream custrm = nvshmemi_state->custreams[s];
            CUevent cuev = nvshmemi_state->cuevents[s];
            status = cuEventRecord(cuev, custrm);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                         "nvshmem_quiet_on_stream() failed \n");
            status = cuStreamWaitEvent((CUstream)cstrm, cuev, 0);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                         "nvshmem_quiet_on_stream() failed \n");
        }
        nvshmemi_state->used_internal_streams = 0;
    }
out:
    return;
}

void nvshmemx_quiet_on_stream(cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(QUIET_ON_STREAM);
    NVSHMEMI_CHECK_INIT_STATUS();

    int tbitmap = nvshmemi_state->transport_bitmap;
    nvshmemi_quiesce_internal_streams(cstrm);

    for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
        if (tbitmap & 1) {
            if (j == NVSHMEM_TRANSPORT_ID_IBRC ||
                j == NVSHMEM_TRANSPORT_ID_UCX ||
                j == NVSHMEM_TRANSPORT_ID_IBDEVX ||
                j == NVSHMEM_TRANSPORT_ID_FABRIC) {
				nvshmemi_call_proxy_quiet_entrypoint(cstrm);
            }
            tbitmap >>= 1;
        }
    }

    return;
}
