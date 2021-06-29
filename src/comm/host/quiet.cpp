/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"

void nvshmem_quiet(void) {
    NVTX_FUNC_RANGE_IN_GROUP(MEMORDER);
    NVSHMEM_CHECK_STATE_AND_INIT();

    int status = 0;

    int tbitmap = nvshmemi_state->transport_bitmap;
    if (nvshmemi_state->npes_node > 1) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            CUstream custrm = nvshmemi_state->custreams[s];
            status = cuStreamSynchronize(custrm);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_quiet() failed \n");
        }
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
