/*
 * * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "nvshmem_internal.h"

void nvshmem_quiet(void) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    int status = 0;

    int tbitmap = nvshmem_state->transport_bitmap;
    if (nvshmem_state->npes_node > 1) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            CUstream custrm = nvshmem_state->custreams[s];
            status = cuStreamSynchronize(custrm);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_quiet() failed \n");
        }
    }

    for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
        if (tbitmap & 1) {
            if (j == NVSHMEM_TRANSPORT_ID_IBRC) {
                struct nvshmem_transport *tcurr =
                    ((nvshmem_transport_t *)nvshmem_state->transports)[j];
                int ep_count = 1;
                if (tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED) {
                    ep_count = tcurr->ep_count * nvshmem_state->npes;
                }
                for (int k = 0; k < ep_count; k++) {
                    if (!tcurr->ep[k]) continue;
                    status = nvshmem_state->quiet[j](tcurr->ep[k]);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_quiet() failed \n");
                }
            }
            tbitmap >>= 1;
        }
    }
out:
    return;
}
