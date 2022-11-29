/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <assert.h>
#include "nvshmemx_error.h"
#include "util.h"
#include "topo.h"
#include "transport.h"
#include <sys/types.h>
#include <unistd.h>

int (*nvshmemi_transport_init_op[NVSHMEM_TRANSPORT_COUNT])(nvshmem_transport_t *transport);

int nvshmemi_transport_show_info(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; ++i) {
        for (size_t j = 0; j < state->handles.size(); j++) {
            status = transports[i]->host_ops.show_info(
                state->handles[j].data(), i, NVSHMEM_TRANSPORT_COUNT, state->npes, state->mype);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport show info failed \n");
        }
    }
out:
    return status;
}

void nvshmemi_add_transport(int id, int (*init_op)(nvshmem_transport_t *)) {
    nvshmemi_transport_init_op[id] = init_op;
}

void nvshmemi_transports_preinit() {
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_P2P, nvshmemt_p2p_init);
#ifdef NVSHMEM_IBRC_SUPPORT
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_IBRC, nvshmemt_ibrc_init);
#endif
#ifdef NVSHMEM_UCX_SUPPORT
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_UCX, nvshmemt_ucx_init);
#endif
#ifdef NVSHMEM_IBDEVX_SUPPORT
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_IBDEVX, nvshmemt_ibdevx_init);
#endif
#ifdef NVSHMEM_LIBFABRIC_SUPPORT
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_FABRIC, nvshmemt_libfabric_init);
#endif
#ifdef NVSHMEM_IBGDA_SUPPORT
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_GIC, nvshmemt_gic_init);
#endif
}

int nvshmemi_transport_init(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = NULL;

    nvshmemi_transports_preinit();

    if (!state->transports)
        state->transports =
            (nvshmem_transport_t *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(nvshmem_transport_t));

    transports = (nvshmem_transport_t *)state->transports;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (transports[i] == NULL) {
            if (nvshmemi_transport_init_op[i]) {
                status = nvshmemi_transport_init_op[i](transports + i);
                if (status) {
                    INFO(NVSHMEM_INIT, "init failed for transport: %d", i);
                    status = 0;
                    continue;
                }

                transports[i]->cap = (int *)calloc(state->npes, sizeof(int));
                INFO(NVSHMEM_INIT, "cap array for transport %d : %p", i, transports[i]->cap);
            }
        }
    }

    return status;
}

int nvshmemi_transport_finalize(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_transport_finalize");
    int status = 0;
    nvshmem_transport_t *transports = NULL;
    ;

    if (!state->transports) return 0;

    transports = (nvshmem_transport_t *)state->transports;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (transports[i] && nvshmemi_transport_init_op[i]) {
            status = transports[i]->host_ops.finalize(transports[i]);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport finalize failed \n");
        }
    }
out:
    return status;
}

int nvshmemi_setup_connections(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_transport_t tcurr;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        // assumes symmetry of transport list at all PEs
        if ((state->transport_bitmap) & (1 << i)) {
            tcurr = transports[i];
            if (!(tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED)) continue;
            status = tcurr->host_ops.connect_endpoints(tcurr);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "endpoint connection failed \n");
            status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "barrier failed \n");
        }
    }

out:
    return status;
}
