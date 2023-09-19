/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef PROXY_COMMON_H
#define PROXY_COMMON_H

#include "internal/common/nvshmem_internal.h"
#include <pthread.h>
#include <assert.h>
#include <stdint.h>
#include "common/nvshmem_common_proxy.h"
#include "modules/transport/transport.h"

typedef struct {
    struct proxy_state *state;
    int stop;
} proxy_progress_params_t;

typedef struct proxy_state {
    int *transport_id;
    int transport_bitmap;
    struct nvshmem_transport **transport;
    int quiet_in_progress;
    int cst_in_progress;
    int quiet_ack_count;
    uint64_t channel_bufsize_log;
    uint64_t channel_bufsize;
    int channel_count;
    proxy_channel_t *channels;
    proxy_channel_t *channels_device;
    uint64_t channel_g_bufsize;
    int channel_in_progress;
    pthread_t progress_thread;
    proxy_progress_params_t progress_params;
    nvshmemi_state_t *nvshmemi_state;
    int *quiet_incoming_in_progress_pe;
    cudaStream_t stream;
    cudaStream_t queue_stream_out;
    cudaStream_t queue_stream_in;
    cudaEvent_t cuev;
    int finalize_count;
    int issued_get;
    nvshmemi_timeout_t *nvshmemi_timeout;
    bool is_consistency_api_supported;
    int gdr_device_native_ordering;
    int *global_exit_request_state;
    int *global_exit_code;
} proxy_state_t;

int nvshmemi_proxy_prep_minimal_state(proxy_state_t *state);
int nvshmemi_proxy_setup_device_channels(proxy_state_t *state);
#endif /* PROXY_COMMON_H */
