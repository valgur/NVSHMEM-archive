/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef PROXY_HOST_H
#define PROXY_HOST_H

#define CHANNEL_COUNT 1
#define COUNTER_TO_FLAG(state, counter) ((uint8_t)(!((counter >> state->channel_bufsize_log) & 1)))
#define WRAPPED_CHANNEL_BUF(state, ch, counter) (ch->buf + (counter & (state->channel_bufsize - 1)))

enum {
    PROXY_QUIET_STATUS_CHANNELS_INACTIVE = 0,
    PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS,
    PROXY_QUIET_STATUS_CHANNELS_DONE
};

enum { PROXY_CST_STATUS_CHANNELS_INACTIVE = 0, PROXY_CST_STATUS_CHANNELS_ACTIVE };

#endif
