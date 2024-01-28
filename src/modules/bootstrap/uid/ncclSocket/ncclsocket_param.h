/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_PARAM_H_
#define NCCL_SOCKET_PARAM_H_

#include <stdint.h>
#include <stdlib.h>

static inline const char *ncclGetEnv(const char *name) {
  return getenv(name);
}

#endif