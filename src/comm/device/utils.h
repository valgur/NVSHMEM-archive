/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _COMM_DEVICE_UTILS_H
#define _COMM_DEVICE_UTILS_H

#define NTOH64(x) *x = ((*(x) & 0xFF00000000000000) >> 56 | \
                       (*(x) & 0x00FF000000000000) >> 40 | \
                       (*(x) & 0x0000FF0000000000) >> 24 | \
                       (*(x) & 0x000000FF00000000) >> 8 | \
                       (*(x) & 0x00000000FF000000) << 8 | \
                       (*(x) & 0x0000000000FF0000) << 24 | \
                       (*(x) & 0x000000000000FF00) << 40 | \
                       (*(x) & 0x00000000000000FF) << 56)

#define NTOH32(x) *x = ((*(x) & 0xFF000000) >> 24 | \
                       (*(x) & 0x00FF0000) >> 8 | \
                       (*(x) & 0x0000FF00) << 8 | \
                       (*(x) & 0x000000FF) << 24)


# define BSWAP64(x) \
    ((((x) & 0xff00000000000000ull) >> 56)                       \
     | (((x) & 0x00ff000000000000ull) >> 40)                     \
     | (((x) & 0x0000ff0000000000ull) >> 24)                     \
     | (((x) & 0x000000ff00000000ull) >> 8)                      \
     | (((x) & 0x00000000ff000000ull) << 8)                      \
     | (((x) & 0x0000000000ff0000ull) << 24)                     \
     | (((x) & 0x000000000000ff00ull) << 40)                     \
     | (((x) & 0x00000000000000ffull) << 56))

#define BSWAP32(x) \
    ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |           \
     (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))

#define HTOBE64(x) BSWAP64(x)
#define HTOBE32(x) BSWAP32(x)

#ifndef MIN
    #define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#endif
