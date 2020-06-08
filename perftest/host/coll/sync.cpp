/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "coll_test.h"

int main(int c, char *v[]) {
    int status = 0;
    int mype, npes;
    size_t size = NVSHMEM_BARRIER_SYNC_SIZE * sizeof(long);
    char *buffer = NULL;
    int iters = 0;
    int skip = MAX_SKIP;
    struct timeval t_start, t_stop;
    double latency = 0;
    int PE_start = 0;
    int logPE_stride = 0;
    int PE_size;
    long *pSync = NULL;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    PE_size = npes;

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    buffer = (char *)nvshmem_malloc(size);
    if (!buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }
    pSync = (long *)buffer;

    latency = 0;
    for (iters = 0; iters < MAX_ITERS + skip; iters++) {
        if (iters >= skip) gettimeofday(&t_start, NULL);

        nvshmem_sync(PE_start, logPE_stride, PE_size, pSync);

        if (iters >= skip) {
            gettimeofday(&t_stop, NULL);
            latency +=
                ((t_stop.tv_usec - t_start.tv_usec) + (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));
        }
    }

    if (0 == mype) printf("%s\t\t%lf\n", "latency (us)", (latency / MAX_ITERS));

    nvshmem_barrier_all();

    nvshmem_free(buffer);

    finalize_wrapper();

out:
    return status;
}
