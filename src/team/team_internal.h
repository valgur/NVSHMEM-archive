#ifndef NVSHMEMI_TEAM_INTERNAL_H
#define NVSHMEMI_TEAM_INTERNAL_H

static inline __host__ __device__ size_t get_fcollect_psync_len_per_team() {
#ifdef __CUDA_ARCH__
    size_t fcollect_ll_threshold = nvshmemi_device_state_d.fcollect_ll_threshold;
    size_t fcollect_sync_size =
        (2 * 2 * nvshmemi_device_state_d.npes * fcollect_ll_threshold) / sizeof(long);
#else
    size_t fcollect_ll_threshold = nvshmemi_device_state.fcollect_ll_threshold;
    size_t fcollect_sync_size =
        (2 * 2 * nvshmemi_state->npes * fcollect_ll_threshold) / sizeof(long);
    assert(fcollect_ll_threshold % sizeof(long) == 0);
#endif

    return fcollect_sync_size;
}

static inline __host__ __device__ size_t get_psync_len_per_team() {
    size_t fcollect_sync_size = get_fcollect_psync_len_per_team();
    /* sync: Two buffers are used - one for sync/barrier collective ops, the second one during team
       split operation reduce: Two pWrk's are used alternatively across consecutive reduce calls,
       this is to avoid having to put a barrier in between bcast: The buffer is split to do multiple
       consecutive broadcast, when all buffers are used, a barrier is called and then again we begin
       from the start of the buffer alltoall: We only need two longs per PE. One to store the
       expected value of the psync locally and another for communication. fcollect: Two sets of
       buffer are used to alternate between - same way as in reduce. The other fator of 2 is
                 because when using LL double the space is needed to fuse flag with data */

    return (2 * NVSHMEMI_SYNC_SIZE + 2 * NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE +
            NVSHMEMI_BCAST_SYNC_SIZE + fcollect_sync_size + 2 * NVSHMEMI_ALLTOALL_SYNC_SIZE);
}
#endif /* NVSHMEMI_TEAM_INTERNAL_H */
