#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

int shared_memory_create(const char *name, size_t sz, nvshmemi_shared_memory_info *info);
int shared_memory_open(const char *name, size_t sz, nvshmemi_shared_memory_info *info);
void shared_memory_close(char *shm_name, nvshmemi_shared_memory_info *info);

#endif
