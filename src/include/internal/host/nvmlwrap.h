#ifndef NVML_WRAP_H
#define NVML_WRAP_H

#include <nvml.h>
#include "modules/transport/transport.h"

struct nvml_function_table {
    nvmlReturn_t (*nvmlInit)(void);
    nvmlReturn_t (*nvmlShutdown)(void);
    nvmlReturn_t (*nvmlDeviceGetHandleByPciBusId)(const char *pciBusId, nvmlDevice_t *device);
    nvmlReturn_t (*nvmlDeviceGetP2PStatus)(nvmlDevice_t device1, nvmlDevice_t device2,
                                           nvmlGpuP2PCapsIndex_enum caps,
                                           nvmlGpuP2PStatus_t *p2pStatus);
    nvmlReturn_t (*nvmlDeviceGetGpuFabricInfoV)(nvmlDevice_t device, nvmlGpuFabricInfoV_t *info);
};

int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle);
void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle);

#endif