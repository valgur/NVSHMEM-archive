/****
 * Copyright (c) 2016-2020, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

/* NVSHMEMI_ENV_DEF( name, kind, default, category, short description )
 *
 * Kinds: long, size, bool, string
 * Categories: NVSHMEMI_ENV_CAT_OPENSHMEM, NVSHMEMI_ENV_CAT_OTHER,
 *             NVSHMEMI_ENV_CAT_COLLECTIVES, NVSHMEMI_ENV_CAT_TRANSPORT,
 *             NVSHMEMI_ENV_CAT_HIDDEN
 */

/*
 * Preincluded header requirements: nvshmem_internal.h
 */

NVSHMEMI_ENV_DEF(VERSION, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print library version at startup")
NVSHMEMI_ENV_DEF(INFO, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print environment variable options at startup")
NVSHMEMI_ENV_DEF(INFO_HIDDEN, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Print hidden environment variable options at startup")
NVSHMEMI_ENV_DEF(SYMMETRIC_SIZE, size, (size_t)(SYMMETRIC_SIZE_DEFAULT), NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Specifies the size (in bytes) of the symmetric heap memory per PE. The resulting size is implementation-defined and must be at least as large as the integer ceiling of the product of the numeric prefix and the scaling factor. The allowed character suffixes for the scaling factor are as follows:\n"
                 "\n"
                 "  *  k or K multiplies by 2^10 (kibibytes)\n"
                 "  *  m or M multiplies by 2^20 (mebibytes)\n"
                 "  *  g or G multiplies by 2^30 (gibibytes)\n"
                 "  *  t or T multiplies by 2^40 (tebibytes)\n"
                 "\n"
                 "For example, string '20m' is equivalent to the integer value 20971520, or 20 mebibytes. Similarly the string '3.1M' is equivalent to the integer value 3250586. Only one multiplier is recognized and any characters following the multiplier are ignored, so '20kk' will not produce the same result as '20m'. Usage of string '.5m' will yield the same result as the string '0.5m'.\n"
                 "An invalid value for ``NVSHMEM_SYMMETRIC_SIZE`` is an error, which the NVSHMEM library shall report by either returning a nonzero value from ``nvshmem_init_thread`` or causing program termination.")
NVSHMEMI_ENV_DEF(DEBUG, string, "", NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Set to enable debugging messages.\n"
                 "Optional values: VERSION, WARN, INFO, ABORT, TRACE")

/** Bootstrap **/

NVSHMEMI_ENV_DEF(BOOTSTRAP, string, "PMI", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the default bootstrap that should be used to initialize NVSHMEM.\n"
                 "Allowed values: PMI, MPI, SHMEM, plugin")

#if   defined(NVSHMEM_DEFAULT_PMIX)
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMIX"
#elif defined(NVSHMEM_DEFAULT_PMI2)
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMI-2"
#else
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMI"
#endif

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI, string, NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT, NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the PMI bootstrap that should be used to initialize NVSHMEM.\n"
                 "Allowed values: PMI, PMI-2, PMIX")

NVSHMEMI_ENV_DEF(BOOTSTRAP_TWO_STAGE, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Ignore CUDA device setting during initialization, forcing two-state init")

#undef NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT

NVSHMEMI_ENV_DEF(BOOTSTRAP_PLUGIN, string, "", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the bootstrap plugin file to load")

/** Debugging **/

NVSHMEMI_ENV_DEF(DEBUG_SUBSYS, string, "", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Comma separated list of debugging message sources. Prefix with '^' to exclude.\n"
                 "Values: INIT, COLL, P2P, PROXY, TRANSPORT, MEM, BOOTSTRAP, TOPO, UTIL, ALL")
NVSHMEMI_ENV_DEF(DEBUG_FILE, string, "", NVSHMEMI_ENV_CAT_OTHER,
                 "Debugging output filename, may contain %h for hostname and %p for pid")
NVSHMEMI_ENV_DEF(ENABLE_ERROR_CHECKS, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Enable error checks")

NVSHMEMI_ENV_DEF(MAX_TEAMS, long, 20l, NVSHMEMI_ENV_CAT_OTHER,
                 "Maximum number of simultaneous teams allowed")

NVSHMEMI_ENV_DEF(MAX_P2P_GPUS, int, 128, NVSHMEMI_ENV_CAT_OTHER,
                 "Maximum number of P2P GPUs")
NVSHMEMI_ENV_DEF(MAX_MEMORY_PER_GPU, size, (size_t)((size_t)128 * (1 << 30)), NVSHMEMI_ENV_CAT_OTHER,
                 "Maximum memory per GPU")
#if defined(NVSHMEM_PPC64LE)
NVSHMEMI_ENV_DEF(DISABLE_CUDA_VMM, bool, true, NVSHMEMI_ENV_CAT_OTHER,
                 "Disable use of CUDA VMM for P2P memory mapping (By default, CUDA VMM is enabled "
                 "on x86 and disabled on P9. CUDA VMM feature in NVSHMEM requires CUDA RT version and "
                 "CUDA Driver version to be greater than or equal to 11.3.")
#else
NVSHMEMI_ENV_DEF(DISABLE_CUDA_VMM, bool, false, NVSHMEMI_ENV_CAT_OTHER,
                 "Disable use of CUDA VMM for P2P memory mapping. By default, CUDA VMM is enabled "
                 "on x86 and disabled on P9. CUDA VMM feature in NVSHMEM requires CUDA RT version and "
                 "CUDA Driver version to be greater than or equal to 11.3.")
#endif
NVSHMEMI_ENV_DEF(DISABLE_P2P, bool, false, NVSHMEMI_ENV_CAT_OTHER,
                 "Disable P2P connectivity of GPUs even when available")
NVSHMEMI_ENV_DEF(CUMEM_GRANULARITY, size, (size_t)((size_t)1 << 29), NVSHMEMI_ENV_CAT_OTHER,
                 "Granularity for ``cuMemAlloc``/``cuMemCreate``")

NVSHMEMI_ENV_DEF(BYPASS_ACCESSIBILITY_CHECK, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass peer GPU accessbility checks")

#if defined(NVSHMEM_PPC64LE)
NVSHMEMI_ENV_DEF(CUDA_LIMIT_STACK_SIZE, size, 0, NVSHMEMI_ENV_CAT_OTHER,
                 "Specify limit on stack size of each GPU thread")
#endif

/** General Collectives **/

NVSHMEMI_ENV_DEF(DISABLE_NCCL, bool, false, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Disable use of NCCL for collective operations")
NVSHMEMI_ENV_DEF(BARRIER_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for barriers")
NVSHMEMI_ENV_DEF(BARRIER_TG_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for thread group barriers")
NVSHMEMI_ENV_DEF(REDUCE_RECEXCH_KVAL, int, 2, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Radix of the recursive exchange reduction algorithm")
NVSHMEMI_ENV_DEF(FCOLLECT_LL_THRESHOLD, size, (size_t)(1 << 11),
                 NVSHMEMI_ENV_CAT_COLLECTIVES, "Message size threshold up to which "
                                               "fcollect LL algo will be used")
NVSHMEMI_ENV_DEF(BCAST_LL_THRESHOLD, size, (size_t)(1 << 11),
                 NVSHMEMI_ENV_CAT_COLLECTIVES, "Message size threshold up to which "
                                               "broadcast LL algo  will be used")

/** CPU Collectives **/

NVSHMEMI_ENV_DEF(RDX_NUM_TPB, int, 32, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Number of threads per block used for reduction purposes")

/** Transport **/

#ifdef NVSHMEM_DEFAULT_UCX
#define NVSHMEMI_ENV_TRANSPORT_DEFAULT "ucx"
#else
#define NVSHMEMI_ENV_TRANSPORT_DEFAULT "ibrc"
#endif

NVSHMEMI_ENV_DEF(ASSERT_ATOMICS_SYNC, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush on wait_until at target")
NVSHMEMI_ENV_DEF(REMOTE_TRANSPORT, string, NVSHMEMI_ENV_TRANSPORT_DEFAULT, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Selected transport for remote operations: ibrc, ucx, libfabric, ibdevx, none")
NVSHMEMI_ENV_DEF(DISABLE_IB_NATIVE_ATOMICS, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Disable use of InfiniBand native atomics")
NVSHMEMI_ENV_DEF(DISABLE_GDRCOPY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Disable use of GDRCopy in IB RC Transport")
NVSHMEMI_ENV_DEF(BYPASS_FLUSH, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush in proxy when enforcing consistency")
NVSHMEMI_ENV_DEF(ENABLE_NIC_PE_MAPPING, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "When not set or set to 0, a PE is assigned the NIC on the node that is "
                 "closest to it by distance. When set to 1, NVSHMEM either assigns NICs to "
                 "PEs on a round-robin basis or uses ``NVSHMEM_HCA_PE_MAPPING`` or "
                 "``NVSHMEM_HCA_LIST`` when they are specified.")
NVSHMEMI_ENV_DEF(IB_GID_INDEX, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Source GID Index for ROCE")
NVSHMEMI_ENV_DEF(IB_TRAFFIC_CLASS, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Traffic calss for ROCE")
NVSHMEMI_ENV_DEF(IB_SL, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Service level to use over IB/ROCE")

NVSHMEMI_ENV_DEF(HCA_LIST, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Comma-separated list of HCAs to use in the NVSHMEM application. Entries "
                 "are of the form ``hca_name:port``, e.g. ``mlx5_1:1,mlx5_2:2`` and entries "
                 "prefixed by ^ are excluded. ``NVSHMEM_ENABLE_NIC_PE_MAPPING`` must be set to "
                 "1 for this variable to be effective.")

NVSHMEMI_ENV_DEF(HCA_PE_MAPPING, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies mapping of HCAs to PEs as a comma-separated list. Each entry "
                 "in the comma separated list is of the form ``hca_name:port:count``.  For "
                 "example, ``mlx5_0:1:2,mlx5_0:2:2`` indicates that PE0, PE1 are mapped to "
                 "port 1 of mlx5_0, and PE2, PE3 are mapped to port 2 of mlx5_0. "
                 "``NVSHMEM_ENABLE_NIC_PE_MAPPING`` must be set to 1 for this variable to be "
                 "effective.")
NVSHMEMI_ENV_DEF(QP_DEPTH, int, 1024, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Number of WRs in QP")
NVSHMEMI_ENV_DEF(SRQ_DEPTH, int, 16384, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Number of WRs in SRQ")

NVSHMEMI_ENV_DEF(DISABLE_LOCAL_ONLY_PROXY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                "When running on an NVLink-only configuaration (No-IB, No-UCX), completely disable "
                "the proxy thread. This will disable device side global exit and device side wait "
                "timeout polling (enabled by ``NVSHMEM_TIMEOUT_DEVICE_POLLING`` build-time variable) "
                "because these are processed by the proxy thread.")

NVSHMEMI_ENV_DEF(LIBFABRIC_PERSONA, string, "cxi", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Set the feature set persona for the libfabric transport: cxi, verbs")

/** Runtime optimimzations **/
NVSHMEMI_ENV_DEF(PROXY_REQUEST_BATCH_MAX, int, 32, NVSHMEMI_ENV_CAT_OTHER,
                 "Maxmum number of requests that the proxy thread processes in a single iteration "
                 "of the progress loop.")

/** NVTX instrumentation **/
NVSHMEMI_ENV_DEF(NVTX, string, "off", NVSHMEMI_ENV_CAT_NVTX,
                 "Set to enable NVTX instrumentation. Accepts a comma separated list of "
                 "instrumentation groups. By default the NVTX instrumentation is disabled.")

#ifdef NVSHMEM_GPUINITIATED_SUPPORT
/** GPU-initiated communication **/
NVSHMEMI_ENV_DEF(IB_GPUINITIATED_NUM_DCT, int, 2, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of DCT QPs used in GPU-initiated communication transport.")
NVSHMEMI_ENV_DEF(IB_GPUINITIATED_NUM_DCI, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Total number of DCI QPs used in GPU-initiated communication transport. "
                 "Set to 0 or a negative number to use automatic configuration.")
NVSHMEMI_ENV_DEF(IB_GPUINITIATED_NUM_DCI_PER_SM, int, 1, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of exclusive DCI QPs assigned to each SM.")
NVSHMEMI_ENV_DEF(IB_GPUINITIATED_FORCE_NIC_BUF_MEMTYPE, string, "auto", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Force NIC buffer memory type. Valid choices are: gpumem, hostmem. "
                 "For other values, use auto discovery (default).")
NVSHMEMI_ENV_DEF(IB_ENABLE_GPUINITIATED, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Set to enable GPU-initiated communication transport.")
#endif
