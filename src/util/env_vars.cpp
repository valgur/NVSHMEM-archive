/****
 * Copyright (c) 2016-2021, NVIDIA Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>
#include <inttypes.h>
#include <errno.h>
#include <math.h>

#include "nvshmem_nvtx.hpp"
#include "util.h"
#include "nvshmem_internal.h"

struct nvshmemi_options_s nvshmemi_options;

/* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
static int nvshmemi_atol_scaled(const char *str, nvshmemi_env_size *out) {
    int scale, n;
    double p = -1.0;
    char f;

    n = sscanf(str, "%lf%c", &p, &f);

    if (n == 2) {
        switch (f) {
            case 'k':
            case 'K':
                scale = 10;
                break;
            case 'm':
            case 'M':
                scale = 20;
                break;
            case 'g':
            case 'G':
                scale = 30;
                break;
            case 't':
            case 'T':
                scale = 40;
                break;
            default:
                return 1;
        }
    }
    else if (p < 0) {
        return 1;
    } else
        scale = 0;

    *out = (nvshmemi_env_size) ceil(p * (1lu << scale));
    return 0;
}

static long nvshmemi_errchk_atol(const char *s) {
    long val;
    char *e;
    errno = 0;

    val = strtol(s,&e,0);
    if (errno != 0 || e == s) {
        ERROR_PRINT("Environment variable conversion failed (%s)\n", s);
    }

    return val;
}

static const char *nvshmemi_getenv_helper(const char *prefix, const char *name) {
    char *env_name;
    const char *env_value = NULL;
    size_t len;
    int ret;

    len = strlen(prefix) + 1 /* '_' */ + strlen(name) + 1 /* '\0' */;
    env_name = (char *)alloca(len);
    ret = snprintf(env_name, len, "%s_%s", prefix, name);
    if (ret < 0)
        WARN_PRINT("Error in sprintf: %s_%s\n", prefix, name);
    else
        env_value = (const char *) getenv(env_name);

    return env_value;
}

static const char *nvshmemi_getenv(const char* name) {
    const char *env_value;

    env_value = nvshmemi_getenv_helper("NVSHMEM", name);
    if (env_value != NULL) return env_value;

    return NULL;
}

static int nvshmemi_getenv_string(const char *name, nvshmemi_env_string default_val,
                                  nvshmemi_env_string *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? env : default_val;
    return 0;
}

static int nvshmemi_getenv_int(const char *name, nvshmemi_env_int default_val,
                                nvshmemi_env_int *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? (int) nvshmemi_errchk_atol(env) : default_val;
    return 0;
}

static int nvshmemi_getenv_long(const char *name, nvshmemi_env_long default_val,
                                nvshmemi_env_long *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? nvshmemi_errchk_atol(env) : default_val;
    return 0;
}

static int nvshmemi_getenv_size(const char *name, nvshmemi_env_size default_val,
                                nvshmemi_env_size *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    if (*provided) {
        int ret = nvshmemi_atol_scaled(env, out);
        if (ret) {
            WARN_PRINT("Invalid size in environment variable '%s' (%s)\n", name, env);
            return ret;
        }
    }
    else
        *out = default_val;
    return 0;
}

static int
nvshmemi_getenv_bool(const char *name, nvshmemi_env_bool default_val,
                     nvshmemi_env_bool *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);

    if (*provided && (env[0] == '0' ||
                      env[0] == 'N' || env[0] == 'n' ||
                      env[0] == 'F' || env[0] == 'f')) {
        *out = false;
    }
    else if (*provided) {
        /* The default behavior specified by OpenSHMEM is to enable boolean
         * options whenever the environment variable is set */
        *out = true;
    }
    else {
        *out = default_val;
    }

    return 0;
}

int nvshmemi_options_init(void) {
    int ret;
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                                     \
    ret = nvshmemi_getenv_##KIND(#NAME, DEFAULT, &(nvshmemi_options.NAME),                              \
                                 &(nvshmemi_options.NAME##_provided));                                  \
    if (ret) return ret;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    return 0;
}

static void nvshmemi_options_print_heading(const char *h, int style) {
    switch(style) {
        case NVSHMEMI_OPTIONS_STYLE_INFO:
            printf("%s:\n", h);
            break;
        case NVSHMEMI_OPTIONS_STYLE_RST:
            printf("%s\n", h);
            for (const char *c = h; *c != '\0'; c++)
                putchar('~');
            printf("\n\n");
            break;
        default:
            assert(0); //FIXME
    }
}

#define NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, DESIRED_CAT, STYLE)               \
    if (CATEGORY == DESIRED_CAT) {                                                                              \
        switch(STYLE) {                                                                                         \
            char *desc_wrapped;                                                                                 \
            case NVSHMEMI_OPTIONS_STYLE_INFO:                                                                   \
                desc_wrapped = nvshmemu_wrap(SHORT_DESC, NVSHMEMI_WRAPLEN, "\t", 1);                            \
                printf("  NVSHMEM_%-20s " NVSHPRI_##KIND " (type: %s, default: " NVSHPRI_##KIND ")\n\t%s\n",    \
                       #NAME, NVSHFMT_##KIND(nvshmemi_options.NAME), #KIND, NVSHFMT_##KIND(DEFAULT), desc_wrapped);\
                free(desc_wrapped);                                                                             \
                break;                                                                                          \
            case NVSHMEMI_OPTIONS_STYLE_RST:                                                                    \
                desc_wrapped = nvshmemu_wrap(SHORT_DESC, NVSHMEMI_WRAPLEN, NULL, 0);                            \
                printf(".. c:var:: NVSHMEM_%s\n", #NAME);                                                       \
                printf("\n");                                                                                   \
                printf("| *Type: %s*\n", #KIND);                                                                \
                printf("| *Default: " NVSHPRI_##KIND "*\n", NVSHFMT_##KIND(DEFAULT));                           \
                printf("\n");                                                                                   \
                printf("%s\n", desc_wrapped);                                                                   \
                printf("\n");                                                                                   \
                free(desc_wrapped);                                                                             \
                break;                                                                                          \
            default:                                                                                            \
                assert(0); /* FIXME */                                                                          \
        }                                                                                                       \
    }

void nvshmemi_options_print(int style) {
    nvshmemi_options_print_heading("Standard options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_OPENSHMEM, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Bootstrap options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Additional options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_OTHER, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Collectives options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_COLLECTIVES, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Transport options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_TRANSPORT, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    if (nvshmemi_options.INFO_HIDDEN) {
        nvshmemi_options_print_heading("Hidden options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_HIDDEN, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
        printf("\n");
    }

#ifndef NVTX_DISABLE
    if (nvshmemi_options.NVTX) {
        nvshmemi_options_print_heading("NVTX options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_NVTX, style)
#include "env_defs.h"

        if (style == NVSHMEMI_OPTIONS_STYLE_RST)
            printf(".. code-block:: none\n\n");

        nvshmem_nvtx_print_options();
#undef NVSHMEMI_ENV_DEF
    }
#endif /* !NVTX_DISABLE */

    printf("\n");
}
