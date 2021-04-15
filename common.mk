#
# Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

# Compiler / Make vars
CXX ?= /usr/bin/g++
CC ?= /usr/bin/gcc
ARCH := $(shell uname -m)

#CUDA Vars
CUDA_HOME ?= /usr/local/cuda-9.0
CUDA_DRV ?= $(CUDA_HOME)/lib64/stubs
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_VERSION = $(strip $(shell $(NVCC) --version | grep release | sed 's/^.*release \([0-9]\+\.[0-9]\+\).*/\1/'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXXFLAGS ?= 
NVCUFLAGS ?= 

# NVSHMEM internal features
NVSHMEM_HOME ?= /usr/local/nvshmem/
# Enable debug mode, builds library with -g
NVSHMEM_DEBUG ?= 0
NVSHMEM_VERBOSE ?= 0
NVSHMEM_DEVEL ?= 0
NVSHMEM_COMPLEX_SUPPORT ?= 0

# whether to build with UCX support. If yes, UCX_HOME should be set
NVSHMEM_UCX_SUPPORT ?= 0
# UCX install location
UCX_HOME ?= /usr/local/ucx
# Whether to build with MPI support. If yes, MPI_HOME should be set
NVSHMEM_MPI_SUPPORT ?= 1
# MPI install location
MPI_HOME ?= /usr/local/ompi/
# Name of MPI library
NVSHMEM_LMPI ?= -lmpi
# Whether to build with SHMEM support
NVSHMEM_SHMEM_SUPPORT ?= 0
# SHMEM install location
SHMEM_HOME ?= $(MPI_HOME)
# Name of SHMEM library
NVSHMEM_LSHMEM ?= -loshmem

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
NVCC_GENCODE += -gencode=arch=compute_80,code=sm_80
endif

MPI_LIBS := $(NVSHMEM_LMPI)
SHMEM_LIBS := $(NVSHMEM_LSHMEM)
TESTCUFLAGS  := -dc -ccbin $(CXX) -std=c++11 $(NVCC_GENCODE)
TESTLDFLAGS := -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(NVSHMEM_HOME)/lib -lnvshmem
TESTINC := -I$(CUDA_INC) -I$(mkfile_dir)/common

ifeq ($(NVSHMEM_COMPLEX_SUPPORT), 1)
CXXFLAGS  += -DNVSHMEM_COMPLEX_SUPPORT
NVCUFLAGS += -DNVSHMEM_COMPLEX_SUPPORT
TESTCUFLAGS += -DENABLE_COMPLEX_SUPPORT
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
TESTINC += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
TESTLDFLAGS += -L$(SHMEM_HOME)/lib $(SHMEM_LIBS)
endif
TESTINC += -I$(NVSHMEM_HOME)/include

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
TESTINC += -I$(MPI_HOME)/include -DNVSHMEM_MPI_SUPPORT
TESTLDFLAGS += -L$(MPI_HOME)/lib $(MPI_LIBS)
endif

ifeq ($(NVSHMEM_NVTX), 0)
TESTCUFLAGS  += -DNVTX_DISABLE
endif

ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
TESTINC += -I$(UCX_HOME)/include -DNVSHMEM_UCX_SUPPORT
TESTLDFLAGS += -L$(UCX_HOME)/lib -lucs -lucp
endif

ifeq ($(NVSHMEM_DEBUG), 0)
TESTCUFLAGS  += -O3
NVCUFLAGS += -O3
CXXFLAGS  += -O3
else
TESTCUFLAGS  += -O0 -g -G -D_NVSHMEM_DEBUG
NVCUFLAGS += -O0 -g -G -DENABLE_TRACE
CXXFLAGS  += -O0 -g -DENABLE_TRACE
endif

ifneq ($(NVSHMEM_VERBOSE), 0)
TESTCUFLAGS  += -Xptxas -v
NVCUFLAGS += -Xptxas -v
endif

#TODO: add -Werror and -Wall to TESTCUFLAGS. Have to fix all warnings first.
ifneq ($(NVSHMEM_DEVEL), 0)
NVCUFLAGS += -Werror all-warnings -Xcompiler -Wall,-Wextra,-Wno-unused-function,-Wno-unused-parameter
CXXFLAGS  += -Werror -Wall -Wextra -Wno-unused-function -Wno-unused-parameter
endif
