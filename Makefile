#
# Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information
#

# Define this variable for the Include Variable in common.mk
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

# External dependencies
include common.mk
include version.mk

# Build/install location
# location where the build will be installed
NVSHMEM_PREFIX ?= /usr/local/nvshmem/
# build location
NVSHMEM_BUILDDIR ?= $(abspath build)
# MPI/SHMEM Support

# GDRCopy install/headers location
GDRCOPY_HOME ?= /usr/local/gdrdrv
# NCCL install/headers location
NCCL_HOME ?= /usr/local/nccl
NVSHMEM_USE_NCCL ?= 0
# Whether to build with MPI support. If yes, MPI_HOME should be set
NVSHMEM_USE_GDRCOPY ?= 1
# Include support for PMIx as the process manager interface
NVSHMEM_PMIX_SUPPORT ?= 0
PMIX_HOME ?= /usr
# One of the below can be set to 1 to override the default PMI
NVSHMEM_DEFAULT_PMIX ?= 0
NVSHMEM_DEFAULT_PMI2 ?= 0

# This can be set to override the default remote transport.
NVSHMEM_DEFAULT_UCX ?= 0

# NVSHMEM internal features
NVSHMEM_TRACE ?= 0
NVSHMEM_NVTX ?= 1

NVSHMEM_DISABLE_COLL_POLL ?= 1
NVSHMEM_GPU_COLL_USE_LDST ?= 0
# Timeout if stuck for long in wait loops in device
NVSHMEM_TIMEOUT_DEVICE_POLLING ?= 0
# Use dlmalloc (instead of custom_malloc) as heap
# allocator (will work only if not using CUDA VMM)
NVSHMEM_USE_DLMALLOC ?= 0

ifeq ($(ARCH), x86_64)
CXXFLAGS += -fPIC -I$(CUDA_INC) -msse
CXXFLAGS += -DNVSHMEM_X86_64
NVCUFLAGS += -DNVSHMEM_X86_64
else
ifeq ($(ARCH), ppc64le)
CXXFLAGS   += -fPIC -I$(CUDA_INC) -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
CXXFLAGS += -DNVSHMEM_PPC64LE
NVCUFLAGS += -DNVSHMEM_PPC64LE
endif
endif
NVCUFLAGS  += -Xcompiler -fPIC -ccbin $(CXX) $(NVCC_GENCODE) -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS

ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
CXXFLAGS  += -DNVSHMEM_PMIX_SUPPORT
else
# Don't allow PMIX to be set as default unless it's in the build
NVSHMEM_DEFAULT_PMIX := 0
endif

ifeq ($(NVSHMEM_DEFAULT_PMIX), 1)
CXXFLAGS  += -DNVSHMEM_DEFAULT_PMIX
else
ifeq ($(NVSHMEM_DEFAULT_PMI2), 1)
CXXFLAGS  += -DNVSHMEM_DEFAULT_PMI2
endif
endif

ifeq ($(NVSHMEM_DEFAULT_UCX), 1)
CXXFLAGS  += -DNVSHMEM_DEFAULT_UCX
endif

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
CXXFLAGS  += -I$(MPI_HOME)/include -DNVSHMEM_MPI_SUPPORT
NVCUFLAGS += -I$(MPI_HOME)/include -DNVSHMEM_MPI_SUPPORT
endif

ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
CXXFLAGS  += -I$(UCX_HOME)/include -DNVSHMEM_UCX_SUPPORT
NVCUFLAGS += -I$(UCX_HOME)/include -DNVSHMEM_UCX_SUPPORT
endif

ifeq ($(NVSHMEM_USE_GDRCOPY), 1)
CXXFLAGS  += -I$(GDRCOPY_HOME)/include -DNVSHMEM_USE_GDRCOPY
NVCUFLAGS += -I$(GDRCOPY_HOME)/include -DNVSHMEM_USE_GDRCOPY
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
CXXFLAGS  += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
NVCUFLAGS += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
endif

ifeq ($(NVSHMEM_USE_NCCL), 1)
CXXFLAGS  += -I$(NCCL_HOME)/include -DNVSHMEM_USE_NCCL
NVCUFLAGS += -I$(NCCL_HOME)/include -DNVSHMEM_USE_NCCL
endif

ifeq ($(NVSHMEM_DISABLE_COLL_POLL), 1)
CXXFLAGS  += -DNVSHMEM_DISABLE_COLL_POLL
NVCUFLAGS += -DNVSHMEM_DISABLE_COLL_POLL
endif
ifeq ($(NVSHMEM_GPU_COLL_USE_LDST), 1)
CXXFLAGS  += -DNVSHMEM_GPU_COLL_USE_LDST
NVCUFLAGS += -DNVSHMEM_GPU_COLL_USE_LDST
endif

ifeq ($(NVSHMEM_TIMEOUT_DEVICE_POLLING), 1)
CXXFLAGS  += -DNVSHMEM_TIMEOUT_DEVICE_POLLING
NVCUFLAGS += -DNVSHMEM_TIMEOUT_DEVICE_POLLING
endif

ifeq ($(NVSHMEM_TRACE), 1)
NVCUFLAGS += -DENABLE_TRACE
CXXFLAGS  += -DENABLE_TRACE
endif

# ignore the following for clean targets
ifeq (,$(findstring $(MAKECMDGOALS),purge clean))
ifeq ($(NVSHMEM_NVTX), 0)
NVCUFLAGS += -DNVTX_DISABLE
CXXFLAGS  += -DNVTX_DISABLE
else
# C++11 is required for NVTX support
cppver := $(shell sh ./scripts/test_cxx11.sh $(CXX) "$(CXXFLAGS)")
ifneq ($(cppver),)
NVCUFLAGS += -DNVTX_DISABLE
CXXFLAGS  += -DNVTX_DISABLE
$(info ${cppver})
endif
endif
endif

NVCU_MAXRREGCOUNT := --maxrregcount 32

.PHONY : default 
default : lib

LICENSE_FILES := NVSHMEM-SLA.txt COPYRIGHT.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(NVSHMEM_BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

${NVSHMEM_BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${NVSHMEM_BUILDDIR}
	cp $< $@

INCEXPORTS_NVSHMEM  := nvshmem.h nvshmemx.h
INCEXPORTS := nvshmem.h \
              nvshmem_api.h \
              nvshmem_bootstrap.h \
              nvshmem_coll_api.h \
              nvshmem_common.cuh \
              nvshmem_constants.h \
              nvshmem_defines.h \
              nvshmem_types.h \
              nvshmemi_util.h \
              nvshmemx.h \
              nvshmemx_api.h \
              nvshmemx_coll_api.h \
              nvshmemi_constants.h \
              nvshmemx_defines.h \
              nvshmemx_error.h

PLUGINEXPORTS := src/bootstrap/bootstrap_pmix.c \
                 src/bootstrap/bootstrap_mpi.c \
                 src/bootstrap/bootstrap_util.h

LIBSRCFILES := bootstrap/bootstrap.cpp \
               bootstrap/bootstrap_loader.cpp

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1) 
LIBSRCFILES += bootstrap/bootstrap_shmem.cpp  
endif
ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
LIBSRCFILES += comm/transports/ucx/ucx.cpp
endif

LIBSRCFILES += bootstrap/bootstrap_pmi.cpp \
               coll/device/alltoall.cu \
               coll/device/kernels/alltoall.cu \
               coll/device/barrier.cu \
               coll/device/kernels/barrier.cu \
               coll/device/bcast.cu \
               coll/device/kernels/bcast.cu \
               coll/device/fcollect.cu \
               coll/device/kernels/fcollect.cu \
               coll/device/gpu_coll.cu \
               coll/device/gpu_coll_dev.cu \
               coll/device/recexchalgo.cu \
               coll/device/rdxn.cu \
               coll/device/rdxn_threadgroup.cu \
               coll/device/kernels/rdxn_threadgroup.cu \
               coll/host/alltoall.cpp \
               coll/host/alltoall_on_stream.cpp \
               coll/host/barrier.cpp \
               coll/host/barrier_on_stream.cpp \
               coll/host/broadcast.cpp \
               coll/host/broadcast_on_stream.cpp \
               coll/host/fcollect.cpp \
               coll/host/fcollect_on_stream.cpp \
               coll/host/cpu_coll.cpp \
               coll/host/rdxn.cpp \
               coll/host/rdxn_on_stream.cpp \
               comm/device/amo.cu \
               comm/device/get.cu \
               comm/device/get_threadgroup.cu \
               comm/device/proxy_device.cu \
               comm/device/put.cu \
               comm/device/put_threadgroup.cu \
               comm/host/putget.cpp \
               comm/host/amo.cu \
               comm/host/cuda_interface_sync.cu \
               comm/host/fence.cpp \
               comm/proxy/proxy.cu \
               comm/host/proxy/rma.cu \
               comm/host/quiet.cpp \
               comm/host/quiet_on_stream.cu \
               comm/host/sync.cpp \
               comm/transport.cpp \
               comm/transports/ibrc/ibrc.cpp \
               comm/transports/p2p/p2p.cpp \
               init/init.cpp \
               init/init_device.cu \
               init/init_nvtx.cpp \
               init/query.cu \
               launch/collective_launch.cpp \
               mem/mem.cpp \
               pmi/pmi-2/pmi2_api.c \
               pmi/pmi-2/pmi2_util.c \
               pmi/simple-pmi/simple_pmi.cpp \
               pmi/simple-pmi/simple_pmiutil.cpp \
               team/team.cu \
               team/team_internal.cu \
               topo/topo.cpp \
               util/cs.cpp \
               util/debug.cpp \
               util/env_vars.cpp \
               util/util.cpp \
               util/sockets.cpp

ifeq ($(NVSHMEM_USE_DLMALLOC), 1)
LIBSRCFILES += mem/dlmalloc.cpp
else
LIBSRCFILES += mem/custom_malloc.cpp
endif

LIBSRCFILES_NOMAXRREGCOUNT = \
               coll/device/kernels/alltoall.cu \
               coll/device/kernels/barrier.cu \
               coll/device/kernels/bcast.cu \
               coll/device/kernels/fcollect.cu \
               coll/device/kernels/rdxn_threadgroup.cu \
               comm/host/amo.cu \
               comm/host/cuda_interface_sync.cu \
               comm/host/proxy/rma.cu \
               comm/host/quiet_on_stream.cu

LIBNAME     := libnvshmem.a

INCDIR := $(NVSHMEM_BUILDDIR)/include
LIBDIR := $(NVSHMEM_BUILDDIR)/lib
PLUGINSDIR := $(NVSHMEM_BUILDDIR)/share/nvshmem/src/bootstrap-plugins
OBJDIR_NVSHMEM := $(NVSHMEM_BUILDDIR)/obj_nvshmem

BUILT_HEADERS := $(INCDIR)/nvshmem_version.h

INCTARGETS := $(patsubst %, $(INCDIR)/%, $(INCEXPORTS))

LIBTARGET  := $(LIBNAME)
LIBOBJ     := $(patsubst %.cpp, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cpp, $(LIBSRCFILES)))
LIBOBJ     += $(patsubst %.c, $(OBJDIR_NVSHMEM)/%.o, $(filter %.c, $(LIBSRCFILES)))
LIBOBJ     += $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(LIBSRCFILES)))
LIBOBJ     += $(OBJDIR_NVSHMEM)/bootstrap/bootstrap_pmi2.o
LIBOBJ_NOMAXRREGCOUNT = $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(LIBSRCFILES_NOMAXRREGCOUNT)))

LIBINC     := -Isrc/include -Isrc/util -Isrc/bootstrap -Isrc/comm -Isrc/coll/host -Isrc/coll/device -Isrc/coll -Isrc/topo
LIBINC     += -Isrc/pmi/pmi-2 -Isrc/pmi/simple-pmi -I$(INCDIR)

PLUGINEXPORTTARGETS := $(addprefix $(PLUGINSDIR)/, $(notdir $(PLUGINEXPORTS)))

ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
PLUGINS    := $(LIBDIR)/nvshmem_bootstrap_pmix.so
endif
ifeq ($(NVSHMEM_MPI_SUPPORT), 1) 
PLUGINS    += $(LIBDIR)/nvshmem_bootstrap_mpi.so
endif

.PHONY: lib
lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PLUGINS) $(PLUGINEXPORTTARGETS)

EXTRA_NVCUFLAGS = $(NVCU_MAXRREGCOUNT)
$(LIBOBJ_NOMAXRREGCOUNT) : EXTRA_NVCUFLAGS =

$(LIBDIR)/$(LIBTARGET) : $(LIBOBJ)
	@mkdir -p $(LIBDIR)
	$(NVCC) -lib -o $@ $(LIBOBJ)

$(PLUGINSDIR)/%: src/bootstrap/%
	@mkdir -p $(PLUGINSDIR)
	cp -f $< $@

$(LIBDIR)/nvshmem_bootstrap_pmix.so: src/bootstrap/bootstrap_pmix.c $(BUILT_HEADERS)
	@mkdir -p $(LIBDIR)
	$(CC) -shared -fpic -Isrc/include -I$(PMIX_HOME)/include $< -L$(PMIX_HOME)/lib -lpmix -o $@

$(LIBDIR)/nvshmem_bootstrap_mpi.so: src/bootstrap/bootstrap_mpi.c
	@mkdir -p $(LIBDIR)
	$(MPICC) -shared -fpic -Isrc/include -Isrc/bootstrap $< -o $@

$(INCDIR)/%.h : src/include/%.h
	@mkdir -p $(INCDIR)
	cp -f $< $@

$(INCDIR)/%.cuh : src/include/%.cuh
	@mkdir -p $(INCDIR)
	cp -f $< $@

$(INCDIR)/nvshmem_version.h :
	@mkdir -p $(INCDIR)
	@echo "#ifndef NVSHMEM_VERSION_H" > $@
	@echo "#define NVSHMEM_VERSION_H" >> $@
	@echo "#define NVSHMEM_VENDOR_MAJOR_VERSION $(NVSHMEM_MAJOR)" >> $@
	@echo "#define NVSHMEM_VENDOR_MINOR_VERSION $(NVSHMEM_MINOR)" >> $@
	@echo "#define NVSHMEM_VENDOR_PATCH_VERSION $(NVSHMEM_PATCH)" >> $@
	@echo "#endif /* NVSHMEM_VERSION_H */" >> $@

$(OBJDIR_NVSHMEM)/%.o : src/%.cpp $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(CXX) -c $(LIBINC) $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/bootstrap/bootstrap_pmi2.o : src/bootstrap/bootstrap_pmi.cpp $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(CXX) -c $(LIBINC) -DNVSHMEM_BUILD_PMI2 $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/%.o : src/%.c $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(CC) -c $(LIBINC) $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/%.o : src/%.cu $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(NVCC) -c $(LIBINC) $(NVCUFLAGS) $(EXTRA_NVCUFLAGS) -rdc=true $< -o $@

.PHONY: clean
clean :
	rm -rf $(NVSHMEM_BUILDDIR)

.PHONY: uninstall
uninstall:
	rm -rf $(NVSHMEM_PREFIX)

.PHONY: purge
purge: clean uninstall

.PHONY: install
install : lib
	mkdir -p $(NVSHMEM_PREFIX)/lib
	mkdir -p $(NVSHMEM_PREFIX)/include
	cp -v $(NVSHMEM_BUILDDIR)/lib/* $(NVSHMEM_PREFIX)/lib/
	cp -P -v $(NVSHMEM_BUILDDIR)/include/nvshmem* $(NVSHMEM_PREFIX)/include
	cp -P -v $(NVSHMEM_BUILDDIR)/include/* $(NVSHMEM_PREFIX)/include/
