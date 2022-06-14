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

ifeq ($(NVSHMEM_LIBFABRIC_SUPPORT), 1)
CXXFLAGS  += -I$(LIBFABRIC_HOME)/include -DNVSHMEM_LIBFABRIC_SUPPORT
NVCUFLAGS += -I$(LIBFABRIC_HOME)/include -DNVSHMEM_LIBFABRIC_SUPPORT
endif

ifeq ($(NVSHMEM_IBRC_SUPPORT), 1)
CXXFLAGS  += -DNVSHMEM_IBRC_SUPPORT
NVCUFLAGS += -DNVSHMEM_IBRC_SUPPORT
endif

ifeq ($(NVSHMEM_IBDEVX_SUPPORT), 1)
CXXFLAGS  += -DNVSHMEM_IBDEVX_SUPPORT
NVCUFLAGS += -DNVSHMEM_IBDEVX_SUPPORT
endif

ifeq ($(NVSHMEM_GPUINITIATED_SUPPORT), 1)
CXXFLAGS  += -DNVSHMEM_GPUINITIATED_SUPPORT
NVCUFLAGS += -DNVSHMEM_GPUINITIATED_SUPPORT
ifeq ($(NVSHMEM_DEBUG), 1)
CXXFLAGS  += -DNVSHMEM_GPUINITIATED_DEBUG
NVCUFLAGS += -DNVSHMEM_GPUINITIATED_DEBUG
endif
endif

ifeq ($(NVSHMEM_USE_GDRCOPY), 1)
CXXFLAGS  += -I$(GDRCOPY_HOME)/include -DNVSHMEM_USE_GDRCOPY
NVCUFLAGS += -I$(GDRCOPY_HOME)/include -DNVSHMEM_USE_GDRCOPY
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
CXXFLAGS  += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
NVCUFLAGS += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
endif

# If we have an internal NCCL header, use it. Otherwise, use the one in NCCL_HOME
ifeq ($(NVSHMEM_USE_NCCL), 1)
ifneq ("$(wildcard $(mkfile_dir)/include_nccl)","")
CXXFLAGS  += -I$(mkfile_dir)/include_nccl -DNVSHMEM_USE_NCCL
NVCUFLAGS += -I$(mkfile_dir)/include_nccl -DNVSHMEM_USE_NCCL
else
CXXFLAGS  += -I$(NCCL_HOME)/include -DNVSHMEM_USE_NCCL
NVCUFLAGS += -I$(NCCL_HOME)/include -DNVSHMEM_USE_NCCL
endif
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

NVSHMEM_INFO_BUILD_VARS ?= 1

ifeq ($(NVSHMEM_INFO_BUILD_VARS), 1)
INFO_BUILD_VARS := $(strip $(foreach v, $(sort $(filter NVSHMEM_% %_HOME, $(.VARIABLES))), $(v)=\\\"$($(v))\\\"))
else
INFO_BUILD_VARS :=
endif

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
              nvshmemi_transfer.h \
              nvshmemx.h \
              nvshmemx_api.h \
              nvshmemx_coll_api.h \
              nvshmemi_constants.h \
              nvshmemx_defines.h \
              nvshmemx_error.h

PLUGINEXPORTS := src/bootstrap/bootstrap_pmix.c \
                 src/bootstrap/bootstrap_pmi.cpp \
                 src/bootstrap/bootstrap_mpi.c \
                 src/bootstrap/bootstrap_util.h \
                 src/bootstrap/bootstrap_shmem.c

HOSTLIBSRCFILES := bootstrap/bootstrap.cpp \
               bootstrap/bootstrap_loader.cpp

HOSTLIBSRCFILES += comm/transports/common/transport_common.cpp

ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
HOSTLIBSRCFILES += comm/transports/ucx/ucx.cpp
endif
ifeq ($(NVSHMEM_IBRC_SUPPORT), 1)
HOSTLIBSRCFILES += comm/transports/ibrc/ibrc.cpp
endif
ifeq ($(NVSHMEM_IBDEVX_SUPPORT), 1)
HOSTLIBSRCFILES += comm/transports/ibdevx/ibdevx.cpp
endif
ifeq ($(NVSHMEM_GPUINITIATED_SUPPORT), 1)
HOSTLIBSRCFILES += comm/transports/gic/gic.cpp
DEVICELIBSRCFILES += comm/device/gic_device.cu
INCEXPORTS += nvshmemi_gic.h
endif
ifeq ($(NVSHMEM_LIBFABRIC_SUPPORT), 1)
HOSTLIBSRCFILES += comm/transports/libfabric/libfabric.cpp
endif

HOSTLIBSRCFILES += coll/host/cpu_coll.cpp \
                   coll/host/alltoall.cpp \
                   coll/host/alltoall_on_stream.cpp \
                   coll/host/barrier.cpp \
                   coll/host/barrier_on_stream.cpp \
                   coll/host/broadcast.cpp \
                   coll/host/broadcast_on_stream.cpp \
                   coll/host/fcollect.cpp \
                   coll/host/fcollect_on_stream.cpp \
                   coll/host/rdxn.cpp \
                   coll/host/rdxn_on_stream.cpp \
                   comm/host/putget.cpp \
                   comm/host/fence.cpp \
                   comm/proxy/proxy.cpp \
                   comm/host/quiet.cpp \
                   comm/host/sync.cpp \
                   comm/host/amo.cpp \
                   comm/transport.cpp \
                   comm/transports/p2p/p2p.cpp \
                   init/init.cu \
                   init/init_nvtx.cpp \
                   init/query_host.cpp \
                   launch/collective_launch.cpp \
                   mem/mem.cpp \
                   team/team.cu \
                   team/team_internal.cu \
                   topo/topo.cpp \
                   util/cs.cpp \
                   util/debug.cpp \
                   util/env_vars.cpp \
                   util/util.cpp \
                   util/sockets.cpp

DEVICELIBSRCFILES += coll/device/alltoall.cu \
                     coll/device/barrier.cu \
                     coll/device/broadcast.cu \
                     coll/device/fcollect.cu \
                     coll/device/gpu_coll.cu \
                     coll/device/gpu_coll_dev.cu \
                     coll/device/recexchalgo.cu \
                     coll/device/rdxn_thread.cu \
                     coll/device/rdxn_warp.cu \
                     coll/device/rdxn_block.cu \
                     comm/device/proxy_device.cu \
                     comm/device/transfer_device.cu \
                     launch/collective_launch_device.cu \
                     init/init_device.cu \
                     init/query_device.cu \
                     team/team_device.cu \
                     team/team_internal_device.cu

ifeq ($(NVSHMEM_USE_DLMALLOC), 1)
HOSTLIBSRCFILES += mem/dlmalloc.cpp
else
HOSTLIBSRCFILES += mem/custom_malloc.cpp
endif

DEVICELIBSRCFILES_NOMAXRREGCOUNT = \
               coll/device/kernels/alltoall.cu \
               coll/device/kernels/barrier.cu \
               coll/device/kernels/broadcast.cu \
               coll/device/kernels/fcollect.cu \
               coll/device/kernels/rdxn.cu \
               comm/host/cuda_interface_sync.cu \
               comm/host/proxy/rma.cu \
               comm/host/quiet_on_stream.cu

HOSTLIBNAME   := libnvshmem_host.so
DEVICELIBNAME := libnvshmem_device.a
LIBNAME       := libnvshmem.a

INCDIR := $(NVSHMEM_BUILDDIR)/include
LIBDIR := $(NVSHMEM_BUILDDIR)/lib
BINDIR := $(NVSHMEM_BUILDDIR)/bin
PLUGINSDIR := $(NVSHMEM_BUILDDIR)/share/nvshmem/src/bootstrap-plugins
OBJDIR_NVSHMEM := $(NVSHMEM_BUILDDIR)/obj_nvshmem

BUILT_HEADERS := $(INCDIR)/nvshmem_version.h

INCTARGETS := $(patsubst %, $(INCDIR)/%, $(INCEXPORTS))

HOSTLIBTARGET   := $(HOSTLIBNAME)
DEVICELIBTARGET := $(DEVICELIBNAME)
LIBTARGET       := $(LIBNAME)
HOSTLIBOBJ      := $(patsubst %.cpp, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cpp, $(HOSTLIBSRCFILES)))
HOSTLIBOBJ      += $(patsubst %.c, $(OBJDIR_NVSHMEM)/%.o, $(filter %.c, $(HOSTLIBSRCFILES)))
HOSTLIBOBJ      += $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(HOSTLIBSRCFILES)))
DEVICELIBOBJ    := $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(DEVICELIBSRCFILES)))
DEVICELIBOBJ_NOMAXRREGCOUNT = $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(DEVICELIBSRCFILES_NOMAXRREGCOUNT)))
DEVICELIBOBJ_NOMAXRREGCOUNT += $(patsubst %.cpp, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cpp, $(DEVICELIBSRCFILES_NOMAXRREGCOUNT)))

LIBINC     := -Isrc/include -Isrc/util -Isrc/bootstrap -Isrc/comm/transports/common -Isrc/coll/host -Isrc/coll/device -Isrc/coll -Isrc/topo
LIBINC     += -Isrc/pmi/pmi-2 -Isrc/pmi/simple-pmi -I$(INCDIR)

PLUGINEXPORTTARGETS := $(addprefix $(PLUGINSDIR)/, $(notdir $(PLUGINEXPORTS)))

PLUGINS    := $(LIBDIR)/nvshmem_bootstrap_pmi.so $(LIBDIR)/nvshmem_bootstrap_pmi2.so
ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
PLUGINS    += $(LIBDIR)/nvshmem_bootstrap_pmix.so
endif
ifeq ($(NVSHMEM_MPI_SUPPORT), 1) 
PLUGINS    += $(LIBDIR)/nvshmem_bootstrap_mpi.so
endif
ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1) 
PLUGINS    += $(LIBDIR)/nvshmem_bootstrap_shmem.so
endif

.PHONY: lib
lib : $(INCTARGETS) $(LIBDIR)/$(DEVICELIBTARGET) $(LIBDIR)/$(HOSTLIBTARGET) $(LIBDIR)/$(LIBTARGET) $(PLUGINS) $(PLUGINEXPORTTARGETS) $(BINDIR)/nvshmem-info

EXTRA_NVCUFLAGS = $(NVCU_MAXRREGCOUNT)
$(LIBOBJ_NOMAXRREGCOUNT) : EXTRA_NVCUFLAGS =

$(LIBDIR)/$(LIBTARGET) : $(HOSTLIBOBJ)  $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT)
	@mkdir -p $(LIBDIR)
	$(NVCC) -lib -o $@ $(HOSTLIBOBJ) $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT) 

$(LIBDIR)/$(HOSTLIBTARGET) : $(HOSTLIBOBJ) $(LIBDIR)/$(DEVICELIBTARGET) nvshmem_host.sym
	@mkdir -p $(LIBDIR)
	$(NVCC) $(NVCC_GENCODE) -shared -Xlinker --version-script=nvshmem_host.sym -o $@ $(HOSTLIBOBJ) $(LIBDIR)/$(DEVICELIBTARGET) $(LDFLAGS)

$(LIBDIR)/$(DEVICELIBTARGET) : $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT)
	@mkdir -p $(LIBDIR)
	$(NVCC) -lib -o $@ $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT)

$(PLUGINSDIR)/%: src/bootstrap/%
	@mkdir -p $(PLUGINSDIR)
	cp -f $< $@

$(LIBDIR)/nvshmem_bootstrap_pmi.so: src/bootstrap/bootstrap_pmi.cpp $(BUILT_HEADERS) $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmi.o $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmiutil.o
	@mkdir -p $(LIBDIR)
	$(CC) $(CFLAGS) -shared -fpic -I$(INCDIR) -Isrc/pmi/simple-pmi $< -o $@ $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmi.o $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmiutil.o

$(LIBDIR)/nvshmem_bootstrap_pmi2.so: src/bootstrap/bootstrap_pmi.cpp $(BUILT_HEADERS) $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_api.o $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_util.o
	@mkdir -p $(LIBDIR)
	$(CC) $(CFLAGS) -shared -fpic -Xlinker --version-script=nvshmem_bootstrap.sym -DNVSHMEM_BUILD_PMI2 -I$(INCDIR) -Isrc/pmi/pmi-2 $< -o $@ $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_api.o $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_util.o

$(LIBDIR)/nvshmem_bootstrap_pmix.so: src/bootstrap/bootstrap_pmix.c $(BUILT_HEADERS)
	@mkdir -p $(LIBDIR)
	$(CC) $(CFLAGS) -shared -fpic -I$(INCDIR) -I$(PMIX_HOME)/include $< -L$(PMIX_HOME)/lib -lpmix -o $@

$(LIBDIR)/nvshmem_bootstrap_mpi.so: src/bootstrap/bootstrap_mpi.c
	@mkdir -p $(LIBDIR)
	$(MPICC) $(CFLAGS) -shared -fpic -Isrc/include -Isrc/bootstrap $< -o $@

$(LIBDIR)/nvshmem_bootstrap_shmem.so: src/bootstrap/bootstrap_shmem.c
	@mkdir -p $(LIBDIR)
	$(OSHCC) $(CFLAGS) -shared -fpic -Isrc/include -Isrc/bootstrap $< -o $@

$(BINDIR)/nvshmem-info: src/util/nvshmem-info.cpp $(LIBDIR)/$(LIBTARGET)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(CXX) -std=c++11 -Isrc/include -I$(INCDIR) $< -o $@ $(LDFLAGS) -L$(LIBDIR) -lnvshmem

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
	@echo "#define NVSHMEM_BUILD_VARS \"$(INFO_BUILD_VARS)\"" >> $@
	@echo "#endif /* NVSHMEM_VERSION_H */" >> $@

$(OBJDIR_NVSHMEM)/%.o : src/%.cpp $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(CXX) -c $(LIBINC) $(CXXFLAGS) $< -o $@

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
	mkdir -p $(NVSHMEM_PREFIX)/share
	mkdir -p $(NVSHMEM_PREFIX)/bin
	cp -v $(NVSHMEM_BUILDDIR)/bin/* $(NVSHMEM_PREFIX)/bin/
	cp -v $(NVSHMEM_BUILDDIR)/lib/* $(NVSHMEM_PREFIX)/lib/
	cp -P -v $(NVSHMEM_BUILDDIR)/include/* $(NVSHMEM_PREFIX)/include/
	cp -P -R -v $(NVSHMEM_BUILDDIR)/share/* $(NVSHMEM_PREFIX)/share/
