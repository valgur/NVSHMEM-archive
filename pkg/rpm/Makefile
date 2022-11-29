#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information
#

include ../../common.mk
include ../../version.mk
BUILDDIR ?= $(abspath ../../build)
RPMPREPDIR := $(BUILDDIR)/redhat
PKGDIR  := $(BUILDDIR)/pkg/rpm/

RPMGEN_IN  := $(wildcard *.in)
RPMGEN     := $(RPMGEN_IN:.in=)
RPMFILES   := $(RPMGEN)
RPMTARGETS := $(patsubst %, $(RPMPREPDIR)/%, $(RPMFILES))

PKG_TIMESTAMP  := $(shell date -R)
ARCH           := $(shell uname -m)
PKG_ARCH       ?= $(shell uname -m)
PKG_MULTIARCH  ?= $(shell $(CXX) -print-multiarch)
ifeq ($(PKG_MULTIARCH),)
# Hardwire the PKG_MULTIARCH directory as the RHEL6 distribution agnostic compiler (gcc 4.8.3) doesn't set it
PKG_MULTIARCH  := $(ARCH)-linux-gnu
endif

ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
define NVSHMEM_PMIX_PKG_INSTALL
install -m 644 lib/nvshmem_bootstrap_pmix.so.${NVSHMEM_MAJOR}.${NVSHMEM_MINOR}.${NVSHMEM_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_bootstrap_pmix.so.${NVSHMEM_MAJOR}.${NVSHMEM_MINOR}.${NVSHMEM_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_pmix.so.${NVSHMEM_MAJOR}\nln -s nvshmem_bootstrap_pmix.so.${NVSHMEM_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_pmix.so
endef
else
NVSHMEM_PMIX_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
define NVSHMEM_MPI_PKG_INSTALL
install -m 644 lib/nvshmem_bootstrap_mpi.so.${NVSHMEM_MAJOR}.${NVSHMEM_MINOR}.${NVSHMEM_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_bootstrap_mpi.so.${NVSHMEM_MAJOR}.${NVSHMEM_MINOR}.${NVSHMEM_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_mpi.so.${NVSHMEM_MAJOR}\nln -s nvshmem_bootstrap_mpi.so.${NVSHMEM_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_mpi.so
endef
else
NVSHMEM_MPI_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
define NVSHMEM_SHMEM_PKG_INSTALL
install -m 644 lib/nvshmem_bootstrap_shmem.so.${NVSHMEM_MAJOR}.${NVSHMEM_MINOR}.${NVSHMEM_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_bootstrap_shmem.so.${NVSHMEM_MAJOR}.${NVSHMEM_MINOR}.${NVSHMEM_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_shmem.so.${NVSHMEM_MAJOR}\nln -s nvshmem_bootstrap_shmem.so.${NVSHMEM_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_shmem.so
endef
else
NVSHMEM_SHMEM_PKG_INSTALL=""
endif

build: $(RPMTARGETS)
	$(MAKE) -C ../txz build BUILDDIR=$(BUILDDIR)
	@printf "Building Redhat package\n"
	mkdir -p $(PKGDIR)
	rpmbuild --define "_sourcedir $(BUILDDIR)/pkg/txz" \
                 --define "_rpmdir $(PKGDIR)" \
                 --define "_builddir $(PKGDIR)/build/" \
                 --define "_buildrootdir $(PKGDIR)/buildroot/" \
                 -bb $(BUILDDIR)/redhat/nvshmem.spec

clean:
	rm -Rf $(RPMPREPDIR) $(PKGDIR)

$(RPMPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(RPMPREPDIR)
	sed -e "s/\$${nvshmem:Major}/$(NVSHMEM_MAJOR)/g" \
	    -e "s/\$${nvshmem:Minor}/$(NVSHMEM_MINOR)/g" \
	    -e "s/\$${nvshmem:Patch}/$(NVSHMEM_PATCH)/g" \
	    -e "s/\$${nvshmem:Suffix}/$(NVSHMEM_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Timestamp}/$(PKG_TIMESTAMP)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    -e "s/\$${pkg:MultiArch}/$(PKG_MULTIARCH)/g" \
	    -e "s,\$${nvshmem:pmix_install},$(NVSHMEM_PMIX_PKG_INSTALL),g" \
	    -e "s,\$${nvshmem:mpi_install},$(NVSHMEM_MPI_PKG_INSTALL),g" \
	    -e "s,\$${nvshmem:shmem_install},$(NVSHMEM_SHMEM_PKG_INSTALL),g" \
	    $< > $@

$(RPMPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(RPMPREPDIR)
	cp -f $< $@
