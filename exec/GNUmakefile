VARS_OLD := $(.VARIABLES)
# Keep installation-specific settings outside the tracked build rules.
# Run from exec/ (or use make -C exec). Relative paths are based there.
CONFIG ?= config.mk
-include $(CONFIG)

# GNU Make supplies a built-in CXX=g++; use MPI unless the user selected CXX.
ifneq ($(filter default undefined,$(origin CXX)),)
CXX = mpicxx
endif
DEPXX ?= $(CXX)
DIM=3
USE_HDF5=TRUE
PROTO ?= ../../proto
CXXSTD ?= c++17

# Radial coordinate index; shell radii and stretching are required in the inputs file.
CUBED_SPHERE_SHELL_RADIAL_COORD ?= 0

# --- Git info ---
GIT_HASH    := $(shell git rev-parse --short HEAD)
GIT_MESSAGE := $(shell git log -1 --pretty=%s | tr -d '"\\')

empty  :=
space  := $(empty) $(empty)
comma  := ,
# First replace commas with underscores
GIT_MESSAGE := $(subst $(comma),_,$(GIT_MESSAGE))
# Then replace spaces with underscores
GIT_MESSAGE := $(subst $(space),_,$(GIT_MESSAGE))

#CXXOPTFLAGS=-g
# CXXOPTFLAGS=-O2 -g
CXXOPTFLAGS ?= -O3
# Disable assertions in optimized builds to reduce iteration overhead.
# Requires the HDF5 assertion fixes in both Proto and MHDReader.
# For assertion-enabled debugging, remove -DNDEBUG and do a clean rebuild.
CXXOPTFLAGS+= -DNDEBUG

# Uncomment to omit HOST-to-HOST Copier packing barriers (default: enabled).
# GPU synchronization and MPI completion waits are retained.
# Rebuild all objects after changing this flag (make -B or a clean rebuild).
CXXOPTFLAGS += -DPR_DISABLE_HOST_COPIER_BARRIERS

# Use temporary HOST cubed-sphere conversion storage to reduce peak memory.
# Comment out to retain conversion buffers/plans for faster iterations.
# Rebuild all objects after changing this flag (make -B or a clean rebuild).
CXXOPTFLAGS += -DPR_DISABLE_CUBED_SPHERE_CONVERSION_CACHE

CXXOPTFLAGS+=-DmemoryFootPrint
CXXOPTFLAGS+=-DPR_AMR
CXXOPTFLAGS+= -DPR_DISABLE_INIT_CHECKS
CXXOPTFLAGS+=-DPR_MMB
CXXOPTFLAGS+=-DPR_LAPACK
CXXOPTFLAGS+=-DPR_CFD
CXXOPTFLAGS+=-DPR_LAPACK
CXXOPTFLAGS+= -DPR_HDF5=TRUE
CXXOPTFLAGS+= -DCUBED_SPHERE_SHELL_RADIAL_COORD=$(CUBED_SPHERE_SHELL_RADIAL_COORD)
CXXOPTFLAGS += -DGIT_HASH_RAW=$(GIT_HASH)
CXXOPTFLAGS += -DGIT_MESSAGE_RAW=$(GIT_MESSAGE)
CXXOPTFLAGS+= -DPR_MPI=TRUE
CXXOPTFLAGS+= -DPR_STACK_ALLOC=4294967296
TARGET:=cubedSphereTest.exe

SRC=$(patsubst ../src/%,%,$(wildcard ../src/*.cpp)) $(wildcard *.cpp) $(wildcard $(PROTO)/examples/_common/*.cpp) 
NODIRSRC = $(notdir $(SRC))
NODIROBJ = $(subst .cpp,.o,$(NODIRSRC))
OBJ=$(patsubst %.o,o/%.o, $(NODIROBJ))

NODIRDEP = $(subst .cpp,.d,$(NODIRSRC))
DEPENDS=$(patsubst %.d,d/%.d, $(NODIRDEP))

VPATH=o. ../src  $(PROTO)/examples/_common


CPPFLAGS:=-DDIM=$(DIM) -I$(PROTO)/include -I$(PROTO)/include/base -I$(PROTO)/include/amr -I$(PROTO)/include/mmb -I$(PROTO)/include/lapack -I$(PROTO)/include/utils -I$(PROTO)/include/tests -I$(PROTO)/include/lapack -I. -I../src -I$(PROTO)/examples/_common
CXXFLAGS:= $(CXXOPTFLAGS) -std=$(CXXSTD)

ifdef HDF5_LIB
LIBS+= -L$(HDF5_LIB) -lhdf5 -lz
CPPFLAGS+= -I$(HDF5_INC)
endif

ifdef LAPACK_LIB
LIBS+= -L$(LAPACK_LIB) -lopenblas
CPPFLAGS+= -I$(LAPACK_INC)
endif

ifdef OLCF_HDF5_ROOT
LIBS+= -L$(OLCF_HDF5_ROOT)/lib -lhdf5 -lz
CPPFLAGS+= -I$(OLCF_HDF5_ROOT)/include
endif

ifdef FFTW3_DIR
LIBS+= -L$(FFTW3_DIR)/lib -lfftw3
CPPFLAGS+= -I$(FFTW3_DIR)/include
endif

# -c tells MPI wrappers to omit linker flags during dependency generation.
DEP=$(DEPXX) -MM -c $(CXXFLAGS)
ifeq ($(DEPXX),nvcc)
DEP=$(DEPXX) -M $(CXXFLAGS)
LIBS+= -lnvToolsExt
endif

ifeq ($(DEPXX),hipcc)
CPPFLAGS+=-I$(ROCM_PATH)/rocthrust/include/ -I$(ROCM_PATH)/rocprim/include/ -I$(ROCM_PATH)/include/roctracer -I${MPICH_DIR}/include
CXXFLAGS+= -Wno-non-c-typedef-for-linkage -Wno-unused-command-line-argument
LIBS+= -L/$(ROCM_PATH)/lib/ -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa
LIBS+= -lroctx64
endif


LINK=$(CXX)
OBJDIR = o
DEPDIR = d


$(TARGET): $(OBJ)
	$(LINK) -o $(TARGET) $(OBJ) $(LIBS)

# Check before compiling/linking, while keeping clean and print-* usable
# without installed dependencies. OLCF_HDF5_ROOT remains an alternative.
$(OBJ) $(TARGET): | check-config

.PHONY: check-config
check-config:
	@test -d "$(PROTO)/include" -a -d "$(PROTO)/examples/_common" || \
		{ echo "Set PROTO to a Proto checkout in $(CONFIG) or on the make command line."; exit 1; }
	@test -n "$(strip $(HDF5_INC))" -a -n "$(strip $(HDF5_LIB))" || \
		test -n "$(strip $(OLCF_HDF5_ROOT))" || \
		{ echo "Set HDF5_INC and HDF5_LIB (or OLCF_HDF5_ROOT) for parallel HDF5 in $(CONFIG)."; exit 1; }
	@test -n "$(strip $(LAPACK_INC))" -a -n "$(strip $(LAPACK_LIB))" || \
		{ echo "Set LAPACK_INC and LAPACK_LIB to OpenBLAS directories in $(CONFIG)."; exit 1; }

$(OBJDIR):
	@if [ ! -d "./$(OBJDIR)" ];then     \
		mkdir $(OBJDIR);           \
	fi

$(DEPDIR):
	@if [ ! -d "./$(DEPDIR)" ];then     \
		mkdir $(DEPDIR);           \
	fi

#compile non-main sources
$(OBJDIR)/%.o : %.cpp   $(OBJDIR) $(DEPDIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(CPPFLAGS)
	$(DEP) $< $(CPPFLAGS) -MT $@ > $(DEPDIR)/$*.d



.PHONY: clean dataclean realclean print dirs


print-%: ; @echo $* = $($*)

-include $(DEPENDS)

vars:
	$(foreach v,$(filter-out $(VARS_OLD) VARS_OLD,$(.VARIABLES)), $(info $(v) = $($(v))))

clean:
	rm -rf d o *.o *.exe *.d $(PROTO)/examples/_common/*.o ../src/*.o ../../common/*.d ../src/*.d

dataclean:
	rm -rf *.hdf5 *.vtk *.table pout* *.log *.dat *.html *.csv *.amr
