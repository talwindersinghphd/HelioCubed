VARS_OLD := $(.VARIABLES)
#DEPXX=clang++
#CXX=clang++
DEPXX=mpicxx
CXX=mpicxx
DIM=3
TURB=0
USE_HDF5=TRUE
HDF5_LIB=/opt/homebrew/opt/hdf5-mpi/lib
HDF5_INC=/opt/homebrew/opt/hdf5-mpi/include
# HDF5_LIB=/apps/spack/anvil/apps/hdf5/1.10.7-gcc-11.2.0-axhcpbs/lib
# HDF5_INC=/apps/spack/anvil/apps/hdf5/1.10.7-gcc-11.2.0-axhcpbs/include

PROTO ?= ../../proto

# Use DEBUG=1 for an assertion-enabled build. Clean or use separate object
# directories when changing compiler flags.
DEBUG ?= 0
CXXOPTFLAGS ?= -O3
ifeq ($(DEBUG),1)
CXXOPTFLAGS += -g
else
CXXOPTFLAGS += -DNDEBUG -DPR_DISABLE_HOST_COPIER_BARRIERS
endif
CXXOPTFLAGS+=-DmemoryFootPrint
CXXOPTFLAGS+= -DPR_DISABLE_INIT_CHECKS
CXXOPTFLAGS+= -DPR_HDF5=TRUE
CXXOPTFLAGS+= -DPR_MPI=TRUE
CXXOPTFLAGS+= -DPR_STACK_ALLOC=4294967296
CXXOPTFLAGS+= -DPR_OPS=TRUE
TARGET ?= FullSphere.exe
OBJDIR ?= o
DEPDIR ?= d

SRC=$(patsubst ../src/%,%,$(wildcard ../src/*.cpp)) $(wildcard *.cpp)
NODIRSRC = $(notdir $(SRC))
NODIROBJ = $(subst .cpp,.o,$(NODIRSRC))
OBJ=$(patsubst %.o,$(OBJDIR)/%.o, $(NODIROBJ))

NODIRDEP = $(subst .cpp,.d,$(NODIRSRC))
DEPENDS=$(patsubst %.d,$(DEPDIR)/%.d, $(NODIRDEP))

VPATH=o. ../src  $(PROTO)/examples/_common



CPPFLAGS:=-DDIM=$(DIM) -DTURB=$(TURB) -I$(PROTO)/include -I$(PROTO)/include/base -I$(PROTO)/include/amr -I$(PROTO)/include/mmb -I$(PROTO)/include/ops -I$(PROTO)/include/utils -I$(PROTO)/include/tests -I$(PROTO)/include/lapack -I. -I../src -I$(PROTO)/examples/_common
CXXFLAGS:= $(CXXOPTFLAGS) -std=c++17 -w -Wno-c++11-narrowing

LIBS += $(LAPACK_LIB)

ifdef HDF5_LIB
LIBS+= -L$(HDF5_LIB) -lhdf5 -lz
CPPFLAGS+= -I$(HDF5_INC)
endif

ifdef BLIS_INC
CPPFLAGS+= -I$(BLIS_INC)
endif

ifdef FFTW3_DIR
LIBS+= -L$(FFTW3_DIR)/lib -lfftw3
CPPFLAGS+= -I$(FFTW3_DIR)/include
endif

DEP=$(DEPXX) -MM $(CXXFLAGS)
ifeq ($(DEPXX),nvcc)
DEP=$(DEPXX) -M $(CXXFLAGS)
LIBS+= -lnvToolsExt
endif

ifeq ($(DEPXX),hipcc)
CPPFLAGS+=-I$(ROCM_PATH)/rocthrust/include/ -I$(ROCM_PATH)/rocprim/include/ -I$(ROCM_PATH)/include/roctracer
CXXFLAGS+=--amdgpu-target=gfx906 -Wno-non-c-typedef-for-linkage -Wno-unused-command-line-argument
LIBS+= -L/$(ROCM_PATH)/lib/
LIBS+= -lroctx64
endif


LINK=$(CXX)


$(TARGET): $(OBJ)
	$(LINK) -o $(TARGET) $(OBJ) $(LIBS)

# Physics and boundary regression checks; use the same build flags as the solver.
$(OBJDIR)/rotating_frame_test.o: ../tests/rotating_frame.cpp | $(OBJDIR) $(DEPDIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(CPPFLAGS)
	$(DEP) $< $(CPPFLAGS) -MT $@ -MP > $(DEPDIR)/rotating_frame_test.d

$(OBJDIR)/rotating_frame_test.exe: $(filter-out $(OBJDIR)/main.o,$(OBJ)) $(OBJDIR)/rotating_frame_test.o
	$(LINK) -o $@ $^ $(LIBS)

.PHONY: test-rotation
test-rotation: $(OBJDIR)/rotating_frame_test.exe
	$(OBJDIR)/rotating_frame_test.exe

$(OBJDIR):
	@if [ ! -d "./$(OBJDIR)" ];then     \
		mkdir -p $(OBJDIR);           \
	fi

$(DEPDIR):
	@if [ ! -d "./$(DEPDIR)" ];then     \
		mkdir -p $(DEPDIR);           \
	fi

#compile non-main sources
$(OBJDIR)/%.o : %.cpp | $(OBJDIR) $(DEPDIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(CPPFLAGS)
	$(DEP) $< $(CPPFLAGS) -MT $@ -MP > $(DEPDIR)/$*.d



.PHONY: clean dataclean realclean print dirs


print-%: ; @echo $* = $($*)

-include $(DEPENDS) $(DEPDIR)/rotating_frame_test.d

vars:
	$(foreach v,$(filter-out $(VARS_OLD) VARS_OLD,$(.VARIABLES)), $(info $(v) = $($(v))))

clean:
	rm -rf $(DEPDIR) $(OBJDIR) $(TARGET)

dataclean:
	rm -rf *.hdf5 *.vtk *.table pout*

