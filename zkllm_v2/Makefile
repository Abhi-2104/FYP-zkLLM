# Compilers
NVCC := /usr/local/cuda-12.5/bin/nvcc

# Include and library paths
# Assuming conda include/lib is still needed for some headers/libs
INCLUDES := -I$(CONDA_PREFIX)/include -I.
LIBS := -L$(CONDA_PREFIX)/lib -lcudart -lstdc++

# get compute capability from retrieved value
ARCH := sm_86

# NVCC compiler flags
NVCC_FLAGS := -arch=$(ARCH) -std=c++17 --compiler-bindir=/usr/bin/gcc-12

# Source and object files
CU_SRCS := bls12-381.cu ioutils.cu commitment_v2.cu fr-tensor.cu g1-tensor.cu proof_v2.cu polynomial.cu rescaling_v2.cu proof_io_v2.cu tlookup_v2.cu zkfc_v2.cu
CU_OBJS := $(CU_SRCS:.cu=.o)
TARGETS := self-attn_v2 ffn_v2 rmsnorm_v2 skip-connection_v2
TARGET_OBJS := $(TARGETS:=.o)

# Pattern rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# Pattern rule for C++ source files
%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# General rule for building each target
$(TARGETS): % : %.o $(CU_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@ -dlto

# Clean rule
clean:
	rm -f $(TARGET_OBJS) $(CU_OBJS) $(CPP_OBJS) $(TARGETS)

# Default rule
all: $(TARGETS)