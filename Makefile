HIPCC ?= hipcc
NVCC ?= nvcc
CUDA_ARCH ?=
CUDA_OUT ?= rdna35_wmma_cuda
CUDAFLAGS ?= -O2 -std=c++17 -Xcompiler=-pthread,-fno-fast-math,-ffp-contract=off
CXX ?= c++
ARCH ?= gfx1151
GPU_OUT ?= rdna35_wmma
CPU_OUT ?= rdna35_wmma_cpu
CXXFLAGS ?= -O2 -std=c++17 -pthread -fno-fast-math -ffp-contract=off
SOURCE_SHA256 := $(shell sha256sum rdna35_wmma.hip | cut -d' ' -f1)
HASH_FLAG = '-DWMMA_SOURCE_SHA256="$(SOURCE_SHA256)"'

.PHONY: all gpu cuda cpu clean
all: gpu
gpu: $(GPU_OUT)
cuda: $(CUDA_OUT)
cpu: $(CPU_OUT)
$(GPU_OUT): rdna35_wmma.hip Makefile
	$(HIPCC) $(CXXFLAGS) --offload-arch=$(ARCH) -mno-wavefrontsize64 $(HASH_FLAG) $< -o $@
$(CUDA_OUT): rdna35_wmma.hip Makefile
	$(NVCC) $(CUDAFLAGS) $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH)) $(HASH_FLAG) -x cu $< -o $@
$(CPU_OUT): rdna35_wmma.hip Makefile
	$(CXX) $(CXXFLAGS) -DWMMA_CPU_ONLY $(HASH_FLAG) -x c++ $< -o $@
clean:
	rm -f $(GPU_OUT) $(CUDA_OUT) $(CPU_OUT)
