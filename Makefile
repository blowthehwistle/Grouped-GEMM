CUDA_PATH = /usr/local/cuda-12.5
CC = $(CUDA_PATH)/bin/nvcc

# CUTLASS: 예) export CUTLASS_ROOT=/path/to/cutlass 또는 Makefile에서 수정
CUTLASS_ROOT = $(CURDIR)/../cutlass

# implementations/cuda 에서 소스 로드
CUDA_DIR = implementations/cuda
CUTLASS_DIR = implementations/cutlass
INCLUDE_DIR = ./include
BIN_DIR = bin
INCLUDES = -I$(INCLUDE_DIR)
FLAGS = -lcublas -arch=sm_86 -Xcompiler -fopenmp $(INCLUDES)
# grouped GEMM (nvtx 사용)
FLAGS_NVTX = $(FLAGS) -lnvToolsExt

SRC_FP32 = $(CUDA_DIR)/single_gemm.cu
SRC_TENSOR = $(CUDA_DIR)/tensor_single_gemm.cu
SRC_GEMM_BATCHED = $(CUDA_DIR)/batched_gemm.cu
SRC_GEMM_GROUPED = $(CUDA_DIR)/grouped_gemm.cu
SRC_GEMM_GROUPED_HALF = $(CUDA_DIR)/grouped_gemm_half.cu
SRC_CUTLASS_GROUPED = $(CUTLASS_DIR)/cutlass_grouped_gemm.cu

TARGET_FP32 = $(BIN_DIR)/cmain_4k
TARGET = $(BIN_DIR)/tmain_4k
TARGET_GEMM_BATCHED = $(BIN_DIR)/tmain_batched_4k
TARGET_GEMM_GROUPED = $(BIN_DIR)/tmain_grouped
TARGET_GEMM_GROUPED_HALF = $(BIN_DIR)/tmain_grouped_half
TARGET_CUTLASS_GROUPED = $(BIN_DIR)/tmain_cutlass_grouped

COMMON_DEPS = $(INCLUDE_DIR)/helpers.h $(INCLUDE_DIR)/config_io.h $(INCLUDE_DIR)/cuda_kernels.cuh $(INCLUDE_DIR)/tensor_kernels.cuh $(INCLUDE_DIR)/runner.cuh

.PHONY: all clean grouped fp32 cutlass_grouped

all: $(TARGET_FP32) $(TARGET) $(TARGET_GEMM_BATCHED) $(TARGET_GEMM_GROUPED) $(TARGET_GEMM_GROUPED_HALF)

# Cutlass grouped (CUTLASS_ROOT 필요)
cutlass_grouped: $(TARGET_CUTLASS_GROUPED)

$(TARGET_CUTLASS_GROUPED): $(SRC_CUTLASS_GROUPED) $(INCLUDE_DIR)/helpers.h | $(BIN_DIR)
	$(CC) -std=c++17 --expt-relaxed-constexpr $(SRC_CUTLASS_GROUPED) -o $@ -I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include -I$(INCLUDE_DIR) -arch=sm_80 -lcublas

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET_FP32): $(SRC_FP32) $(COMMON_DEPS) | $(BIN_DIR)
	$(CC) $(SRC_FP32) -o $@ $(FLAGS)

$(TARGET): $(SRC_TENSOR) $(COMMON_DEPS) | $(BIN_DIR)
	$(CC) $(SRC_TENSOR) -o $@ $(FLAGS)

$(TARGET_GEMM_BATCHED): $(SRC_GEMM_BATCHED) $(COMMON_DEPS) | $(BIN_DIR)
	$(CC) $(SRC_GEMM_BATCHED) -o $@ $(FLAGS)

$(TARGET_GEMM_GROUPED): $(SRC_GEMM_GROUPED) $(COMMON_DEPS) | $(BIN_DIR)
	$(CC) $(SRC_GEMM_GROUPED) -o $@ $(FLAGS_NVTX)

$(TARGET_GEMM_GROUPED_HALF): $(SRC_GEMM_GROUPED_HALF) $(COMMON_DEPS) | $(BIN_DIR)
	$(CC) $(SRC_GEMM_GROUPED_HALF) -o $@ $(FLAGS_NVTX)

grouped: $(TARGET_GEMM_GROUPED) $(TARGET_GEMM_GROUPED_HALF) $(TARGET_CUTLASS_GROUPED)

fp32: $(TARGET_FP32)

clean:
	rm -f $(TARGET_FP32) $(TARGET) $(TARGET_GEMM_BATCHED) $(TARGET_GEMM_GROUPED) $(TARGET_GEMM_GROUPED_HALF) $(TARGET_CUTLASS_GROUPED)
