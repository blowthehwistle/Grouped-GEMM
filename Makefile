CC = nvcc

# implementations/cuda 에서 소스 로드
CUDA_DIR = implementations/cuda
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

TARGET_FP32 = $(BIN_DIR)/cmain_4k
TARGET = $(BIN_DIR)/tmain_4k
TARGET_GEMM_BATCHED = $(BIN_DIR)/tmain_batched_4k
TARGET_GEMM_GROUPED = $(BIN_DIR)/tmain_grouped
TARGET_GEMM_GROUPED_HALF = $(BIN_DIR)/tmain_grouped_half

COMMON_DEPS = $(INCLUDE_DIR)/helpers.h $(INCLUDE_DIR)/cuda_kernels.cuh $(INCLUDE_DIR)/tensor_kernels.cuh $(INCLUDE_DIR)/runner.cuh

.PHONY: all clean grouped fp32

all: $(TARGET_FP32) $(TARGET) $(TARGET_GEMM_BATCHED) $(TARGET_GEMM_GROUPED) $(TARGET_GEMM_GROUPED_HALF)

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

grouped: $(TARGET_GEMM_GROUPED)

fp32: $(TARGET_FP32)

clean:
	rm -f $(TARGET_FP32) $(TARGET) $(TARGET_GEMM_BATCHED) $(TARGET_GEMM_GROUPED) $(TARGET_GEMM_GROUPED_HALF)
