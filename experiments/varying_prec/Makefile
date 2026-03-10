CC=nvcc
SRC=varying_precision_gemm.cu
TARGET=bin/main
FLAGS=-lcublas -arch=sm_89
INCLUDE_DIR=../../include

all: $(TARGET)

bin:
	mkdir -p bin

$(TARGET): $(SRC) $(INCLUDE_DIR)/helpers.h $(INCLUDE_DIR)/cuda_kernels.cuh $(INCLUDE_DIR)/tensor_kernels.cuh | bin
	$(CC) --ptxas-options=-v $(SRC) -o $(TARGET) $(FLAGS) -I$(INCLUDE_DIR)

.PHONY: clean
clean:
	rm -f $(TARGET)
