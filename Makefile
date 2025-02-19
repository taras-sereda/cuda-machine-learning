
BUILD_DIR := ./bin
SRC_DIR := ./src

all: directories dynamic_parallelism device_info mat_mul attention

directories:
	mkdir -p bin

dynamic_parallelism: $(SRC_DIR)/dynamic_parallelism/main.cu
	nvcc -rdc=true $(SRC_DIR)/dynamic_parallelism/main.cu -o $(BUILD_DIR)/dynamic_parallelism

device_info: $(SRC_DIR)/device_info/main.cu
	nvcc $(SRC_DIR)/device_info/main.cu -o $(BUILD_DIR)/device_info

mat_mul: $(SRC_DIR)/mat_mul/main.cu
	nvcc $(SRC_DIR)/mat_mul/main.cu -o $(BUILD_DIR)/mat_mul

attention: $(SRC_DIR)/attention/main.cu
	nvcc $(SRC_DIR)/attention/main.cu -o $(BUILD_DIR)/attention

clean:
	rm -rf $(BUILD_DIR)