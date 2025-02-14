
BUILD_DIR := ./bin
SRC_DIR := ./src

all: directories dynamic_parallelism device_info

directories:
	mkdir -p bin

dynamic_parallelism: $(SRC_DIR)/dynamic_parallelism/main.cu
	nvcc -rdc=true $(SRC_DIR)/dynamic_parallelism/main.cu -o $(BUILD_DIR)/dynamic_parallelism

device_info: $(SRC_DIR)/device_info/main.cu
	nvcc $(SRC_DIR)/device_info/main.cu -o $(BUILD_DIR)/device_info

clean:
	rm -rf $(BUILD_DIR)