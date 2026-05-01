BUILD_DIR := ./bin
SRC_DIR := ./src

NVCC_FLAGS := --ptxas-options=-v
# Debug mode can be enabled by running: make DEBUG=1
DEBUG ?= 0
ifeq ($(DEBUG), 1)
    NVCC_FLAGS += -lineinfo
    BIN_SUFFIX := _debug
else
    BIN_SUFFIX :=
endif

PTX ?= 0
ifeq ($(PTX), 1)
    NVCC_FLAGS += -ptx
    BIN_SUFFIX := .ptx
else
    BIN_SUFFIX :=
endif



all: directories dynamic_parallelism device_info mat_mul mat_mul_cublas attention saxpy, atomic-non-determinism

directories:
	mkdir -p $(BUILD_DIR)

dynamic_parallelism: $(SRC_DIR)/dynamic_parallelism/main.cu
	nvcc -rdc=true $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@$(BIN_SUFFIX)

device_info: $(SRC_DIR)/device_info/main.cu
	nvcc $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@$(BIN_SUFFIX)

mat_mul: $(SRC_DIR)/mat_mul/main.cu
	$(info Using NVCC_FLAGS: $(NVCC_FLAGS))
	nvcc $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@$(BIN_SUFFIX)

mat_mul_cublas: $(SRC_DIR)/mat_mul_cublas/main.cu
	nvcc $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@$(BIN_SUFFIX) -lcublas

saxpy: $(SRC_DIR)/saxpy/main.cu
	nvcc $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@$(BIN_SUFFIX)

attention: $(SRC_DIR)/attention/main.cu
	nvcc $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@$(BIN_SUFFIX)

atomic-non-determinism: $(SRC_DIR)/atomic-non-determinism/main.cu
	nvcc $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@$(BIN_SUFFIX)
# Debug target - builds everything with DEBUG=1
debug:
	$(MAKE) DEBUG=1

ptx:
	$(MAKE) PTX=1
clean:
	rm -rf $(BUILD_DIR)
