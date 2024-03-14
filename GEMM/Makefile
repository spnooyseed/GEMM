# 定义编译器和编译选项
NVCC = nvcc
NVCCFLAGS = -std=c++11 -lcublas # 根据您的CUDA架构替换sm_XX

# 定义目标目录
BUILD_DIR = build

# 定义源文件列表
SRCS := $(wildcard *.cu)

# 定义目标文件列表，将源文件替换为build目录下的可执行文件
TARGETS := $(SRCS:%.cu=$(BUILD_DIR)/%)

# 默认目标
all: clean $(TARGETS) run

# 编译生成可执行文件
$(BUILD_DIR)/%: %.cu
	@mkdir -p $(BUILD_DIR) # 确保build目录存在
	$(NVCC) -o $@ $< $(NVCCFLAGS)

# 清理目标目录
clean:
	rm -rf $(BUILD_DIR)
run:
	$(foreach exec, $(TARGETS), $(exec) 2048 2048 2048;)

# 伪目标，表示依赖文件
.PHONY: all clean run