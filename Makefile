# Makefile for Hyperspectral Vector Processing
# Usage:
#   make                    # CPU-only version with OpenMP
#   make cuda              # CUDA version (requires NVIDIA GPU and CUDA toolkit)
#   make opencl            # OpenCL version (requires OpenCL drivers)
#   make all               # Build all versions if dependencies are available

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3 -fopenmp -std=c99
NVCCFLAGS = -O3 -arch=sm_35
LDFLAGS = -lm -lgomp

# Source files
MAIN_SRC = main.c
TARGET_CPU = hypna_cpu
TARGET_CUDA = hypna_cuda
TARGET_OPENCL = hypna_opencl

# Default target - CPU only
default: $(TARGET_CPU)

# CPU-only version with OpenMP
$(TARGET_CPU): $(MAIN_SRC)
	$(CC) $(CFLAGS) -o $(TARGET_CPU) $(MAIN_SRC) $(LDFLAGS)
	@echo "Built CPU version with OpenMP support"

# CUDA version
cuda: $(TARGET_CUDA)
$(TARGET_CUDA): $(MAIN_SRC)
	@if command -v nvcc >/dev/null 2>&1; then \
		$(NVCC) $(NVCCFLAGS) -DCUDA_AVAILABLE -Xcompiler "$(CFLAGS)" \
		-o $(TARGET_CUDA) $(MAIN_SRC) -lcuda -lcudart -lcublas; \
		echo "Built CUDA version"; \
	else \
		echo "CUDA toolkit not found. Install CUDA toolkit to build CUDA version."; \
		exit 1; \
	fi

# OpenCL version
opencl: $(TARGET_OPENCL)
$(TARGET_OPENCL): $(MAIN_SRC)
	@if pkg-config --exists OpenCL 2>/dev/null; then \
		$(CC) $(CFLAGS) -DOPENCL_AVAILABLE `pkg-config --cflags OpenCL` \
		-o $(TARGET_OPENCL) $(MAIN_SRC) $(LDFLAGS) `pkg-config --libs OpenCL`; \
		echo "Built OpenCL version"; \
	else \
		echo "OpenCL not found. Install OpenCL development package."; \
		echo "Ubuntu/Debian: sudo apt-get install opencl-headers ocl-icd-opencl-dev"; \
		echo "CentOS/RHEL: sudo yum install opencl-headers ocl-icd-devel"; \
		exit 1; \
	fi

# Build all versions
all: $(TARGET_CPU)
	@echo "Building all available versions..."
	@$(MAKE) cuda 2>/dev/null || echo "Skipping CUDA version (not available)"
	@$(MAKE) opencl 2>/dev/null || echo "Skipping OpenCL version (not available)"

# Clean build artifacts
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA) $(TARGET_OPENCL)
	@echo "Cleaned build artifacts"

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	sudo apt-get update
	sudo apt-get install build-essential libomp-dev pkg-config
	@echo "Basic dependencies installed"
	@echo "For CUDA: Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
	@echo "For OpenCL: sudo apt-get install opencl-headers ocl-icd-opencl-dev"

# Install dependencies (CentOS/RHEL/Fedora)
install-deps-centos:
	sudo yum groupinstall "Development Tools"
	sudo yum install libomp-devel pkg-config
	@echo "Basic dependencies installed"
	@echo "For CUDA: Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
	@echo "For OpenCL: sudo yum install opencl-headers ocl-icd-devel"

# Test the built executables
test: $(TARGET_CPU)
	@echo "Testing CPU version..."
	./$(TARGET_CPU)
	@if [ -f $(TARGET_CUDA) ]; then \
		echo "Testing CUDA version..."; \
		./$(TARGET_CUDA); \
	fi
	@if [ -f $(TARGET_OPENCL) ]; then \
		echo "Testing OpenCL version..."; \
		./$(TARGET_OPENCL); \
	fi

# Check system capabilities
check-system:
	@echo "System capability check:"
	@echo "========================"
	@echo "CPU cores: $$(nproc)"
	@echo "OpenMP support: $$(if command -v gcc >/dev/null 2>&1; then gcc --version | head -1; else echo "GCC not found"; fi)"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected:"; \
		nvidia-smi -L; \
	else \
		echo "No NVIDIA GPU detected or nvidia-smi not available"; \
	fi
	@if command -v clinfo >/dev/null 2>&1; then \
		echo "OpenCL devices:"; \
		clinfo -l 2>/dev/null || echo "No OpenCL devices found"; \
	else \
		echo "clinfo not available (install with: sudo apt-get install clinfo)"; \
	fi

# Help target
help:
	@echo "Hyperspectral Vector Processing Build System"
	@echo "==========================================="
	@echo ""
	@echo "Targets:"
	@echo "  default        Build CPU-only version with OpenMP"
	@echo "  cuda          Build CUDA version (requires NVIDIA GPU and CUDA toolkit)"
	@echo "  opencl        Build OpenCL version (requires OpenCL)"
	@echo "  all           Build all available versions"
	@echo "  clean         Remove build artifacts"
	@echo "  test          Test built executables"
	@echo "  check-system  Check system capabilities"
	@echo ""
	@echo "Dependencies:"
	@echo "  install-deps-ubuntu   Install basic dependencies on Ubuntu/Debian"
	@echo "  install-deps-centos   Install basic dependencies on CentOS/RHEL"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                  # Build CPU version"
	@echo "  make cuda             # Build CUDA version"
	@echo "  make all              # Build all available versions"
	@echo "  make check-system     # Check what hardware is available"

.PHONY: default cuda opencl all clean test check-system help install-deps-ubuntu install-deps-centos
