# Makefile for Hyperspectral Image Processing with GDAL
# Usage:
#   make                    # CPU-only version with OpenMP and GDAL
#   make cuda              # CUDA version (requires NVIDIA GPU and CUDA toolkit)
#   make opencl            # OpenCL version (requires OpenCL drivers)
#   make all               # Build all versions if dependencies are available

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3 -fopenmp -std=c99
NVCCFLAGS = -O3 -arch=sm_35

# GDAL flags
GDAL_CFLAGS = $(shell gdal-config --cflags)
GDAL_LDFLAGS = $(shell gdal-config --libs)

LDFLAGS = -lm -lgomp $(GDAL_LDFLAGS)

# Source files
MAIN_SRC = main.c
HEADER_FILE = reference_materials.h
TARGET_CPU = hypna_cpu
TARGET_CUDA = hypna_cuda
TARGET_OPENCL = hypna_opencl

# Check GDAL availability
GDAL_CHECK = $(shell command -v gdal-config >/dev/null 2>&1 && echo "yes" || echo "no")

# Default target - CPU only
default: check-gdal $(TARGET_CPU)

check-gdal:
	@if [ "$(GDAL_CHECK)" = "no" ]; then \
		echo "Error: GDAL not found. Please install GDAL development package."; \
		echo "Ubuntu/Debian: sudo apt-get install libgdal-dev gdal-bin"; \
		echo "CentOS/RHEL: sudo yum install gdal-devel"; \
		echo "macOS: brew install gdal"; \
		exit 1; \
	fi
	@echo "GDAL found: $$(gdal-config --version)"

# CPU-only version with OpenMP and GDAL
$(TARGET_CPU): $(MAIN_SRC) $(HEADER_FILE)
	$(CC) $(CFLAGS) $(GDAL_CFLAGS) -o $(TARGET_CPU) $(MAIN_SRC) $(LDFLAGS)
	@echo "Built CPU version with OpenMP and GDAL support"

# CUDA version
cuda: check-gdal $(TARGET_CUDA)
$(TARGET_CUDA): $(MAIN_SRC) $(HEADER_FILE)
	@if command -v nvcc >/dev/null 2>&1; then \
		$(NVCC) $(NVCCFLAGS) -DCUDA_AVAILABLE -Xcompiler "$(CFLAGS) $(GDAL_CFLAGS)" \
		-o $(TARGET_CUDA) $(MAIN_SRC) -lcuda -lcudart -lcublas $(GDAL_LDFLAGS) -lm; \
		echo "Built CUDA version with GDAL support"; \
	else \
		echo "CUDA toolkit not found. Install CUDA toolkit to build CUDA version."; \
		exit 1; \
	fi

# OpenCL version
opencl: check-gdal $(TARGET_OPENCL)
$(TARGET_OPENCL): $(MAIN_SRC) $(HEADER_FILE)
	@if pkg-config --exists OpenCL 2>/dev/null; then \
		$(CC) $(CFLAGS) $(GDAL_CFLAGS) -DOPENCL_AVAILABLE $$(pkg-config --cflags OpenCL) \
		-o $(TARGET_OPENCL) $(MAIN_SRC) $(LDFLAGS) $$(pkg-config --libs OpenCL); \
		echo "Built OpenCL version with GDAL support"; \
	else \
		echo "OpenCL not found. Install OpenCL development package."; \
		echo "Ubuntu/Debian: sudo apt-get install opencl-headers ocl-icd-opencl-dev"; \
		echo "CentOS/RHEL: sudo yum install opencl-headers ocl-icd-devel"; \
		exit 1; \
	fi

# Build all versions
all: check-gdal $(TARGET_CPU)
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
	sudo apt-get install libgdal-dev gdal-bin
	@echo "Basic dependencies and GDAL installed"
	@echo "For CUDA: Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
	@echo "For OpenCL: sudo apt-get install opencl-headers ocl-icd-opencl-dev"

# Install dependencies (CentOS/RHEL/Fedora)
install-deps-centos:
	sudo yum groupinstall "Development Tools"
	sudo yum install libomp-devel pkg-config
	sudo yum install gdal-devel gdal
	@echo "Basic dependencies and GDAL installed"
	@echo "For CUDA: Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
	@echo "For OpenCL: sudo yum install opencl-headers ocl-icd-devel"

# Install dependencies (macOS)
install-deps-macos:
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "Homebrew not found. Please install from https://brew.sh/"; \
		exit 1; \
	fi
	brew install gcc libomp gdal
	@echo "Dependencies installed via Homebrew"

# Test the built executables
test: $(TARGET_CPU)
	@echo "Testing CPU version..."
	@if [ ! -f "hyper.tif" ]; then \
		echo "Warning: hyper.tif not found. Please provide a hyperspectral GeoTIFF image."; \
		echo "Creating test with small synthetic image..."; \
		gdal_create -of GTiff -outsize 100 100 -bands 50 test_hyper.tif; \
		./$(TARGET_CPU) test_hyper.tif test_output.tif; \
	else \
		./$(TARGET_CPU) hyper.tif classification.tif; \
	fi
	@if [ -f $(TARGET_CUDA) ]; then \
		echo "Testing CUDA version..."; \
		./$(TARGET_CUDA) hyper.tif classification_cuda.tif 2>/dev/null || echo "CUDA test failed"; \
	fi
	@if [ -f $(TARGET_OPENCL) ]; then \
		echo "Testing OpenCL version..."; \
		./$(TARGET_OPENCL) hyper.tif classification_opencl.tif 2>/dev/null || echo "OpenCL test failed"; \
	fi

# Check system capabilities
check-system:
	@echo "System capability check:"
	@echo "========================"
	@echo "CPU cores: $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")"
	@echo "OpenMP support: $$(if command -v gcc >/dev/null 2>&1; then gcc --version | head -1; else echo "GCC not found"; fi)"
	@if command -v gdal-config >/dev/null 2>&1; then \
		echo "GDAL version: $$(gdal-config --version)"; \
	else \
		echo "GDAL not found"; \
	fi
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
	@echo "Hyperspectral Image Processing Build System"
	@echo "==========================================="
	@echo ""
	@echo "Targets:"
	@echo "  default        Build CPU-only version with OpenMP and GDAL"
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
	@echo "  install-deps-macos    Install basic dependencies on macOS"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                  # Build CPU version"
	@echo "  make cuda             # Build CUDA version"
	@echo "  make all              # Build all available versions"
	@echo "  make check-system     # Check what hardware is available"

.PHONY: default cuda opencl all clean test check-system help install-deps-ubuntu install-deps-centos install-deps-macos check-gdal