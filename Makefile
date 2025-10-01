# Makefile for Hyperspectral Image Processing with GDAL and FFTW3
# Usage:
#   make                    # CPU-only version with OpenMP, GDAL, and FFTW3
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

# FFTW3 flags
FFTW3_LDFLAGS = -lfftw3f -lfftw3f_threads

LDFLAGS = -lm -lgomp $(GDAL_LDFLAGS) $(FFTW3_LDFLAGS)

# Source files
MAIN_SRC = main.c gis_export.c gpu_utils.c io_utils.c fourier.c
#HEADER_FILE = reference_materials.h gis_export.h common_types.h material_colors.h gpu_utils.h io_utils.h fourier.h
HEADER_FILE = enhanced_reference_materials.h gis_export.h common_types.h material_colors.h gpu_utils.h io_utils.h fourier.h
TARGET_CPU = hypna_cpu
TARGET_CUDA = hypna_cuda
TARGET_OPENCL = hypna_opencl

# Check GDAL availability
GDAL_CHECK = $(shell command -v gdal-config >/dev/null 2>&1 && echo "yes" || echo "no")

# Check FFTW3 availability
FFTW3_CHECK = $(shell pkg-config --exists fftw3f 2>/dev/null && echo "yes" || echo "no")

# Default target - CPU only
default: check-gdal check-fftw3 $(TARGET_CPU)

check-gdal:
	@if [ "$(GDAL_CHECK)" = "no" ]; then \
		echo "Error: GDAL not found. Please install GDAL development package."; \
		echo "Ubuntu/Debian: sudo apt-get install libgdal-dev gdal-bin"; \
		echo "CentOS/RHEL: sudo yum install gdal-devel"; \
		echo "macOS: brew install gdal"; \
		exit 1; \
	fi
	@echo "GDAL found: $$(gdal-config --version)"

check-fftw3:
	@if [ "$(FFTW3_CHECK)" = "no" ]; then \
		echo "Warning: FFTW3 not found via pkg-config, checking libraries..."; \
		if ! ldconfig -p | grep -q libfftw3f.so; then \
			echo "Error: FFTW3 not found. Please install FFTW3 development package."; \
			echo "Ubuntu/Debian: sudo apt-get install libfftw3-dev"; \
			echo "CentOS/RHEL: sudo yum install fftw-devel"; \
			echo "macOS: brew install fftw"; \
			exit 1; \
		fi; \
	fi
	@echo "FFTW3 found"

# CPU-only version with OpenMP, GDAL, and FFTW3
$(TARGET_CPU): $(MAIN_SRC) $(HEADER_FILE)
	$(CC) $(CFLAGS) $(GDAL_CFLAGS) -o $(TARGET_CPU) $(MAIN_SRC) $(LDFLAGS)
	@echo "Built CPU version with OpenMP, GDAL, and FFTW3 support"

# CUDA version - separate compilation
cuda: check-gdal check-fftw3
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "Compiling CUDA kernels..."; \
		nvcc -c cuda_kernels.cu -o cuda_kernels.o $(NVCCFLAGS) -Xcompiler "-fPIC"; \
		echo "Compiling main C code..."; \
		$(CC) -c $(MAIN_SRC) -o main.o $(CFLAGS) $(GDAL_CFLAGS) -DCUDA_AVAILABLE; \
		echo "Linking..."; \
		$(CC) main.o cuda_kernels.o -o $(TARGET_CUDA) \
			-fopenmp \
			-L/usr/local/cuda/lib64 -lcudart -lcublas -lcufft \
			$(GDAL_LDFLAGS) $(FFTW3_LDFLAGS) -lm -lstdc++; \
		echo "Built CUDA version with GDAL and FFTW3 support"; \
	else \
		echo "CUDA toolkit not found. Install CUDA toolkit to build CUDA version."; \
		exit 1; \
	fi

# OpenCL version
opencl: check-gdal check-fftw3 $(TARGET_OPENCL)
$(TARGET_OPENCL): $(MAIN_SRC) $(HEADER_FILE)
	@if pkg-config --exists OpenCL 2>/dev/null; then \
		$(CC) $(CFLAGS) $(GDAL_CFLAGS) -DOPENCL_AVAILABLE $$(pkg-config --cflags OpenCL) \
		-o $(TARGET_OPENCL) $(MAIN_SRC) $(LDFLAGS) $$(pkg-config --libs OpenCL); \
		echo "Built OpenCL version with GDAL and FFTW3 support"; \
	else \
		echo "OpenCL not found. Install OpenCL development package."; \
		echo "Ubuntu/Debian: sudo apt-get install opencl-headers ocl-icd-opencl-dev"; \
		echo "CentOS/RHEL: sudo yum install opencl-headers ocl-icd-devel"; \
		exit 1; \
	fi

# Build all versions
all: check-gdal check-fftw3 $(TARGET_CPU)
	@echo "Building all available versions..."
	@$(MAKE) cuda 2>/dev/null || echo "Skipping CUDA version (not available)"
	@$(MAKE) opencl 2>/dev/null || echo "Skipping OpenCL version (not available)"

# Clean build artifacts
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA) $(TARGET_OPENCL)
	rm -f main.o·gis_export.o·gpu_utils.o·io_utils.o·fourier.o
	rm -rf .fourier_cache
	@echo "Cleaned build artifacts and Fourier cache"

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	sudo apt-get update
	sudo apt-get install build-essential libomp-dev pkg-config
	sudo apt-get install libgdal-dev gdal-bin
	sudo apt-get install libfftw3-dev libfftw3-single3
	@echo "Basic dependencies, GDAL, and FFTW3 installed"
	@echo "For CUDA: Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
	@echo "For OpenCL: sudo apt-get install opencl-headers ocl-icd-opencl-dev"

# Install dependencies (CentOS/RHEL/Fedora)
install-deps-centos:
	sudo yum groupinstall "Development Tools"
	sudo yum install libomp-devel pkg-config
	sudo yum install gdal-devel gdal
	sudo yum install fftw-devel fftw
	@echo "Basic dependencies, GDAL, and FFTW3 installed"
	@echo "For CUDA: Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
	@echo "For OpenCL: sudo yum install opencl-headers ocl-icd-devel"

# Install dependencies (macOS)
install-deps-macos:
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "Homebrew not found. Please install from https://brew.sh/"; \
		exit 1; \
	fi
	brew install gcc libomp gdal fftw
	@echo "Dependencies installed via Homebrew"

# Test the built executables
test: $(TARGET_CPU)
	@echo "Testing CPU version with Fourier support..."
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
	@if pkg-config --exists fftw3f 2>/dev/null; then \
		echo "FFTW3 version: $$(pkg-config --modversion fftw3f)"; \
	elif ldconfig -p | grep -q libfftw3f.so; then \
		echo "FFTW3 installed (version unknown)"; \
	else \
		echo "FFTW3 not found"; \
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
	@echo "  default        Build CPU-only version with OpenMP, GDAL, and FFTW3"
	@echo "  cuda          Build CUDA version (requires NVIDIA GPU and CUDA toolkit)"
	@echo "  opencl        Build OpenCL version (requires OpenCL)"
	@echo "  all           Build all available versions"
	@echo "  clean         Remove build artifacts and Fourier cache"
	@echo "  test          Test built executables"
	@echo "  check-system  Check system capabilities"
	@echo ""
	@echo "Dependencies:"
	@echo "  install-deps-ubuntu   Install basic dependencies on Ubuntu/Debian"
	@echo "  install-deps-centos   Install basic dependencies on CentOS/RHEL"
	@echo "  install-deps-macos    Install basic dependencies on macOS"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                  # Build CPU version with Fourier support"
	@echo "  make cuda             # Build CUDA version"
	@echo "  make all              # Build all available versions"
	@echo "  make check-system     # Check what hardware is available"

.PHONY: default cuda opencl all clean test check-system help install-deps-ubuntu install-deps-centos install-deps-macos check-gdal check-fftw3
