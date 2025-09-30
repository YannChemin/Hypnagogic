#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "common_types.h"

/**
 * Detect available GPU hardware and capabilities
 * - Checks for CUDA-capable NVIDIA GPUs
 * - Falls back to OpenCL detection if CUDA unavailable
 * - Reports available GPU memory
 * - Sets gpu_type flag (0=CUDA, 1=OpenCL, -1=none)
 * @param ctx Processing context to populate with GPU info
 * @return 1 if GPU detected, 0 if CPU-only
 */
int detect_gpu_capabilities(ProcessingContext* ctx);

/**
 * Transfer image and reference data to GPU memory
 * - Allocates device memory for image data
 * - Copies reference material spectra to GPU
 * - Transfers wavelength metadata
 * - Allocates output buffers for classification results
 * - Creates GPU streams/queues for asynchronous processing
 * @param ctx Processing context with loaded data
 */
void load_data_to_gpu(ProcessingContext* ctx);

#ifdef CUDA_AVAILABLE
/**
 * Classify hyperspectral image using CUDA
 * - Launches CUDA kernel for parallel pixel classification
 * - Each thread processes one pixel against all references
 * - Uses shared memory for reference data caching
 * - Computes spectral similarity using cosine distance
 * - Transfers results back to host memory
 * @param ctx Processing context with GPU-loaded data
 * @return 0 on success, -1 on error
 */
int classify_image_cuda(ProcessingContext* ctx);
#endif

#ifdef OPENCL_AVAILABLE
/**
 * Classify hyperspectral image using OpenCL
 * - Compiles and launches OpenCL kernel for classification
 * - Supports multiple GPU vendors (NVIDIA, AMD, Intel)
 * - Each work item processes one pixel
 * - Performs wavelength interpolation and similarity calculation
 * - Retrieves results from device memory
 * @param ctx Processing context with GPU-loaded data
 * @return 0 on success, -1 on error
 */
int classify_image_opencl(ProcessingContext* ctx);
#endif

#endif // GPU_UTILS_H
