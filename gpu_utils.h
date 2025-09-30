#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "common_types.h"

// GPU detection and initialization
int detect_gpu_capabilities(ProcessingContext* ctx);
void load_data_to_gpu(ProcessingContext* ctx);

// GPU classification functions
#ifdef CUDA_AVAILABLE
int classify_image_cuda(ProcessingContext* ctx);
#endif

#ifdef OPENCL_AVAILABLE
int classify_image_opencl(ProcessingContext* ctx);
#endif

#endif // GPU_UTILS_H
