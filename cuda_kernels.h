#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_cuda_classify_pixels(
    float* d_image_data, float* d_image_wavelengths,
    float* d_reference_data, float* d_reference_wavelengths,
    uint16_t* d_classification, float* d_confidence,
    int width, int height, int image_bands,
    int num_references, int* d_ref_band_counts,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
