#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Launch CUDA kernel for hyperspectral pixel classification
 * - Configures grid and block dimensions for optimal GPU utilization
 * - Each thread classifies one pixel against all reference materials
 * - Performs wavelength-based spectral interpolation
 * - Computes cosine similarity for material matching
 * - Stores best match class ID and confidence score
 * - Executes asynchronously on provided CUDA stream
 * @param d_image_data Device pointer to image data (bands × width × height)
 * @param d_image_wavelengths Device pointer to image wavelength array
 * @param d_reference_data Device pointer to reference material spectra
 * @param d_reference_wavelengths Device pointer to reference wavelengths
 * @param d_classification Device pointer to output classification array (uint16_t)
 * @param d_confidence Device pointer to output confidence scores (float)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param image_bands Number of spectral bands in image
 * @param num_references Number of reference materials
 * @param d_ref_band_counts Device pointer to array of band counts per reference
 * @param stream CUDA stream for asynchronous execution
 */
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

#endif // CUDA_KERNELS_H
