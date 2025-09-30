// common_types.h
#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>
#include <fftw3.h>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

#ifdef OPENCL_AVAILABLE
#include <CL/cl.h>
#endif

// Forward declarations
typedef struct HyperspectralVector HyperspectralVector;
typedef struct HyperspectralImage HyperspectralImage;
typedef struct ClassificationResult ClassificationResult;
typedef struct ProcessingContext ProcessingContext;

// Full structure definitions
struct HyperspectralVector {
    float* wavelengths;
    float* reflectance;
    int size;
    int material;
    char name[64];
};

struct HyperspectralImage {
    float* data;
    float* wavelengths;
    int width;
    int height;
    int bands;
    double geotransform[6];
    char* projection;
    size_t data_size;
};

struct ClassificationResult {
    uint16_t* classification;
    float* confidence;
    int width;
    int height;
    double geotransform[6];
    char* projection;
};

struct ProcessingContext {
    HyperspectralVector* reference_vectors;
    int num_reference_vectors;
    int use_gpu;
    int gpu_type;
    size_t available_gpu_memory;
    
    HyperspectralImage* image;
    ClassificationResult* result;
    
    // Fourier-specific additions
    int use_fourier;
    int fft_size;
    float complex** reference_fft;
    float** reference_fft_mag;
    float* reference_energies;
    
    // Thread-local FFT resources
    fftwf_plan* thread_fft_plans;
    float** thread_fft_input;
    float complex** thread_fft_output;
    float** thread_fft_mag;
    int num_threads;
    
    #ifdef CUDA_AVAILABLE
    float* d_reference_data;
    float* d_image_data;
    float* d_wavelengths;
    uint16_t* d_classification;
    float* d_confidence;
    cudaStream_t stream;
    #endif
    
    #ifdef OPENCL_AVAILABLE
    cl_context cl_context;
    cl_command_queue cl_queue;
    cl_program cl_program;
    cl_kernel cl_classify_kernel;
    cl_mem cl_reference_buffer;
    cl_mem cl_image_buffer;
    cl_mem cl_wavelength_buffer;
    cl_mem cl_classification_buffer;
    cl_mem cl_confidence_buffer;
    #endif
};

#endif // COMMON_TYPES_H
