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

#define GPU_MEMORY_THRESHOLD (2 * 1024 * 1024 * 1024) // 2GB

// Forward declarations
typedef struct HyperspectralVector HyperspectralVector;
typedef struct HyperspectralImage HyperspectralImage;
typedef struct ClassificationResult ClassificationResult;
typedef struct ProcessingContext ProcessingContext;

/**
 * Hyperspectral reference material spectrum
 * Stores wavelength-reflectance pairs for a single material type
 */
struct HyperspectralVector {
    float* wavelengths;      ///< Array of wavelengths in nanometers
    float* reflectance;      ///< Array of reflectance values [0.0, 1.0]
    int size;                ///< Number of spectral samples
    int material;            ///< Material type ID (enum MaterialType)
    char name[64];           ///< Human-readable material name
};

/**
 * Multi-band hyperspectral image with geospatial metadata
 * Data stored in band-sequential (BSQ) format
 */
struct HyperspectralImage {
    float* data;             ///< Image data array [bands × width × height]
    float* wavelengths;      ///< Array of wavelengths for each band
    int width;               ///< Image width in pixels
    int height;              ///< Image height in pixels
    int bands;               ///< Number of spectral bands
    double geotransform[6];  ///< GDAL geotransform coefficients
    char* projection;        ///< WKT projection string
    size_t data_size;        ///< Total size of data array in bytes
};

/**
 * Classification result with confidence scores
 * Stores per-pixel class assignments and similarity scores
 */
struct ClassificationResult {
    uint16_t* classification; ///< Per-pixel material class ID (supports >256 classes)
    float* confidence;        ///< Per-pixel classification confidence [0.0, 1.0]
    int width;                ///< Result width in pixels
    int height;               ///< Result height in pixels
    double geotransform[6];   ///< GDAL geotransform (copied from input)
    char* projection;         ///< WKT projection (copied from input)
};

/**
 * Main processing context structure
 * Holds all data needed for classification workflow
 */
struct ProcessingContext {
    // Reference material library
    HyperspectralVector* reference_vectors;  ///< Array of reference spectra
    int num_reference_vectors;               ///< Number of reference materials
    
    // GPU configuration
    int use_gpu;                  ///< Flag: 1 if using GPU, 0 for CPU
    int gpu_type;                 ///< GPU type: 0=CUDA, 1=OpenCL, -1=none
    size_t available_gpu_memory;  ///< Available GPU memory in bytes
    
    // Image data
    HyperspectralImage* image;      ///< Input hyperspectral image
    ClassificationResult* result;   ///< Output classification result
    
    // Fourier transform configuration
    int use_fourier;                ///< Flag: 1 to use Fourier classification
    int fft_size;                   ///< FFT size (power of 2, ≥ image bands)
    float complex** reference_fft;  ///< Pre-computed reference FFTs
    float** reference_fft_mag;      ///< Pre-computed FFT magnitudes
    float* reference_energies;      ///< Pre-computed spectral energies
    int classification_mode;  // 0=spatial, 1=fourier_baseline, 2=fourier_quality, 3=fourier_fastest
    
    // Thread-local FFT resources for parallel processing
    fftwf_plan* thread_fft_plans;      ///< FFTW plans (one per thread)
    float** thread_fft_input;          ///< Input buffers (one per thread)
    float complex** thread_fft_output; ///< Output buffers (one per thread)
    float** thread_fft_mag;            ///< Magnitude buffers (one per thread)
    int num_threads;                   ///< Number of OpenMP threads
    
    #ifdef CUDA_AVAILABLE
    // CUDA device memory pointers
    float* d_reference_data;      ///< Device reference spectra
    float* d_image_data;          ///< Device image data
    float* d_wavelengths;         ///< Device wavelength array
    uint16_t* d_classification;   ///< Device classification output
    float* d_confidence;          ///< Device confidence output
    cudaStream_t stream;          ///< CUDA stream for async operations
    #endif
    
    #ifdef OPENCL_AVAILABLE
    // OpenCL context and resources
    cl_context cl_context;                 ///< OpenCL context
    cl_command_queue cl_queue;             ///< OpenCL command queue
    cl_program cl_program;                 ///< Compiled OpenCL program
    cl_kernel cl_classify_kernel;          ///< Classification kernel
    cl_mem cl_reference_buffer;            ///< Device reference buffer
    cl_mem cl_image_buffer;                ///< Device image buffer
    cl_mem cl_wavelength_buffer;           ///< Device wavelength buffer
    cl_mem cl_classification_buffer;       ///< Device classification buffer
    cl_mem cl_confidence_buffer;           ///< Device confidence buffer
    #endif
};

#endif // COMMON_TYPES_H
