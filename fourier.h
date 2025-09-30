#ifndef FOURIER_H
#define FOURIER_H

#include <complex.h>
#include <fftw3.h>
#include "common_types.h"

// Fourier cache configuration
#define FOURIER_CACHE_VERSION 1
#define FOURIER_CACHE_DIR ".fourier_cache"

/**
 * Calculate next power of 2 for FFT padding
 * @param n Input size
 * @return Next power of 2 >= n
 */
int next_power_of_2(int n);

/**
 * Generate cache filename based on FFT parameters
 * @param fft_size FFT size (must be power of 2)
 * @param num_materials Number of reference materials
 * @return Dynamically allocated filename string (caller must free)
 */
char* generate_fourier_cache_filename(int fft_size, int num_materials);

/**
 * Save pre-computed Fourier transforms to cache file
 * @param ctx Processing context with computed FFT data
 * @param cache_file Path to cache file
 * @return 0 on success, -1 on error
 */
int save_fourier_cache(ProcessingContext* ctx, const char* cache_file);

/**
 * Load pre-computed Fourier transforms from cache file
 * @param ctx Processing context (must have allocated FFT arrays)
 * @param cache_file Path to cache file
 * @return 0 on success, -1 on error or cache mismatch
 */
int load_fourier_cache(ProcessingContext* ctx, const char* cache_file);

/**
 * Initialize Fourier processing resources
 * - Determines optimal FFT size
 * - Allocates FFT buffers for each thread
 * - Creates FFTW plans for parallel processing
 * @param ctx Processing context
 * @return 0 on success, -1 on error
 */
int setup_fourier_processing(ProcessingContext* ctx);

/**
 * Pre-compute Fourier transforms for all reference materials
 * - Interpolates reference spectra to image wavelength grid
 * - Computes FFT for each reference
 * - Calculates magnitude and energy normalization
 * @param ctx Processing context
 */
void precompute_fourier_references(ProcessingContext* ctx);

/**
 * Calculate optimized Fourier-based similarity between pixel and reference
 * Uses magnitude correlation in frequency domain for robust matching
 * @param pixel_fft Pixel FFT (complex values)
 * @param ref_fft Reference FFT (complex values)
 * @param pixel_mag Pixel FFT magnitudes
 * @param ref_mag Reference FFT magnitudes
 * @param fft_size Number of frequency bins
 * @return Similarity score [0.0, 1.0]
 */
float calculate_fourier_similarity_optimized(float complex* pixel_fft, float complex* ref_fft,
                                            float* pixel_mag, float* ref_mag, int fft_size);

/**
 * Classify entire hyperspectral image using Fourier transforms
 * - Multi-threaded CPU implementation with OpenMP
 * - Each thread maintains its own FFT workspace
 * - Uses pre-computed reference FFTs for efficiency
 * @param ctx Processing context
 * @return 0 on success, -1 on error
 */
int classify_image_fourier_cpu(ProcessingContext* ctx);

/**
 * Clean up all Fourier processing resources
 * - Frees FFT buffers and plans
 * - Releases FFTW thread resources
 * @param ctx Processing context
 */
void cleanup_fourier_resources(ProcessingContext* ctx);

#endif // FOURIER_H
