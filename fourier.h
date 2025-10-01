#ifndef FOURIER_H
#define FOURIER_H

#include <complex.h>
#include <fftw3.h>
#include "common_types.h"

// Fourier cache configuration
#define FOURIER_CACHE_VERSION 1
#define FOURIER_CACHE_DIR ".fourier_cache"

/**
 * Structure to hold candidate material matches with their similarity scores
 * 
 * Used in spatial coherence refinement to track multiple plausible material
 * classifications per pixel. Enables post-processing refinement based on
 * neighborhood context without recomputing expensive Fourier transforms.
 * 
 * Typical usage: Store top-3 candidates per pixel, then refine classification
 * based on which candidate has strongest spatial support from neighbors.
 * 
 * @field material_id Index into ctx->reference_vectors array [0, num_materials)
 * @field similarity Fourier similarity score [0.0, 1.0], higher = better match
 *   Computed via calculate_fourier_similarity_optimized()
 */
typedef struct {
    int material_id;
    float similarity;
} CandidateMatch;

/**
 * Calculate next power of 2 for FFT padding
 * - Ensures FFT size is optimal for FFTW performance
 * - Pads input data with zeros to reach power of 2
 * @param n Input size
 * @return Next power of 2 >= n
 */
int next_power_of_2(int n);

/**
 * Generate cache filename based on FFT parameters
 * - Creates cache directory if it doesn't exist
 * - Filename includes FFT size, material count, and version
 * - Allows automatic cache invalidation on parameter changes
 * @param fft_size FFT size (must be power of 2)
 * @param num_materials Number of reference materials
 * @return Dynamically allocated filename string (caller must free)
 */
char* generate_fourier_cache_filename(int fft_size, int num_materials);

/**
 * Save pre-computed Fourier transforms to cache file
 * - Writes header with version and parameters for validation
 * - Stores FFT data, magnitudes, and energy for each material
 * - Enables fast loading on subsequent runs
 * @param ctx Processing context with computed FFT data
 * @param cache_file Path to cache file
 * @return 0 on success, -1 on error
 */
int save_fourier_cache(ProcessingContext* ctx, const char* cache_file);

/**
 * Load pre-computed Fourier transforms from cache file
 * - Validates cache version and parameters match current setup
 * - Skips expensive FFT computation if cache is valid
 * - Returns error if cache is missing or incompatible
 * @param ctx Processing context (must have allocated FFT arrays)
 * @param cache_file Path to cache file
 * @return 0 on success, -1 on error or cache mismatch
 */
int load_fourier_cache(ProcessingContext* ctx, const char* cache_file);

/**
 * Initialize Fourier processing resources
 * - Determines optimal FFT size (next power of 2)
 * - Allocates FFT buffers for each thread
 * - Creates FFTW plans for parallel processing
 * - Initializes thread-local workspaces for efficiency
 * @param ctx Processing context
 * @return 0 on success, -1 on error
 */
int setup_fourier_processing(ProcessingContext* ctx);

/**
 * Pre-compute Fourier transforms for all reference materials
 * - Interpolates reference spectra to image wavelength grid
 * - Computes FFT for each reference material
 * - Calculates magnitude and energy normalization
 * - Enables fast pixel-by-pixel comparison during classification
 * @param ctx Processing context
 */
void precompute_fourier_references(ProcessingContext* ctx);

/**
 * Calculate optimized Fourier-based similarity between pixel and reference
 * - Uses magnitude correlation in frequency domain for robust matching
 * - Skips DC component for better material discrimination
 * - Normalized by energy for scale-invariant comparison
 * - More robust to spectral shifts than spatial domain methods
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
 * - Dynamic scheduling for load balancing across threads
 * @param ctx Processing context
 * @return 0 on success, -1 on error
 */
int classify_image_fourier_cpu(ProcessingContext* ctx);

/**
 * Clean up all Fourier processing resources
 * - Frees FFT buffers and plans
 * - Releases FFTW thread resources
 * - Safe to call even if Fourier processing not enabled
 * @param ctx Processing context
 */
void cleanup_fourier_resources(ProcessingContext* ctx);

// ------- Fourier with spatial coherence boost -----


/**
 * Classify hyperspectral image using two-phase spatial coherence refinement
 * 
 * This optimized version improves classification accuracy by leveraging spatial
 * coherence in hyperspectral images (e.g., contiguous agricultural fields, soil patches).
 * 
 * Algorithm:
 * Phase 1: Compute top-3 candidate materials for each pixel using Fourier similarity
 *   - Full spectral comparison against all reference materials
 *   - Stores best 3 matches with their similarity scores
 *   - Provides fallback options for spatial refinement
 * 
 * Phase 2: Spatial coherence refinement using neighbor voting
 *   - Examines 3x3 neighborhood around each pixel
 *   - Counts votes for each of the pixel's top-3 candidates
 *   - Reclassifies if alternative candidate has strong neighbor support (≥3 votes)
 *   - Combined score: 80% spectral similarity + 20% spatial coherence
 *   - No additional FFT computation required
 * 
 * Performance characteristics:
 *   - Memory overhead: ~36 bytes per pixel (3 candidates × 12 bytes)
 *   - Time overhead: ~5-10% vs. standard classification
 *   - Accuracy improvement: Reduces salt-and-pepper noise significantly
 *   - Preserves fine boundaries while smoothing homogeneous regions
 * 
 * Best for: Agricultural imagery, geological surveys, land cover mapping
 * 
 * @param ctx Processing context with:
 *   - Pre-computed reference FFTs (via precompute_fourier_references)
 *   - Thread-local FFT workspaces (via setup_fourier_processing)
 *   - Allocated classification result buffer
 * @return 0 on success, -1 on error
 *   Falls back to classify_image_fourier_cpu if candidate buffer allocation fails
 * 
 * @see classify_image_fourier_cpu for standard classification
 * @see classify_image_fourier_cpu_light for ultra-lightweight alternative
 */
int classify_image_fourier_cpu_optimized(ProcessingContext* ctx);

/**
 * Classify hyperspectral image using lightweight neighbor-aware prediction
 * 
 * Ultra-fast classification that exploits spatial continuity to skip expensive
 * comparisons in homogeneous regions. Ideal for real-time or large-scale processing.
 * 
 * Algorithm:
 *   1. For each pixel, check if left and top neighbors agree on classification
 *   2. If neighbors agree, test only that material first
 *   3. If neighbor material similarity > 0.85, use it immediately (early exit)
 *   4. Otherwise, perform full search through all reference materials
 * 
 * Optimization strategy:
 *   - In homogeneous regions: Skips ~95% of comparisons (1 vs. N materials)
 *   - At boundaries: Falls back to full search automatically
 *   - No additional memory allocation required
 *   - Processing order: left-to-right, top-to-bottom (enables causal dependency)
 * 
 * Performance characteristics:
 *   - Memory overhead: Zero (uses existing classification buffer)
 *   - Time overhead: Can be 30-50% faster on agricultural scenes
 *   - Accuracy: Comparable to standard method on high-contrast boundaries
 *   - May propagate errors in noisy regions (trades accuracy for speed)
 * 
 * Best for: Real-time processing, large datasets, preliminary classification
 * 
 * Configuration parameters (adjustable):
 *   - Neighbor agreement threshold: 2+ neighbors (currently left + top)
 *   - Early exit similarity: 0.85 (higher = more conservative)
 *   - Can be modified for 4-neighbor or 8-neighbor checking
 * 
 * @param ctx Processing context with:
 *   - Pre-computed reference FFTs
 *   - Thread-local FFT workspaces
 *   - Allocated classification result buffer (used for neighbor checking)
 * @return 0 on success, -1 on error
 * 
 * @note Processing must be sequential (row-by-row) to ensure neighbors are available
 *   OpenMP parallelization uses dynamic scheduling with row granularity
 * 
 * @see classify_image_fourier_cpu_optimized for accuracy-focused alternative
 */
int classify_image_fourier_cpu_light(ProcessingContext* ctx);


#endif // FOURIER_H
