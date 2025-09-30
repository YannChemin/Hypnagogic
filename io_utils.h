#ifndef IO_UTILS_H
#define IO_UTILS_H

#include "common_types.h"
#include <gdal.h>
#include <stdbool.h>

/**
 * Load hyperspectral image from GeoTIFF file
 * - Opens multi-band GeoTIFF using GDAL
 * - Extracts wavelength metadata from band descriptions
 * - Reads geospatial transform and projection information
 * - Loads all band data into memory for processing
 * @param ctx Processing context to populate with image data
 * @param filename Path to input GeoTIFF file
 * @return 0 on success, -1 on error
 */
int load_hyperspectral_image(ProcessingContext* ctx, const char* filename);

/**
 * Extract wavelength information from GeoTIFF bands
 * - Parses band descriptions for wavelength metadata
 * - Supports multiple wavelength metadata formats
 * - Falls back to band index if no wavelength found
 * - Populates image wavelength array
 * @param dataset Open GDAL dataset handle
 * @param img Hyperspectral image structure to populate
 * @return 0 on success, -1 on error
 */
int extract_wavelengths_from_image(GDALDatasetH dataset, HyperspectralImage* img);

/**
 * Save classification result as GeoTIFF with styling
 * - Writes 16-bit classification raster with geospatial metadata
 * - Copies projection and geotransform from input image
 * - Generates QGIS style file (.qml) for visualization
 * - Creates GRASS GIS color rules and category labels
 * @param ctx Processing context with classification result
 * @param output_filename Path to output GeoTIFF file
 * @return 0 on success, -1 on error
 */
int save_classification_result(ProcessingContext* ctx, const char* output_filename);

/**
 * Diagnose and print GeoTIFF file structure
 * - Reports dimensions, band count, data type
 * - Displays geospatial metadata (projection, transform)
 * - Shows wavelength information if available
 * - Helps debug file format issues
 * @param filename Path to GeoTIFF file to analyze
 */
void diagnose_geotiff_structure(const char* filename);

/**
 * Create and allocate hyperspectral image structure
 * - Allocates image data array (bands * width * height)
 * - Allocates wavelength array
 * - Initializes geotransform and projection
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param bands Number of spectral bands
 * @return Allocated image structure or NULL on error
 */
HyperspectralImage* create_hyperspectral_image(int width, int height, int bands);

/**
 * Free hyperspectral image structure and all allocated memory
 * - Releases image data array
 * - Frees wavelength array and projection string
 * - Safe to call with NULL pointer
 * @param img Image structure to destroy
 */
void destroy_hyperspectral_image(HyperspectralImage* img);

/**
 * Create and allocate classification result structure
 * - Allocates classification array (16-bit for >256 classes)
 * - Allocates confidence score array (float)
 * - Initializes geotransform and projection
 * @param width Result width in pixels
 * @param height Result height in pixels
 * @return Allocated result structure or NULL on error
 */
ClassificationResult* create_classification_result(int width, int height);

/**
 * Free classification result structure and all allocated memory
 * - Releases classification and confidence arrays
 * - Frees projection string
 * - Safe to call with NULL pointer
 * @param result Result structure to destroy
 */
void destroy_classification_result(ClassificationResult* result);

#endif // IO_UTILS_H
