#ifndef IO_UTILS_H
#define IO_UTILS_H

#include "common_types.h"
#include <gdal.h>
#include <stdbool.h>

// Image loading functions
int load_hyperspectral_image(ProcessingContext* ctx, const char* filename);
int extract_wavelengths_from_image(GDALDatasetH dataset, HyperspectralImage* img);

// Image saving functions
int save_classification_result(ProcessingContext* ctx, const char* output_filename);

// Diagnostic functions
void diagnose_geotiff_structure(const char* filename);

// Helper functions
HyperspectralImage* create_hyperspectral_image(int width, int height, int bands);
void destroy_hyperspectral_image(HyperspectralImage* img);
ClassificationResult* create_classification_result(int width, int height);
void destroy_classification_result(ClassificationResult* result);

#endif // IO_UTILS_H
