#ifndef GIS_EXPORT_H
#define GIS_EXPORT_H

#include <gdal.h>
#include <stdio.h>

// Forward declarations for types used from main.c
typedef struct ProcessingContext ProcessingContext;
typedef struct HyperspectralVector HyperspectralVector;

// GIS export function prototypes
int save_qgis_style_file(ProcessingContext* ctx, const char* output_filename, const GDALColorEntry* colors);
int save_grass_color_rules(ProcessingContext* ctx, const char* output_filename, const GDALColorEntry* colors);
void print_grass_usage_instructions(const char* output_filename, const char* rules_filename, const char* cats_filename);

#endif // GIS_EXPORT_H
