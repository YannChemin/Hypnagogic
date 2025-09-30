#ifndef GIS_EXPORT_H
#define GIS_EXPORT_H

#include <gdal.h>
#include <stdio.h>

// Forward declarations for types used from main.c
typedef struct ProcessingContext ProcessingContext;
typedef struct HyperspectralVector HyperspectralVector;

/**
 * Generate QGIS style file (.qml) for classification visualization
 * - Creates paletted/unique values renderer configuration
 * - Maps each material class to a distinct color
 * - Includes material names as labels
 * - Enables automatic styling when loading in QGIS
 * @param ctx Processing context with reference materials
 * @param output_filename Base filename (adds .qml extension)
 * @param colors Array of GDAL color entries for each class
 * @return 0 on success, -1 on error
 */
int save_qgis_style_file(ProcessingContext* ctx, const char* output_filename, const GDALColorEntry* colors);

/**
 * Generate GRASS GIS color rules file
 * - Creates color table in GRASS format (value R:G:B)
 * - Maps material IDs to RGB colors
 * - Used with r.colors to apply colors to raster
 * @param ctx Processing context with reference materials
 * @param output_filename Path to color rules file
 * @param colors Array of GDAL color entries for each class
 * @return 0 on success, -1 on error
 */
int save_grass_color_rules(ProcessingContext* ctx, const char* output_filename, const GDALColorEntry* colors);

/**
 * Print GRASS GIS usage instructions to console
 * - Shows commands for importing classification raster
 * - Explains how to apply color table and labels
 * - Includes visualization commands
 * @param output_filename Classification raster filename
 * @param rules_filename Color rules filename
 * @param cats_filename Category labels filename
 */
void print_grass_usage_instructions(const char* output_filename, const char* rules_filename, const char* cats_filename);

#endif // GIS_EXPORT_H
