#include "gis_export.h"
#include "common_types.h"
#include "material_colors.h"
#include <string.h>
#include <stdlib.h>

#define MAX_FILENAME 512

int save_qgis_style_file(ProcessingContext* ctx, const char* output_filename, const GDALColorEntry* colors) {
    char qml_filename[MAX_FILENAME];
    snprintf(qml_filename, sizeof(qml_filename), "%s", output_filename);
    
    // Replace .tif with .qml
    char* ext = strrchr(qml_filename, '.');
    if (ext) {
        strcpy(ext, ".qml");
    } else {
        strcat(qml_filename, ".qml");
    }
    
    FILE* qml = fopen(qml_filename, "w");
    if (!qml) {
        printf("Warning: Could not create QML style file\n");
        return -1;
    }
    
    fprintf(qml, "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n");
    fprintf(qml, "<qgis version=\"3.0\">\n");
    fprintf(qml, "  <pipe>\n");
    fprintf(qml, "    <rasterrenderer band=\"1\" type=\"paletted\">\n");
    fprintf(qml, "      <rasterTransparency/>\n");
    fprintf(qml, "      <colorPalette>\n");
    
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        // Use colors array if within bounds, otherwise use gray
        int r = (i < NUM_MATERIAL_COLORS) ? MATERIAL_COLORS[i].c1 : 128;
        int g = (i < NUM_MATERIAL_COLORS) ? MATERIAL_COLORS[i].c2 : 128;
        int b = (i < NUM_MATERIAL_COLORS) ? MATERIAL_COLORS[i].c3 : 128;
        
        fprintf(qml, "        <paletteEntry value=\"%d\" color=\"#%02x%02x%02x\" label=\"%s\" alpha=\"255\"/>\n",
                i, r, g, b, ctx->reference_vectors[i].name);
    }
    
    fprintf(qml, "      </colorPalette>\n");
    fprintf(qml, "    </rasterrenderer>\n");
    fprintf(qml, "  </pipe>\n");
    fprintf(qml, "</qgis>\n");
    
    fclose(qml);
    printf("Created QGIS style file: %s\n", qml_filename);
    return 0;
}

int save_grass_color_rules(ProcessingContext* ctx, const char* output_filename, const GDALColorEntry* colors) {
    char rules_filename[MAX_FILENAME];
    char cats_filename[MAX_FILENAME];
    snprintf(rules_filename, sizeof(rules_filename), "%s", output_filename);
    snprintf(cats_filename, sizeof(cats_filename), "%s", output_filename);
    
    // Replace .tif with .txt for GRASS color rules
    char* ext = strrchr(rules_filename, '.');
    if (ext) {
        strcpy(ext, "_grass_colors.txt");
    } else {
        strcat(rules_filename, "_grass_colors.txt");
    }
    
    // Replace .tif with _cats.txt for GRASS categories
    ext = strrchr(cats_filename, '.');
    if (ext) {
        strcpy(ext, "_grass_cats.txt");
    } else {
        strcat(cats_filename, "_grass_cats.txt");
    }
    
    // Create color rules file
    FILE* rules = fopen(rules_filename, "w");
    if (!rules) {
        printf("Warning: Could not create GRASS color rules file\n");
        return -1;
    }
    
    // GRASS color rules format: value red:green:blue
    fprintf(rules, "# GRASS GIS color rules for hyperspectral classification\n");
    fprintf(rules, "# Generated from %d material classes\n", ctx->num_reference_vectors);
    fprintf(rules, "# Usage: r.colors map=classification rules=%s\n", rules_filename);
    fprintf(rules, "#\n");
    
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        // Use colors array if within bounds, otherwise use gray
        int r = (i < NUM_MATERIAL_COLORS) ? MATERIAL_COLORS[i].c1 : 128;
        int g = (i < NUM_MATERIAL_COLORS) ? MATERIAL_COLORS[i].c2 : 128;
        int b = (i < NUM_MATERIAL_COLORS) ? MATERIAL_COLORS[i].c3 : 128;
        
        fprintf(rules, "%d %d:%d:%d\n", i, r, g, b);
    }
    
    // Add default/null value handling
    fprintf(rules, "nv 0:0:0\n");
    fprintf(rules, "default 128:128:128\n");
    
    fclose(rules);
    printf("Created GRASS GIS color rules file: %s\n", rules_filename);
    
    // Create categories file - NO COMMENTS, r.category is strict about format
    FILE* cats = fopen(cats_filename, "w");
    if (!cats) {
        printf("Warning: Could not create GRASS categories file\n");
        return -1;
    }
    
    // GRASS categories format: value|label (no comments allowed!)
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        fprintf(cats, "%d|%s\n", i, ctx->reference_vectors[i].name);
    }
    
    fclose(cats);
    printf("Created GRASS GIS categories file: %s\n", cats_filename);
    
    // Print usage instructions
    print_grass_usage_instructions(output_filename, rules_filename, cats_filename);
    
    return 0;
}

void print_grass_usage_instructions(const char* output_filename, const char* rules_filename, const char* cats_filename) {
    printf("\nGRASS GIS Usage:\n");
    printf("================\n");
    printf("# Import raster\n");
    printf("r.import --o input=%s output=classification\n\n", output_filename);
    printf("# Apply color table\n");
    printf("r.colors map=classification.1 rules=%s\n\n", rules_filename);
    printf("# Apply category labels\n");
    printf("r.category map=classification.1 rules=%s separator=pipe\n\n", cats_filename);
    printf("# Verify categories\n");
    printf("r.category map=classification.1\n\n");
    printf("# Display map\n");
    printf("d.rast map=classification.1\n");
    printf("d.legend raster=classification.1 at=5,50,7,10\n");
}
