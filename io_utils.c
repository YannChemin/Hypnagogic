#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <gdal.h>
#include <ogr_srs_api.h>
#include <cpl_conv.h>
#include <cpl_string.h>

#include "io_utils.h"
#include "common_types.h"
#include "gis_export.h"
#include "material_colors.h"

// For Windows/Unix mkdir compatibility
#ifdef _WIN32
#include <direct.h>
#define strdup _strdup
#else
#include <sys/stat.h>
#endif

#define MAX_VECTOR_SIZE 4096

void diagnose_geotiff_structure(const char* filename) {
    printf("\n=== GeoTIFF Structure Diagnosis ===\n");
    printf("File: %s\n", filename);
    
    GDALDatasetH dataset = GDALOpen(filename, GA_ReadOnly);
    if (!dataset) {
        printf("Error: Cannot open file for diagnosis\n");
        return;
    }
    
    // Basic info
    printf("Driver: %s\n", GDALGetDriverShortName(GDALGetDatasetDriver(dataset)));
    printf("Size: %d x %d x %d\n", 
           GDALGetRasterXSize(dataset), 
           GDALGetRasterYSize(dataset),
           GDALGetRasterCount(dataset));
    
    // Dataset metadata
    printf("\n--- Dataset Metadata ---\n");
    char** metadata = GDALGetMetadata(dataset, NULL);
    if (metadata) {
        for (int i = 0; metadata[i] != NULL; i++) {
            printf("  %s\n", metadata[i]);
        }
    } else {
        printf("  No dataset metadata found\n");
    }
    
    // Check for ENVI metadata domain
    printf("\n--- ENVI Metadata Domain ---\n");
    char** envi_metadata = GDALGetMetadata(dataset, "ENVI");
    if (envi_metadata) {
        for (int i = 0; envi_metadata[i] != NULL; i++) {
            printf("  ENVI: %s\n", envi_metadata[i]);
        }
    } else {
        printf("  No ENVI metadata found\n");
    }
    
    // Check each band
    printf("\n--- Band Information ---\n");
    int bands = GDALGetRasterCount(dataset);
    for (int i = 0; i < bands && i < 5; i++) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, i + 1);
        printf("Band %d:\n", i + 1);
        printf("  Description: '%s'\n", GDALGetDescription(band));
        printf("  Data Type: %s\n", GDALGetDataTypeName(GDALGetRasterDataType(band)));
        
        char** band_metadata = GDALGetMetadata(band, NULL);
        if (band_metadata) {
            for (int j = 0; band_metadata[j] != NULL; j++) {
                printf("    %s\n", band_metadata[j]);
            }
        } else {
            printf("    No band metadata\n");
        }
        
        const char* wl_keys[] = {"wavelength", "WAVELENGTH", "center_wavelength", "CENTER_WAVELENGTH", NULL};
        for (int j = 0; wl_keys[j] != NULL; j++) {
            const char* wl_val = GDALGetMetadataItem(band, wl_keys[j], NULL);
            if (wl_val) {
                printf("    %s = %s\n", wl_keys[j], wl_val);
            }
        }
        printf("\n");
    }
    
    if (bands > 5) {
        printf("... (showing first 5 of %d bands)\n", bands);
    }
    
    GDALClose(dataset);
    printf("=== End Diagnosis ===\n\n");
}

int extract_wavelengths_from_image(GDALDatasetH dataset, HyperspectralImage* img) {
    int bands = img->bands;
    printf("Attempting to extract wavelengths from %d bands...\n", bands);
    
    // Method 1: Dataset metadata
    char** metadata = GDALGetMetadata(dataset, NULL);
    if (metadata) {
        printf("Dataset metadata found, searching for wavelength information...\n");
        for (int i = 0; metadata[i] != NULL; i++) {
            if (strstr(metadata[i], "wavelength") || strstr(metadata[i], "WAVELENGTH")) {
                printf("Found wavelength metadata: %s\n", metadata[i]);
            }
        }
    }
    
    // Method 2: Per-band metadata
    int wavelengths_found = 0;
    for (int i = 0; i < bands; i++) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, i + 1);
        
        const char* wavelength_keys[] = {
            "wavelength", "WAVELENGTH", "center_wavelength", "CENTER_WAVELENGTH",
            "band_wavelength", "BAND_WAVELENGTH", NULL
        };
        
        float wavelength_val = 0.0f;
        int found_this_band = 0;
        
        for (int j = 0; wavelength_keys[j] != NULL; j++) {
            const char* wl_str = GDALGetMetadataItem(band, wavelength_keys[j], NULL);
            if (wl_str && strlen(wl_str) > 0) {
                wavelength_val = atof(wl_str);
                if (wavelength_val > 0.0f && wavelength_val < 50000.0f) {
                    img->wavelengths[i] = wavelength_val;
                    found_this_band = 1;
                    wavelengths_found = 1;
                    printf("Band %d: %.1f nm (from %s)\n", i+1, wavelength_val, wavelength_keys[j]);
                    break;
                }
            }
        }
        
        // Try band description
        if (!found_this_band) {
            const char* desc = GDALGetDescription(band);
            if (desc && strlen(desc) > 0) {
                char* endptr;
                float desc_val = strtof(desc, &endptr);
                if (desc_val > 0.0f && desc_val < 50000.0f && endptr != desc) {
                    img->wavelengths[i] = desc_val;
                    found_this_band = 1;
                    wavelengths_found = 1;
                    printf("Band %d: %.1f nm (from description)\n", i+1, desc_val);
                }
            }
        }
    }
    
    // Method 3: ENVI header file
    if (!wavelengths_found) {
        printf("Checking for ENVI header file...\n");
        const char* filename = GDALGetDescription(dataset);
        if (filename) {
            char hdr_filename[512];
            snprintf(hdr_filename, sizeof(hdr_filename), "%s", filename);
            
            char* dot = strrchr(hdr_filename, '.');
            if (dot) {
                strcpy(dot, ".hdr");
            } else {
                strcat(hdr_filename, ".hdr");
            }
            
            printf("Looking for: %s\n", hdr_filename);
            FILE* hdr_file = fopen(hdr_filename, "r");
            if (hdr_file) {
                printf("Found ENVI header, parsing...\n");
                char line[1024];
                int in_wavelength_section = 0;
                int wl_index = 0;
                
                while (fgets(line, sizeof(line), hdr_file) && wl_index < bands) {
                    line[strcspn(line, "\r\n")] = 0;
                    
                    if (strstr(line, "wavelength") && strstr(line, "=")) {
                        in_wavelength_section = 1;
                        continue;
                    }
                    
                    if (in_wavelength_section) {
                        if (strchr(line, '}')) break;
                        
                        char* token = strtok(line, ",");
                        while (token && wl_index < bands) {
                            float wl = atof(token);
                            if (wl > 0.0f && wl < 50000.0f) {
                                img->wavelengths[wl_index++] = wl;
                                wavelengths_found = 1;
                            }
                            token = strtok(NULL, ",");
                        }
                    }
                }
                fclose(hdr_file);
                if (wavelengths_found) {
                    printf("Extracted %d wavelengths from ENVI header\n", wl_index);
                }
            }
        }
    }
    
    // Method 4: Intelligent defaults
    if (!wavelengths_found) {
        printf("Using sensor-specific defaults based on band count...\n");
        float start_wl, end_wl;
        
        if (bands == 23) {
            start_wl = 400.0f; end_wl = 850.0f;
            printf("Applied RapidEye-like range\n");
        } else if (bands >= 200 && bands <= 250) {
            start_wl = 400.0f; end_wl = 2500.0f;
            printf("Applied AVIRIS-like range\n");
        } else if (bands >= 100 && bands < 200) {
            start_wl = 400.0f; end_wl = 2400.0f;
            printf("Applied Hyperion-like range\n");
        } else {
            start_wl = 400.0f; end_wl = 1000.0f;
            printf("Applied generic VIS-NIR range\n");
        }
        
        for (int i = 0; i < bands; i++) {
            img->wavelengths[i] = start_wl + i * (end_wl - start_wl) / (bands - 1);
        }
        wavelengths_found = 1;
    }
    
    // Validate wavelengths
    if (wavelengths_found) {
        printf("Final wavelength range: %.1f - %.1f nm\n", 
               img->wavelengths[0], img->wavelengths[bands-1]);
        return 0;
    }
    
    printf("Error: Could not determine wavelengths\n");
    return -1;
}

int load_hyperspectral_image(ProcessingContext* ctx, const char* filename) {
    GDALDatasetH dataset = GDALOpen(filename, GA_ReadOnly);
    if (!dataset) {
        printf("Error: Cannot open file: %s\n", filename);
        return -1;
    }
    
    int width = GDALGetRasterXSize(dataset);
    int height = GDALGetRasterYSize(dataset);
    int bands = GDALGetRasterCount(dataset);
    
    printf("Image dimensions: %dx%dx%d\n", width, height, bands);
    
    if (bands > MAX_VECTOR_SIZE) {
        printf("Error: Too many bands (%d), max is %d\n", bands, MAX_VECTOR_SIZE);
        GDALClose(dataset);
        return -1;
    }
    
    ctx->image = create_hyperspectral_image(width, height, bands);
    if (!ctx->image) {
        GDALClose(dataset);
        return -1;
    }
    
    // Geotransform
    if (GDALGetGeoTransform(dataset, ctx->image->geotransform) != CE_None) {
        printf("Warning: No geotransform, using default\n");
        ctx->image->geotransform[0] = 0.0;
        ctx->image->geotransform[1] = 1.0;
        ctx->image->geotransform[2] = 0.0;
        ctx->image->geotransform[3] = 0.0;
        ctx->image->geotransform[4] = 0.0;
        ctx->image->geotransform[5] = -1.0;
    }
    
    // Projection
    const char* projection = GDALGetProjectionRef(dataset);
    if (projection && strlen(projection) > 0) {
        ctx->image->projection = strdup(projection);
        printf("Projection: %.100s...\n", projection);
    } else {
        ctx->image->projection = NULL;
    }
    
    // Extract wavelengths
    if (extract_wavelengths_from_image(dataset, ctx->image) != 0) {
        GDALClose(dataset);
        return -1;
    }
    
    // Read band data
    printf("Reading image data...\n");
    for (int band = 0; band < bands; band++) {
        GDALRasterBandH raster_band = GDALGetRasterBand(dataset, band + 1);
        
        CPLErr err = GDALRasterIO(raster_band, GF_Read, 0, 0, width, height,
                                 ctx->image->data + (band * width * height),
                                 width, height, GDT_Float32, 0, 0);
        
        if (err != CE_None) {
            printf("Error: Failed to read band %d\n", band + 1);
            GDALClose(dataset);
            return -1;
        }
        
        if (band % 10 == 0 || band == bands - 1) {
            printf("Read band %d/%d (%.1f nm)\n", band + 1, bands, ctx->image->wavelengths[band]);
        }
    }
    
    ctx->image->data_size = (size_t)width * height * bands * sizeof(float);
    printf("Image loaded (%.2f MB)\n", ctx->image->data_size / (1024.0 * 1024.0));
    
    GDALClose(dataset);
    return 0;
}

int save_classification_result(ProcessingContext* ctx, const char* output_filename) {
    GDALDriverH driver = GDALGetDriverByName("GTiff");
    if (!driver) {
        printf("Error: Cannot get GTiff driver\n");
        return -1;
    }
    
    char** options = NULL;
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "TILED", "YES");
    
    GDALDatasetH output_dataset = GDALCreate(driver, output_filename, 
                                           ctx->result->width, ctx->result->height, 
                                           2, GDT_UInt16, options);
    CSLDestroy(options);
    
    if (!output_dataset) {
        printf("Error: Cannot create output dataset\n");
        return -1;
    }
    
    // Geotransform and projection
    GDALSetGeoTransform(output_dataset, ctx->result->geotransform);
    if (ctx->result->projection) {
        GDALSetProjection(output_dataset, ctx->result->projection);
    }
    
    // Classification band
    GDALRasterBandH class_band = GDALGetRasterBand(output_dataset, 1);
    GDALSetDescription(class_band, "Material Classification");
    
    CPLErr err = GDALRasterIO(class_band, GF_Write, 0, 0, 
                             ctx->result->width, ctx->result->height,
                             ctx->result->classification,
                             ctx->result->width, ctx->result->height,
                             GDT_UInt16, 0, 0);
    
    if (err != CE_None) {
        printf("Error: Failed to write classification band\n");
        GDALClose(output_dataset);
        return -1;
    }
    
    // Color table
    GDALColorTableH color_table = GDALCreateColorTable(GPI_RGB);
    for (int i = 0; i < ctx->num_reference_vectors && i < NUM_MATERIAL_COLORS; i++) {
        GDALSetColorEntry(color_table, i, &MATERIAL_COLORS[i]);
    }
    GDALSetRasterColorTable(class_band, color_table);
    GDALDestroyColorTable(color_table);
    
    // Confidence band
    GDALRasterBandH conf_band = GDALGetRasterBand(output_dataset, 2);
    GDALSetDescription(conf_band, "Classification Confidence");
    
    uint16_t* scaled_confidence = (uint16_t*)malloc(ctx->result->width * ctx->result->height * sizeof(uint16_t));
    for (long long i = 0; i < (long long)ctx->result->width * ctx->result->height; i++) {
        float conf = ctx->result->confidence[i];
        if (conf < 0.0f) conf = 0.0f;
        if (conf > 1.0f) conf = 1.0f;
        scaled_confidence[i] = (uint16_t)(conf * 65535.0f);
    }
    
    err = GDALRasterIO(conf_band, GF_Write, 0, 0, 
                      ctx->result->width, ctx->result->height,
                      scaled_confidence,
                      ctx->result->width, ctx->result->height,
                      GDT_UInt16, 0, 0);
    free(scaled_confidence);
    
    if (err != CE_None) {
        printf("Error: Failed to write confidence band\n");
        GDALClose(output_dataset);
        return -1;
    }
    
    // Metadata
    char** metadata = NULL;
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        char key[64], value[128];
        snprintf(key, sizeof(key), "CLASS_%d_NAME", i);
        snprintf(value, sizeof(value), "%s", ctx->reference_vectors[i].name);
        metadata = CSLSetNameValue(metadata, key, value);
    }
    
    char stats[256];
    snprintf(stats, sizeof(stats), "%d", ctx->num_reference_vectors);
    metadata = CSLSetNameValue(metadata, "NUM_CLASSES", stats);
    metadata = CSLSetNameValue(metadata, "MAX_CLASSES_SUPPORTED", "65535");
    metadata = CSLSetNameValue(metadata, "CLASSIFICATION_TYPE", "Hyperspectral_Material_Classification");
    
    GDALSetMetadata(class_band, metadata, NULL);
    CSLDestroy(metadata);
    
    // Raster Attribute Table
    GDALRasterAttributeTableH rat = GDALCreateRasterAttributeTable();
    GDALRATCreateColumn(rat, "VALUE", GFT_Integer, GFU_MinMax);
    GDALRATCreateColumn(rat, "NAME", GFT_String, GFU_Name);
    GDALRATCreateColumn(rat, "RED", GFT_Integer, GFU_Red);
    GDALRATCreateColumn(rat, "GREEN", GFT_Integer, GFU_Green);
    GDALRATCreateColumn(rat, "BLUE", GFT_Integer, GFU_Blue);
    
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        GDALRATSetValueAsInt(rat, i, 0, i);
        GDALRATSetValueAsString(rat, i, 1, ctx->reference_vectors[i].name);
        if (i < NUM_MATERIAL_COLORS) {
            GDALRATSetValueAsInt(rat, i, 2, MATERIAL_COLORS[i].c1);
            GDALRATSetValueAsInt(rat, i, 3, MATERIAL_COLORS[i].c2);
            GDALRATSetValueAsInt(rat, i, 4, MATERIAL_COLORS[i].c3);
        }
    }
    
    GDALSetDefaultRAT(class_band, rat);
    GDALDestroyRasterAttributeTable(rat);
    
    // Export styling files
    save_qgis_style_file(ctx, output_filename, MATERIAL_COLORS);
    save_grass_color_rules(ctx, output_filename, MATERIAL_COLORS);
    
    GDALClose(output_dataset);
    
    printf("Classification saved with 16-bit support (up to 65,535 classes)\n");
    return 0;
}

HyperspectralImage* create_hyperspectral_image(int width, int height, int bands) {
    HyperspectralImage* img = (HyperspectralImage*)malloc(sizeof(HyperspectralImage));
    if (!img) return NULL;
    
    img->width = width;
    img->height = height;
    img->bands = bands;
    img->projection = NULL;
    
    size_t data_size = (size_t)width * height * bands * sizeof(float);
    img->data = (float*)malloc(data_size);
    img->wavelengths = (float*)malloc(bands * sizeof(float));
    
    if (!img->data || !img->wavelengths) {
        destroy_hyperspectral_image(img);
        return NULL;
    }
    
    img->data_size = data_size;
    return img;
}

void destroy_hyperspectral_image(HyperspectralImage* img) {
    if (img) {
        if (img->data) free(img->data);
        if (img->wavelengths) free(img->wavelengths);
        if (img->projection) free(img->projection);
        free(img);
    }
}

ClassificationResult* create_classification_result(int width, int height) {
    ClassificationResult* result = (ClassificationResult*)malloc(sizeof(ClassificationResult));
    if (!result) return NULL;
    
    result->width = width;
    result->height = height;
    result->projection = NULL;
    
    size_t pixel_count = (size_t)width * height;
    result->classification = (uint16_t*)malloc(pixel_count * sizeof(uint16_t));
    result->confidence = (float*)malloc(pixel_count * sizeof(float));
    
    if (!result->classification || !result->confidence) {
        destroy_classification_result(result);
        return NULL;
    }
    
    return result;
}

void destroy_classification_result(ClassificationResult* result) {
    if (result) {
        if (result->classification) free(result->classification);
        if (result->confidence) free(result->confidence);
        if (result->projection) free(result->projection);
        free(result);
    }
}
