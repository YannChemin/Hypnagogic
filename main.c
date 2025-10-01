#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <gdal.h>
#include <ogr_srs_api.h>
#include <cpl_conv.h>
#include <cpl_string.h>

// Import common structures
#include "common_types.h"

// Deals with auxiliary files for styling and labeling
#include "gis_export.h"
#include "material_colors.h"

// Various utils
#include "gpu_utils.h"
#include "io_utils.h"

// Spectral Libraries
#include "enhanced_reference_materials.h"
//#include "reference_materials.h"

//Fourier transform reflectance dans bibliotheque et par pixel pour comparaison
#include "fourier.h"

//TODO geosampling remember last three pixels class and compare directly with that class, if the difference is small apply directly the class without more computation

// For Windows compatibility
#ifdef _WIN32
#define strdup _strdup
#endif

// Conditional includes based on available GPU support
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef OPENCL_AVAILABLE
#include <CL/cl.h>
#endif

#define MAX_VECTOR_SIZE 4096
#define SIMILARITY_THRESHOLD 0.75
#define GPU_MEMORY_THRESHOLD (1024 * 1024 * 1024) // 1GB
#define MAX_FILENAME 512

// Function prototypes
void initialize_context(ProcessingContext* ctx);
void cleanup_context(ProcessingContext* ctx);
int load_reference_vectors_from_header(ProcessingContext* ctx);
void create_material_lookup_table(char* lookup_table[]);
int classify_image_cpu(ProcessingContext* ctx);
float calculate_pixel_similarity(float* pixel_reflectance, float* pixel_wavelengths, int pixel_bands,
                                float* ref_reflectance, float* ref_wavelengths, int ref_bands);
void interpolate_spectrum(float* src_wav, float* src_ref, int src_size, 
                         float* dst_wav, float* dst_ref, int dst_size);


// OpenCL kernel source for image classification
#ifdef OPENCL_AVAILABLE
const char* opencl_classify_kernel_source = 
"__kernel void classify_pixels(__global float* image_data,\n"
"                             __global float* image_wavelengths,\n"
"                             __global float* reference_data,\n"
"                             __global float* reference_wavelengths,\n"
"                             __global unsigned char* classification,\n"
"                             __global float* confidence,\n"
"                             int width, int height, int image_bands,\n"
"                             int num_references, __global int* ref_band_counts) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    \n"
"    if (x >= width || y >= height) return;\n"
"    \n"
"    int pixel_idx = y * width + x;\n"
"    int pixel_offset = pixel_idx * image_bands;\n"
"    \n"
"    float best_similarity = -1.0f;\n"
"    int best_material = 0;\n"
"    \n"
"    int ref_offset = 0;\n"
"    for (int ref = 0; ref < num_references; ref++) {\n"
"        int ref_bands = ref_band_counts[ref];\n"
"        \n"
"        // Calculate cosine similarity\n"
"        float dot_product = 0.0f;\n"
"        float norm_pixel = 0.0f;\n"
"        float norm_ref = 0.0f;\n"
"        \n"
"        // Simple wavelength matching (nearest neighbor)\n"
"        for (int i = 0; i < image_bands; i++) {\n"
"            float pixel_val = image_data[pixel_offset + i];\n"
"            float img_wav = image_wavelengths[i];\n"
"            \n"
"            // Find closest reference wavelength\n"
"            int closest_ref = 0;\n"
"            float min_diff = fabs(reference_wavelengths[ref_offset] - img_wav);\n"
"            for (int j = 1; j < ref_bands; j++) {\n"
"                float diff = fabs(reference_wavelengths[ref_offset + j] - img_wav);\n"
"                if (diff < min_diff) {\n"
"                    min_diff = diff;\n"
"                    closest_ref = j;\n"
"                }\n"
"            }\n"
"            \n"
"            float ref_val = reference_data[ref_offset + ref_bands + closest_ref];\n"
"            \n"
"            dot_product += pixel_val * ref_val;\n"
"            norm_pixel += pixel_val * pixel_val;\n"
"            norm_ref += ref_val * ref_val;\n"
"        }\n"
"        \n"
"        float similarity = dot_product / (sqrt(norm_pixel) * sqrt(norm_ref));\n"
"        if (similarity > best_similarity) {\n"
"            best_similarity = similarity;\n"
"            best_material = ref;\n"
"        }\n"
"        \n"
"        ref_offset += ref_bands * 2;\n"
"    }\n"
"    \n"
"    classification[pixel_idx] = (unsigned char)best_material;\n"
"    confidence[pixel_idx] = best_similarity;\n"
"}\n";
#endif




int main(int argc, char* argv[]) {
    // Initialize GDAL
    GDALAllRegister();
    
    printf("Hyperspectral Image Classification System V6\n");
    printf("============================================\n");
    
    // Parse command line arguments
    const char* input_filename = NULL;   // Changed from "hyper.tif"
    const char* output_filename = NULL;  // Changed from "classification.tif"
    bool diagnose_only = false;
    
    // Classification modes
    typedef enum {
        MODE_FOURIER_CPU,              // Legacy/baseline Fourier
        MODE_FOURIER_CPU_COHERENCE_QUALITY,   // Optimized with spatial coherence
        MODE_FOURIER_CPU_COHERENCE_FASTEST,   // Lightweight neighbor-aware
        MODE_SPATIAL_CPU                // Original spatial domain (non-Fourier)
    } ClassificationMode;
    
    ClassificationMode mode = MODE_FOURIER_CPU_COHERENCE_QUALITY; // Default to quality mode
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--diagnose") == 0 || strcmp(argv[i], "-d") == 0) {
            diagnose_only = true;
        } 
        else if (strncmp(argv[i], "--mode=", 7) == 0) {
            const char* mode_str = argv[i] + 7;
            
            if (strcmp(mode_str, "fourier_cpu") == 0) {
                mode = MODE_FOURIER_CPU;
                printf("Mode: Fourier CPU (baseline)\n");
            } 
            else if (strcmp(mode_str, "fourier_cpu_coherence_quality") == 0) {
                mode = MODE_FOURIER_CPU_COHERENCE_QUALITY;
                printf("Mode: Fourier CPU with spatial coherence (quality)\n");
            } 
            else if (strcmp(mode_str, "fourier_cpu_coherence_fastest") == 0) {
                mode = MODE_FOURIER_CPU_COHERENCE_FASTEST;
                printf("Mode: Fourier CPU with spatial coherence (fastest)\n");
            } 
            else if (strcmp(mode_str, "spatial_cpu") == 0) {
                mode = MODE_SPATIAL_CPU;
                printf("Mode: Spatial domain CPU (non-Fourier)\n");
            } 
            else {
                printf("Error: Unknown mode '%s'\n", mode_str);
                printf("Valid modes: fourier_cpu, fourier_cpu_coherence_quality, fourier_cpu_coherence_fastest, spatial_cpu\n");
                return -1;
            }
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s input.tif output.tif [options]\n", argv[0]);
            printf("\nOptions:\n");
            printf("  --diagnose, -d              : Only diagnose the input file structure\n");
            printf("  --mode=<mode>               : Classification algorithm mode\n");
            printf("  --help, -h                  : Show this help message\n");
            printf("\nAvailable modes:\n");
            printf("  fourier_cpu                 : Baseline Fourier transform method\n");
            printf("                                 - Standard accuracy\n");
            printf("                                 - Good for validation/comparison\n");
            printf("\n  fourier_cpu_coherence_quality : Spatial coherence refinement (DEFAULT)\n");
            printf("                                 - Best accuracy (~5-10%% slower)\n");
            printf("                                 - Reduces salt-and-pepper noise\n");
            printf("                                 - Preserves field boundaries\n");
            printf("                                 - Recommended for production\n");
            printf("\n  fourier_cpu_coherence_fastest : Lightweight neighbor prediction\n");
            printf("                                 - Fastest method (30-50%% speedup)\n");
            printf("                                 - Good accuracy on homogeneous scenes\n");
            printf("                                 - Ideal for large datasets\n");
            printf("\n  spatial_cpu                 : Original spatial domain method\n");
            printf("                                 - No Fourier transform\n");
            printf("                                 - Direct spectral comparison\n");
            printf("                                 - Baseline for benchmarking\n");
            printf("\nExamples:\n");
            printf("  %s field.tif result.tif\n", argv[0]);
            printf("  %s field.tif result.tif --mode=fourier_cpu_coherence_fastest\n", argv[0]);
            printf("  %s field.tif result.tif --mode=fourier_cpu\n", argv[0]);
            printf("  %s field.tif --diagnose\n", argv[0]);
            return 0;
        }
        // THIS IS THE MISSING PART - Handle .tif files
        else if (strstr(argv[i], ".tif") != NULL || strstr(argv[i], ".TIF") != NULL) {
            // Assign input and output files in order
            if (input_filename == NULL) {
                input_filename = argv[i];
            } else if (output_filename == NULL) {
                output_filename = argv[i];
            } else {
                printf("Warning: Extra .tif file argument ignored: %s\n", argv[i]);
            }
        }
        // Catch unknown arguments
        else if (strncmp(argv[i], "--", 2) != 0) {
            printf("Error: Unknown argument '%s'\n", argv[i]);
            printf("Use --help for usage information\n");
            return -1;
        }
    }
    if (input_filename == NULL) {
        printf("Error: Input file is required\n");
        printf("Usage: %s <input.tif> <output.tif> [options]\n", argv[0]);
        printf("Use --help for more information\n");
        return -1;
    }
    printf("Input file: %s\n", input_filename);
    
    if (!diagnose_only && output_filename == NULL) {
        printf("Error: Output file is required\n");
        printf("Usage: %s <input.tif> <output.tif> [options]\n", argv[0]);
        printf("Use --help for more information\n");
        return -1;
    }
    printf("Output file: %s\n", output_filename);
    
    // Diagnose mode - just analyze the file structure
    if (diagnose_only) {
        diagnose_geotiff_structure(input_filename);
        return 0;
    }
    
    ProcessingContext ctx;
    initialize_context(&ctx);
    
    // Load reference vectors from header file
    printf("\nLoading reference materials from header file...\n");
    if (load_reference_vectors_from_header(&ctx) != 0) {
        printf("Error: Failed to load reference vectors from header file\n");
        cleanup_context(&ctx);
        return -1;
    }
    
    // Load hyperspectral image with enhanced wavelength extraction
    printf("\nLoading hyperspectral image...\n");
    if (load_hyperspectral_image(&ctx, input_filename) != 0) {
        printf("Error: Failed to load hyperspectral image\n");
        cleanup_context(&ctx);
        return -1;
    }

    // Configure Fourier processing based on mode
    if (mode != MODE_SPATIAL_CPU) {
        ctx.use_fourier = 1;
        
        // Setup Fourier processing
        if (setup_fourier_processing(&ctx) != 0) {
            printf("Error: Failed to setup Fourier processing\n");
            cleanup_context(&ctx);
            return -1;
        }
        
        // Try to load Fourier cache
        char* cache_filename = generate_fourier_cache_filename(ctx.fft_size, ctx.num_reference_vectors);
        printf("\nChecking for Fourier cache: %s\n", cache_filename);
        
        if (load_fourier_cache(&ctx, cache_filename) == 0) {
            printf("✓ Loaded pre-computed Fourier library from cache!\n");
        } else {
            printf("Cache not found. Computing Fourier transforms...\n");
            precompute_fourier_references(&ctx);
            
            // Save cache
            if (save_fourier_cache(&ctx, cache_filename) == 0) {
                printf("✓ Fourier library cached for future use\n");
            }
        }
        
        free(cache_filename);
    }
        
    // Detect GPU capabilities
    int gpu_detected = detect_gpu_capabilities(&ctx);
    
    if (gpu_detected) {
        printf("GPU detected: %s\n", ctx.gpu_type == 0 ? "CUDA" : "OpenCL");
        printf("Available GPU memory: %.2f GB\n", ctx.available_gpu_memory / (1024.0 * 1024.0 * 1024.0));
        
        // Check if image fits in GPU memory
        size_t required_memory = ctx.image->data_size + 
                               ctx.num_reference_vectors * MAX_VECTOR_SIZE * 2 * sizeof(float) +
                               ctx.image->width * ctx.image->height * sizeof(float) +
                               ctx.image->width * ctx.image->height * sizeof(unsigned char);
        
        if (required_memory < ctx.available_gpu_memory * 0.8) { // Use 80% of available memory
            printf("Loading data to GPU memory...\n");
            load_data_to_gpu(&ctx);
        } else {
            printf("Image too large for GPU memory, falling back to CPU processing\n");
            ctx.use_gpu = 0;
        }
    } else {
        printf("Using CPU processing with OpenMP\n");
        printf("Number of CPU cores: %d\n", omp_get_max_threads());
    }
    
    // Create classification result
    ctx.result = create_classification_result(ctx.image->width, ctx.image->height);
    if (!ctx.result) {
        printf("Error: Failed to create classification result\n");
        cleanup_context(&ctx);
        return -1;
    }
    
    // Copy geospatial information
    memcpy(ctx.result->geotransform, ctx.image->geotransform, 6 * sizeof(double));
    if (ctx.image->projection) {
        ctx.result->projection = strdup(ctx.image->projection);
    }
    
    // Perform classification
    printf("\nClassifying hyperspectral image (%dx%dx%d)...\n", 
           ctx.image->width, ctx.image->height, ctx.image->bands);
    
    clock_t start = clock();
    int classification_result = 0;
    
    if (ctx.use_gpu && ctx.gpu_type == 0) {
        #ifdef CUDA_AVAILABLE
        classification_result = classify_image_cuda(&ctx);
        #endif
    } else if (ctx.use_gpu && ctx.gpu_type == 1) {
        #ifdef OPENCL_AVAILABLE
        classification_result = classify_image_opencl(&ctx);
        #endif
    } else {
        // CPU classification based on selected mode
        switch (mode) {
            case MODE_FOURIER_CPU:
                printf("Running baseline Fourier classification...\n");
                classification_result = classify_image_fourier_cpu(&ctx);
                break;
                
            case MODE_FOURIER_CPU_COHERENCE_QUALITY:
                printf("Running quality-optimized Fourier classification with spatial coherence...\n");
                classification_result = classify_image_fourier_cpu_optimized(&ctx);
                break;
                
            case MODE_FOURIER_CPU_COHERENCE_FASTEST:
                printf("Running fastest Fourier classification with neighbor prediction...\n");
                classification_result = classify_image_fourier_cpu_light(&ctx);
                break;
                
            case MODE_SPATIAL_CPU:
                printf("Running spatial domain classification (non-Fourier)...\n");
                classification_result = classify_image_cpu(&ctx);
                break;
                
            default:
                printf("Error: Invalid classification mode\n");
                cleanup_context(&ctx);
                return -1;
        }
    }
    
    clock_t end = clock();
    double processing_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    if (classification_result != 0) {
        printf("Error: Classification failed\n");
        cleanup_context(&ctx);
        return -1;
    }
    
    printf("Classification completed in %.2f seconds\n", processing_time);
    
    // Calculate and display statistics
    long long pixel_count = (long long)ctx.image->width * ctx.image->height;
    long long* material_counts = (long long*)calloc(ctx.num_reference_vectors, sizeof(long long));
    double total_confidence = 0.0;
    
    for (long long i = 0; i < pixel_count; i++) {
        material_counts[ctx.result->classification[i]]++;
        total_confidence += ctx.result->confidence[i];
    }
    
    printf("\nClassification Statistics:\n");
    printf("========================\n");
    printf("Total pixels: %lld\n", pixel_count);
    printf("Average confidence: %.4f\n", total_confidence / pixel_count);
    printf("\nMaterial distribution:\n");
    for (int i = 0; i < ctx.num_reference_vectors; i++) {
        double percentage = 100.0 * material_counts[i] / pixel_count;
        printf("%-12s: %8lld pixels (%.2f%%)\n", 
               ctx.reference_vectors[i].name, material_counts[i], percentage);
    }
    
    free(material_counts);
    
    // Save classification result
    printf("\nSaving classification result...\n");
    if (save_classification_result(&ctx, output_filename) != 0) {
        printf("Error: Failed to save classification result\n");
        cleanup_context(&ctx);
        return -1;
    }
    
    printf("Classification saved to: %s\n", output_filename);
    printf("Ready for use in QGIS and GRASS GIS\n\n");
    printf("# Import your classification raster\n");
    printf("r.in.gdal --o input=classification.tif output=classification\n");
    printf("# Apply the color table\n");
    printf("r.colors map=classification.1 rules=classification_grass_colors.txt\n");
    printf("# View the result\n");
    printf("d.rast map=classification\n");

    // Cleanup
    cleanup_context(&ctx);
    
    if (ctx.use_fourier) {
        cleanup_fourier_resources(&ctx);
    }

    return 0;
}


void initialize_context(ProcessingContext* ctx) {
    memset(ctx, 0, sizeof(ProcessingContext));
    ctx->num_reference_vectors = 0;
    ctx->reference_vectors = NULL;
    ctx->image = NULL;
    ctx->result = NULL;
    ctx->use_gpu = 0;
    ctx->gpu_type = -1;
    ctx->available_gpu_memory = 0;
}

void cleanup_context(ProcessingContext* ctx) {
    // Clean up reference vectors
    if (ctx->reference_vectors) {
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            if (ctx->reference_vectors[i].wavelengths) {
                free(ctx->reference_vectors[i].wavelengths);
            }
            if (ctx->reference_vectors[i].reflectance) {
                free(ctx->reference_vectors[i].reflectance);
            }
        }
        free(ctx->reference_vectors);
    }
    
    // Clean up image and result
    if (ctx->image) {
        destroy_hyperspectral_image(ctx->image);
    }
    if (ctx->result) {
        destroy_classification_result(ctx->result);
    }
    
    #ifdef CUDA_AVAILABLE
    if (ctx->use_gpu && ctx->gpu_type == 0) {
        if (ctx->d_reference_data) cudaFree(ctx->d_reference_data);
        if (ctx->d_image_data) cudaFree(ctx->d_image_data);
        if (ctx->d_wavelengths) cudaFree(ctx->d_wavelengths);
        if (ctx->d_classification) cudaFree(ctx->d_classification);
        if (ctx->d_confidence) cudaFree(ctx->d_confidence);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
    }
    #endif
    
    #ifdef OPENCL_AVAILABLE
    if (ctx->use_gpu && ctx->gpu_type == 1) {
        if (ctx->cl_reference_buffer) clReleaseMemObject(ctx->cl_reference_buffer);
        if (ctx->cl_image_buffer) clReleaseMemObject(ctx->cl_image_buffer);
        if (ctx->cl_wavelength_buffer) clReleaseMemObject(ctx->cl_wavelength_buffer);
        if (ctx->cl_classification_buffer) clReleaseMemObject(ctx->cl_classification_buffer);
        if (ctx->cl_confidence_buffer) clReleaseMemObject(ctx->cl_confidence_buffer);
        if (ctx->cl_classify_kernel) clReleaseKernel(ctx->cl_classify_kernel);
        if (ctx->cl_program) clReleaseProgram(ctx->cl_program);
        if (ctx->cl_queue) clReleaseCommandQueue(ctx->cl_queue);
        if (ctx->cl_context) clReleaseContext(ctx->cl_context);
    }
    #endif
}

int load_reference_vectors_from_header(ProcessingContext* ctx) {
    ctx->num_reference_vectors = NUM_REFERENCE_MATERIALS;
    
    if (ctx->num_reference_vectors == 0) {
        printf("Error: No reference materials found in header file\n");
        return -1;
    }
    
    printf("Found %d reference materials in header file\n", ctx->num_reference_vectors);
    
    ctx->reference_vectors = (HyperspectralVector*)malloc(
        ctx->num_reference_vectors * sizeof(HyperspectralVector));
    
    if (!ctx->reference_vectors) {
        printf("Error: Failed to allocate memory for reference vectors\n");
        return -1;
    }
    
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        const ReferenceMaterialData* mat_data = &reference_materials[i];
        HyperspectralVector* vec = &ctx->reference_vectors[i];
        
        vec->size = mat_data->size;
        vec->material = mat_data->material;
        strncpy(vec->name, mat_data->name, sizeof(vec->name) - 1);
        vec->name[sizeof(vec->name) - 1] = '\0';
        
        if (vec->size <= 0 || vec->size > MAX_VECTOR_SIZE) {
            printf("Error: Invalid size %d for material %s\n", vec->size, vec->name);
            return -1;
        }
        
        vec->wavelengths = (float*)malloc(vec->size * sizeof(float));
        vec->reflectance = (float*)malloc(vec->size * sizeof(float));
        
        if (!vec->wavelengths || !vec->reflectance) {
            printf("Error: Failed to allocate memory for vector %s\n", vec->name);
            return -1;
        }
        
        for (int j = 0; j < vec->size; j++) {
            vec->wavelengths[j] = mat_data->wavelengths[j];
            vec->reflectance[j] = mat_data->reflectance[j];
        }
        
        printf("Loaded %s: %d bands (%.1f-%.1f nm)\n", 
               vec->name, vec->size, vec->wavelengths[0], vec->wavelengths[vec->size-1]);
    }
    
    printf("Successfully loaded %d reference vectors\n", ctx->num_reference_vectors);
    return 0;
}

int classify_image_cpu(ProcessingContext* ctx) {
    printf("Classifying %d pixels using CPU with OpenMP...\n", ctx->image->width * ctx->image->height);
    
    #pragma omp parallel for
    for (int y = 0; y < ctx->image->height; y++) {
        for (int x = 0; x < ctx->image->width; x++) {
            int pixel_idx = y * ctx->image->width + x;
            
            // Extract pixel spectrum
            float* pixel_spectrum = (float*)malloc(ctx->image->bands * sizeof(float));
            for (int b = 0; b < ctx->image->bands; b++) {
                pixel_spectrum[b] = ctx->image->data[b * ctx->image->width * ctx->image->height + pixel_idx];
            }
            
            // Find best matching reference material
            float best_similarity = -1.0f;
            int best_material = 0;
            
            for (int ref = 0; ref < ctx->num_reference_vectors; ref++) {
                float similarity = calculate_pixel_similarity(
                    pixel_spectrum, ctx->image->wavelengths, ctx->image->bands,
                    ctx->reference_vectors[ref].reflectance, 
                    ctx->reference_vectors[ref].wavelengths, 
                    ctx->reference_vectors[ref].size
                );
                
                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    best_material = ref;
                }
            }
            
            // Store as uint16_t to support >256 classes
            ctx->result->classification[pixel_idx] = (uint16_t)best_material;
            ctx->result->confidence[pixel_idx] = best_similarity;
            
            free(pixel_spectrum);
        }
        
        if (y % 100 == 0) {
            printf("Processed %d/%d rows (%.1f%%)\n", y + 1, ctx->image->height, 
                   100.0 * (y + 1) / ctx->image->height);
        }
    }
    
    return 0;
}

float calculate_pixel_similarity(float* pixel_reflectance, float* pixel_wavelengths, int pixel_bands,
                                float* ref_reflectance, float* ref_wavelengths, int ref_bands) {
    // Simple cosine similarity with nearest wavelength matching
    float dot_product = 0.0f;
    float norm_pixel = 0.0f;
    float norm_ref = 0.0f;
    
    for (int i = 0; i < pixel_bands; i++) {
        float pixel_val = pixel_reflectance[i];
        float pixel_wav = pixel_wavelengths[i];
        
        // Find closest reference wavelength
        int closest_ref = 0;
        float min_diff = fabsf(ref_wavelengths[0] - pixel_wav);
        for (int j = 1; j < ref_bands; j++) {
            float diff = fabsf(ref_wavelengths[j] - pixel_wav);
            if (diff < min_diff) {
                min_diff = diff;
                closest_ref = j;
            }
        }
        
        float ref_val = ref_reflectance[closest_ref];
        
        dot_product += pixel_val * ref_val;
        norm_pixel += pixel_val * pixel_val;
        norm_ref += ref_val * ref_val;
    }
    
    if (norm_pixel == 0.0f || norm_ref == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (sqrtf(norm_pixel) * sqrtf(norm_ref));
}

void interpolate_spectrum(float* src_wav, float* src_ref, int src_size, 
                         float* dst_wav, float* dst_ref, int dst_size) {
    for (int i = 0; i < dst_size; i++) {
        float target_wav = dst_wav[i];
        
        if (target_wav <= src_wav[0]) {
            dst_ref[i] = src_ref[0];
            continue;
        }
        if (target_wav >= src_wav[src_size-1]) {
            dst_ref[i] = src_ref[src_size-1];
            continue;
        }
        
        // Find surrounding points
        int left = 0, right = src_size - 1;
        for (int j = 0; j < src_size - 1; j++) {
            if (src_wav[j] <= target_wav && src_wav[j+1] >= target_wav) {
                left = j;
                right = j + 1;
                break;
            }
        }
        
        // Linear interpolation
        if (fabsf(src_wav[right] - src_wav[left]) < 1e-6f) {
            dst_ref[i] = src_ref[left];
        } else {
            float t = (target_wav - src_wav[left]) / (src_wav[right] - src_wav[left]);
            t = fmaxf(0.0f, fminf(1.0f, t));
            dst_ref[i] = src_ref[left] + t * (src_ref[right] - src_ref[left]);
        }
    }
}
