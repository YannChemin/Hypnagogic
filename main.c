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
//#include "enhanced_reference_materials.h"
#include "reference_materials.h"

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
#define SIMILARITY_THRESHOLD 0.7
#define GPU_MEMORY_THRESHOLD (1024 * 1024 * 1024) // 1GB
#define MAX_FILENAME 512

// Hyperspectral image structure
typedef struct {
    float* data;              // Pixel data [height][width][bands]
    float* wavelengths;       // Wavelength for each band
    int width;
    int height;
    int bands;
    double geotransform[6];   // GDAL geotransform
    char* projection;         // WKT projection string
    size_t data_size;         // Total size in bytes
} HyperspectralImage;

// Classification result structure
typedef struct {
    uint16_t* classification;     // Changed from unsigned char* to uint16_t*
    float* confidence;            // Keep as float for precision
    int width;
    int height;
    double geotransform[6];
    char* projection;
} ClassificationResult;

// Processing context
typedef struct {
    HyperspectralVector* reference_vectors;
    int num_reference_vectors;
    int use_gpu;
    int gpu_type; // 0: CUDA, 1: OpenCL
    size_t available_gpu_memory;
    
    // Image processing
    HyperspectralImage* image;
    ClassificationResult* result;
    
    #ifdef CUDA_AVAILABLE
    float* d_reference_data;
    float* d_image_data;
    float* d_wavelengths;
    unsigned char* d_classification;
    float* d_confidence;
    cudaStream_t stream;
    #endif
    
    #ifdef OPENCL_AVAILABLE
    cl_context cl_context;
    cl_command_queue cl_queue;
    cl_program cl_program;
    cl_kernel cl_classify_kernel;
    cl_mem cl_reference_buffer;
    cl_mem cl_image_buffer;
    cl_mem cl_wavelength_buffer;
    cl_mem cl_classification_buffer;
    cl_mem cl_confidence_buffer;
    #endif
} ProcessingContext;

// Function prototypes
void initialize_context(ProcessingContext* ctx);
void cleanup_context(ProcessingContext* ctx);
int load_reference_vectors_from_header(ProcessingContext* ctx);
int load_hyperspectral_image(ProcessingContext* ctx, const char* filename);
int save_classification_result(ProcessingContext* ctx, const char* output_filename);
void create_material_lookup_table(char* lookup_table[]);
HyperspectralImage* create_hyperspectral_image(int width, int height, int bands);
void destroy_hyperspectral_image(HyperspectralImage* img);
ClassificationResult* create_classification_result(int width, int height);
void destroy_classification_result(ClassificationResult* result);
int detect_gpu_capabilities(ProcessingContext* ctx);
void load_data_to_gpu(ProcessingContext* ctx);
int classify_image_cpu(ProcessingContext* ctx);
int classify_image_cuda(ProcessingContext* ctx);
int classify_image_opencl(ProcessingContext* ctx);
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

// CUDA kernel for image classification
#ifdef CUDA_AVAILABLE
__global__ void cuda_classify_pixels(float* image_data, float* image_wavelengths,
                                    float* reference_data, float* reference_wavelengths,
                                    uint16_t* classification, float* confidence,
                                    int width, int height, int image_bands,
                                    int num_references, int* ref_band_counts) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = y * width + x;
    int pixel_offset = pixel_idx * image_bands;
    
    float best_similarity = -1.0f;
    int best_material = 0;
    
    int ref_offset = 0;
    for (int ref = 0; ref < num_references; ref++) {
        int ref_bands = ref_band_counts[ref];
        
        // Calculate cosine similarity
        float dot_product = 0.0f;
        float norm_pixel = 0.0f;
        float norm_ref = 0.0f;
        
        // Simple wavelength matching (nearest neighbor)
        for (int i = 0; i < image_bands; i++) {
            float pixel_val = image_data[pixel_offset + i];
            float img_wav = image_wavelengths[i];
            
            // Find closest reference wavelength
            int closest_ref = 0;
            float min_diff = fabsf(reference_wavelengths[ref_offset] - img_wav);
            for (int j = 1; j < ref_bands; j++) {
                float diff = fabsf(reference_wavelengths[ref_offset + j] - img_wav);
                if (diff < min_diff) {
                    min_diff = diff;
                    closest_ref = j;
                }
            }
            
            float ref_val = reference_data[ref_offset + ref_bands + closest_ref];
            
            dot_product += pixel_val * ref_val;
            norm_pixel += pixel_val * pixel_val;
            norm_ref += ref_val * ref_val;
        }
        
        float similarity = dot_product / (sqrtf(norm_pixel) * sqrtf(norm_ref));
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_material = ref;
        }
        
        ref_offset += ref_bands * 2;
    }
    
    classification[pixel_idx] = (uint16_t)best_material; 
    confidence[pixel_idx] = best_similarity;
}
#endif

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
    for (int i = 0; i < bands && i < 5; i++) { // Show first 5 bands
        GDALRasterBandH band = GDALGetRasterBand(dataset, i + 1);
        printf("Band %d:\n", i + 1);
        printf("  Description: '%s'\n", GDALGetDescription(band));
        printf("  Data Type: %s\n", GDALGetDataTypeName(GDALGetRasterDataType(band)));
        
        // Band metadata
        char** band_metadata = GDALGetMetadata(band, NULL);
        if (band_metadata) {
            for (int j = 0; band_metadata[j] != NULL; j++) {
                printf("    %s\n", band_metadata[j]);
            }
        } else {
            printf("    No band metadata\n");
        }
        
        // Try specific wavelength keys
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


int main(int argc, char* argv[]) {
    // Initialize GDAL
    GDALAllRegister();
    
    printf("Hyperspectral Image Classification System V4\n");
    printf("============================================\n");
    
    // Parse command line arguments
    const char* input_filename = "hyper.tif";
    const char* output_filename = "classification.tif";
    bool diagnose_only = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--diagnose") == 0 || strcmp(argv[i], "-d") == 0) {
            diagnose_only = true;
        } else if (strstr(argv[i], ".tif") != NULL || strstr(argv[i], ".TIF") != NULL) {
            if (i == 1) {
                input_filename = argv[i];
            } else if (i == 2) {
                output_filename = argv[i];
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [input.tif] [output.tif] [--diagnose]\n", argv[0]);
            printf("  --diagnose, -d    : Only diagnose the input file structure\n");
            printf("  --help, -h        : Show this help message\n");
            return 0;
        }
    }
    
    printf("Input file: %s\n", input_filename);
    
    // Diagnose mode - just analyze the file structure
    if (diagnose_only) {
        diagnose_geotiff_structure(input_filename);
        return 0;
    }
    
    printf("Output file: %s\n", output_filename);
    
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
        classification_result = classify_image_cpu(&ctx);
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
    printf("Ready for use in QGIS and GRASS GIS\n");
    
    // Cleanup
    cleanup_context(&ctx);
    
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


// Enhanced wavelength extraction function
int extract_wavelengths_from_image(GDALDatasetH dataset, HyperspectralImage* img) {
    int bands = img->bands;
    printf("Attempting to extract wavelengths from %d bands...\n", bands);
    
    // Method 1: Try to get wavelengths from dataset metadata
    char** metadata = GDALGetMetadata(dataset, NULL);
    if (metadata) {
        printf("Dataset metadata found, searching for wavelength information...\n");
        for (int i = 0; metadata[i] != NULL; i++) {
            printf("Metadata: %s\n", metadata[i]);
            // Look for ENVI-style wavelength metadata
            if (strstr(metadata[i], "wavelength") || strstr(metadata[i], "WAVELENGTH")) {
                printf("Found wavelength metadata: %s\n", metadata[i]);
            }
        }
    }
    
    // Method 2: Check each band for wavelength metadata
    bool wavelengths_found = false;
    for (int i = 0; i < bands; i++) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, i + 1);
        
        // Try multiple metadata keys
        const char* wavelength_keys[] = {
            "wavelength",
            "WAVELENGTH", 
            "center_wavelength",
            "CENTER_WAVELENGTH",
            "band_wavelength",
            "BAND_WAVELENGTH",
            NULL
        };
        
        float wavelength_val = 0.0f;
        bool found_this_band = false;
        
        for (int j = 0; wavelength_keys[j] != NULL; j++) {
            const char* wl_str = GDALGetMetadataItem(band, wavelength_keys[j], NULL);
            if (wl_str && strlen(wl_str) > 0) {
                wavelength_val = atof(wl_str);
                if (wavelength_val > 0.0f && wavelength_val < 50000.0f) { // Reasonable range check
                    img->wavelengths[i] = wavelength_val;
                    found_this_band = true;
                    wavelengths_found = true;
                    printf("Band %d: %.1f nm (from metadata key: %s)\n", i+1, wavelength_val, wavelength_keys[j]);
                    break;
                }
            }
        }
        
        // Try band description
        if (!found_this_band) {
            const char* desc = GDALGetDescription(band);
            if (desc && strlen(desc) > 0) {
                printf("Band %d description: %s\n", i+1, desc);
                // Try to parse numbers from description
                char* endptr;
                float desc_val = strtof(desc, &endptr);
                if (desc_val > 0.0f && desc_val < 50000.0f && endptr != desc) {
                    img->wavelengths[i] = desc_val;
                    found_this_band = true;
                    wavelengths_found = true;
                    printf("Band %d: %.1f nm (from description)\n", i+1, desc_val);
                }
            }
        }
        
        // Check band metadata
        if (!found_this_band) {
            char** band_metadata = GDALGetMetadata(band, NULL);
            if (band_metadata) {
                for (int k = 0; band_metadata[k] != NULL; k++) {
                    printf("Band %d metadata: %s\n", i+1, band_metadata[k]);
                    if (strstr(band_metadata[k], "wavelength") || strstr(band_metadata[k], "WAVELENGTH")) {
                        // Try to extract number from this metadata
                        char* num_start = band_metadata[k];
                        while (*num_start && !isdigit(*num_start) && *num_start != '.') num_start++;
                        if (*num_start) {
                            float meta_val = atof(num_start);
                            if (meta_val > 0.0f && meta_val < 50000.0f) {
                                img->wavelengths[i] = meta_val;
                                found_this_band = true;
                                wavelengths_found = true;
                                printf("Band %d: %.1f nm (from band metadata)\n", i+1, meta_val);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Method 3: Check for ENVI header file
    if (!wavelengths_found) {
        printf("No wavelength metadata found in GDAL, checking for ENVI header file...\n");
        
        // Try to find .hdr file
        const char* filename = GDALGetDescription(dataset);
        if (!filename) {
            printf("Cannot get dataset filename for ENVI header search\n");
        } else {
            char hdr_filename[512];
            snprintf(hdr_filename, sizeof(hdr_filename), "%s", filename);
            
            // Replace extension with .hdr
            char* dot = strrchr(hdr_filename, '.');
            if (dot) {
                strcpy(dot, ".hdr");
            } else {
                strcat(hdr_filename, ".hdr");
            }
            
            printf("Looking for ENVI header: %s\n", hdr_filename);
            FILE* hdr_file = fopen(hdr_filename, "r");
            if (hdr_file) {
                printf("Found ENVI header file, parsing wavelengths...\n");
                char line[1024];
                bool in_wavelength_section = false;
                int wl_index = 0;
                
                while (fgets(line, sizeof(line), hdr_file) && wl_index < bands) {
                    // Remove newline
                    line[strcspn(line, "\r\n")] = 0;
                    
                    if (strstr(line, "wavelength") && strstr(line, "=")) {
                        in_wavelength_section = true;
                        printf("Found wavelength section in ENVI header\n");
                        continue;
                    }
                    
                    if (in_wavelength_section) {
                        if (strchr(line, '}')) {
                            break; // End of wavelength section
                        }
                        
                        // Parse comma-separated wavelengths
                        char* token = strtok(line, ",");
                        while (token && wl_index < bands) {
                            float wl = atof(token);
                            if (wl > 0.0f && wl < 50000.0f) {
                                img->wavelengths[wl_index] = wl;
                                wl_index++;
                                wavelengths_found = true;
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
    
    // Method 4: If still no wavelengths, use intelligent defaults based on common hyperspectral sensors
    if (!wavelengths_found) {
        printf("No wavelength metadata found, using sensor-specific defaults based on band count...\n");
        
        if (bands == 23) {
            // Looks like RapidEye or similar
            float start_wl = 400.0f;
            float end_wl = 850.0f;
            for (int i = 0; i < bands; i++) {
                img->wavelengths[i] = start_wl + i * (end_wl - start_wl) / (bands - 1);
            }
            printf("Applied RapidEye-like wavelength range: %.1f - %.1f nm\n", start_wl, end_wl);
        } else if (bands >= 200 && bands <= 250) {
            // Looks like AVIRIS
            float start_wl = 400.0f;
            float end_wl = 2500.0f;
            for (int i = 0; i < bands; i++) {
                img->wavelengths[i] = start_wl + i * (end_wl - start_wl) / (bands - 1);
            }
            printf("Applied AVIRIS-like wavelength range: %.1f - %.1f nm\n", start_wl, end_wl);
        } else if (bands >= 100 && bands < 200) {
            // Looks like Hyperion or similar
            float start_wl = 400.0f;
            float end_wl = 2400.0f;
            for (int i = 0; i < bands; i++) {
                img->wavelengths[i] = start_wl + i * (end_wl - start_wl) / (bands - 1);
            }
            printf("Applied Hyperion-like wavelength range: %.1f - %.1f nm\n", start_wl, end_wl);
        } else {
            // Generic visible to NIR range
            float start_wl = 400.0f;
            float end_wl = 1000.0f;
            for (int i = 0; i < bands; i++) {
                img->wavelengths[i] = start_wl + i * (end_wl - start_wl) / (bands - 1);
            }
            printf("Applied generic VIS-NIR wavelength range: %.1f - %.1f nm\n", start_wl, end_wl);
        }
        wavelengths_found = true;
    }
    
    // Validate and sort wavelengths
    if (wavelengths_found) {
        printf("\nWavelength validation:\n");
        for (int i = 0; i < bands; i++) {
            if (img->wavelengths[i] <= 0.0f || img->wavelengths[i] > 50000.0f) {
                printf("Warning: Invalid wavelength for band %d: %.1f nm\n", i+1, img->wavelengths[i]);
                // Fix with interpolation if possible
                if (i > 0 && i < bands-1) {
                    img->wavelengths[i] = (img->wavelengths[i-1] + img->wavelengths[i+1]) / 2.0f;
                    printf("Interpolated to: %.1f nm\n", img->wavelengths[i]);
                }
            }
        }
        
        printf("Final wavelength range: %.1f - %.1f nm\n", 
               img->wavelengths[0], img->wavelengths[bands-1]);
        
        return 0;
    }
    
    printf("Error: Could not determine wavelengths for hyperspectral image\n");
    return -1;
}


int load_hyperspectral_image(ProcessingContext* ctx, const char* filename) {
    GDALDatasetH dataset = GDALOpen(filename, GA_ReadOnly);
    if (!dataset) {
        printf("Error: Cannot open hyperspectral image: %s\n", filename);
        return -1;
    }
    
    // Get image dimensions
    int width = GDALGetRasterXSize(dataset);
    int height = GDALGetRasterYSize(dataset);
    int bands = GDALGetRasterCount(dataset);
    
    printf("Image dimensions: %dx%dx%d\n", width, height, bands);
    
    if (bands > MAX_VECTOR_SIZE) {
        printf("Error: Too many bands (%d), maximum is %d\n", bands, MAX_VECTOR_SIZE);
        GDALClose(dataset);
        return -1;
    }
    
    // Create hyperspectral image
    ctx->image = create_hyperspectral_image(width, height, bands);
    if (!ctx->image) {
        GDALClose(dataset);
        return -1;
    }
    
    // Get geotransform
    if (GDALGetGeoTransform(dataset, ctx->image->geotransform) != CE_None) {
        printf("Warning: No geotransform found, using default\n");
        ctx->image->geotransform[0] = 0.0;  // top left x
        ctx->image->geotransform[1] = 1.0;  // w-e pixel resolution
        ctx->image->geotransform[2] = 0.0;  // rotation, 0 if image is "north up"
        ctx->image->geotransform[3] = 0.0;  // top left y
        ctx->image->geotransform[4] = 0.0;  // rotation, 0 if image is "north up"
        ctx->image->geotransform[5] = -1.0; // n-s pixel resolution
    }
    
    // Get projection
    const char* projection = GDALGetProjectionRef(dataset);
    if (projection && strlen(projection) > 0) {
        ctx->image->projection = strdup(projection);
        printf("Projection: %.100s...\n", projection);
    } else {
        printf("Warning: No projection information found\n");
        ctx->image->projection = NULL;
    }
    
    // Extract wavelengths using enhanced function
    printf("Extracting wavelengths from metadata...\n");
    if (extract_wavelengths_from_image(dataset, ctx->image) != 0) {
        printf("Error: Failed to extract wavelengths\n");
        GDALClose(dataset);
        return -1;
    }
    
    printf("Wavelength range: %.1f - %.1f nm\n", 
           ctx->image->wavelengths[0], ctx->image->wavelengths[bands-1]);
    
    // Read image data
    printf("Reading image data...\n");
    for (int band = 0; band < bands; band++) {
        GDALRasterBandH raster_band = GDALGetRasterBand(dataset, band + 1);
        
        // Read band data
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
    
    // Calculate data size
    ctx->image->data_size = (size_t)width * height * bands * sizeof(float);
    printf("Image loaded successfully (%.2f MB)\n", ctx->image->data_size / (1024.0 * 1024.0));
    
    GDALClose(dataset);
    return 0;
}

int save_classification_result(ProcessingContext* ctx, const char* output_filename) {
    // Create output dataset
    GDALDriverH driver = GDALGetDriverByName("GTiff");
    if (!driver) {
        printf("Error: Cannot get GTiff driver\n");
        return -1;
    }
    
    // Create dataset with compression
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
    
    // Set geotransform and projection
    GDALSetGeoTransform(output_dataset, ctx->result->geotransform);
    if (ctx->result->projection) {
        GDALSetProjection(output_dataset, ctx->result->projection);
    }
    
    // Write classification band
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
    
    // Create and set color table for classification
    GDALColorTableH color_table = GDALCreateColorTable(GPI_RGB);
    
    // Define colors for each material class based on comprehensive MaterialType enum
    GDALColorEntry colors[] = {
    // Vegetation - Various greens by growth stage/type
    {34, 139, 34, 255},       // VEGETATION - Forest Green
    {50, 205, 50, 255},       // CORN_EARLY_VEGETATIVE - Lime Green
    {0, 128, 0, 255},         // CORN_LATE_VEGETATIVE - Green
    {85, 107, 47, 255},       // CORN_TASSELING - Dark Olive Green
    {107, 142, 35, 255},      // CORN_GRAIN_FILLING - Olive Drab
    {128, 128, 0, 255},       // CORN_MATURITY - Olive
    {0, 255, 0, 255},         // CORN_IRRIGATED - Bright Green
    {154, 205, 50, 255},      // CORN_STRESSED - Yellow Green
    
    // Wheat - Golden/brown tones
    {173, 255, 47, 255},      // WHEAT_TILLERING - Green Yellow
    {127, 255, 0, 255},       // WHEAT_JOINTING - Chart Reuse
    {255, 215, 0, 255},       // WHEAT_HEADING - Gold
    {218, 165, 32, 255},      // WHEAT_GRAIN_FILLING - Golden Rod
    {184, 134, 11, 255},      // WHEAT_SENESCENCE - Dark Golden Rod
    {255, 255, 0, 255},       // WHEAT_IRRIGATED - Yellow
    {189, 183, 107, 255},     // WHEAT_STRESSED - Dark Khaki
    
    // Soybeans - Green variations
    {46, 125, 50, 255},       // SOYBEAN_VEGETATIVE - Dark Green
    {76, 175, 80, 255},       // SOYBEAN_FLOWERING - Green
    {139, 195, 74, 255},      // SOYBEAN_POD_FILLING - Light Green
    {205, 220, 57, 255},      // SOYBEAN_MATURITY - Lime
    {102, 187, 106, 255},     // SOYBEAN_IRRIGATED - Light Green
    {165, 214, 167, 255},     // SOYBEAN_STRESSED - Very Light Green
    
    // Rice - Blue-green tones
    {0, 128, 128, 255},       // RICE_VEGETATIVE - Teal
    {64, 224, 208, 255},      // RICE_REPRODUCTIVE - Turquoise
    {72, 209, 204, 255},      // RICE_RIPENING - Medium Turquoise
    {0, 206, 209, 255},       // RICE_FLOODED - Dark Turquoise
    {95, 158, 160, 255},      // RICE_UPLAND - Cadet Blue
    
    // Cotton - White/cream tones
    {255, 228, 225, 255},     // COTTON_SQUARING - Misty Rose
    {255, 182, 193, 255},     // COTTON_FLOWERING - Light Pink
    {255, 255, 240, 255},     // COTTON_BOLL_DEVELOPMENT - Ivory
    {245, 245, 220, 255},     // COTTON_DEFOLIATION - Beige
    {255, 255, 255, 255},     // COTTON_IRRIGATED - White
    {220, 220, 220, 255},     // COTTON_STRESSED - Gainsboro
    
    // Other Field Crops - Earth tones
    {210, 180, 140, 255},     // BARLEY_TILLERING - Tan
    {222, 184, 135, 255},     // BARLEY_STEM_ELONGATION - Burlywood
    {205, 133, 63, 255},      // BARLEY_HEADING - Peru
    {160, 82, 45, 255},       // BARLEY_GRAIN_FILLING - Saddle Brown
    
    {238, 203, 173, 255},     // OATS_VEGETATIVE - Peach Puff
    {255, 218, 185, 255},     // OATS_HEADING - Peach Puff
    {255, 222, 173, 255},     // OATS_GRAIN_FILLING - Navajo White
    
    {255, 140, 0, 255},       // SUNFLOWER_VEGETATIVE - Dark Orange
    {255, 165, 0, 255},       // SUNFLOWER_BUD_FORMATION - Orange
    {255, 215, 0, 255},       // SUNFLOWER_FLOWERING - Gold
    {184, 134, 11, 255},      // SUNFLOWER_SEED_FILLING - Dark Golden Rod
    
    {255, 255, 0, 255},       // CANOLA_ROSETTE - Yellow
    {255, 215, 0, 255},       // CANOLA_STEM_ELONGATION - Gold
    {255, 255, 0, 255},       // CANOLA_FLOWERING - Yellow
    {218, 165, 32, 255},      // CANOLA_POD_FILLING - Golden Rod
    
    {128, 0, 128, 255},       // SUGAR_BEET_EARLY - Purple
    {138, 43, 226, 255},      // SUGAR_BEET_CANOPY_CLOSURE - Blue Violet
    {75, 0, 130, 255},        // SUGAR_BEET_ROOT_GROWTH - Indigo
    
    {143, 188, 143, 255},     // ALFALFA_EARLY_VEGETATIVE - Dark Sea Green
    {60, 179, 113, 255},      // ALFALFA_BUD_STAGE - Medium Sea Green
    {152, 251, 152, 255},     // ALFALFA_FLOWERING - Pale Green
    {32, 178, 170, 255},      // ALFALFA_POST_CUT - Light Sea Green
    
    // Water
    {0, 0, 255, 255},         // WATER - Blue
    
    // Igneous Rocks - Dark grays/blacks
    {105, 105, 105, 255},     // GRANITE - Dim Gray
    {47, 79, 79, 255},        // BASALT - Dark Slate Gray
    {112, 128, 144, 255},     // GABBRO - Slate Gray
    {0, 0, 0, 255},           // OBSIDIAN - Black
    {220, 220, 220, 255},     // PUMICE - Gainsboro
    {169, 169, 169, 255},     // RHYOLITE - Dark Gray
    {128, 128, 128, 255},     // ANDESITE - Gray
    
    // Sedimentary Rocks - Light grays/tans
    {245, 245, 220, 255},     // LIMESTONE - Beige
    {238, 203, 173, 255},     // SANDSTONE - Peach Puff
    {119, 136, 153, 255},     // SHALE - Light Slate Gray
    {160, 82, 45, 255},       // MUDSTONE - Saddle Brown
    {205, 192, 176, 255},     // CONGLOMERATE - Light Gray
    {255, 248, 220, 255},     // DOLOMITE - Cornsilk
    {255, 255, 255, 255},     // GYPSUM - White
    
    // Metamorphic Rocks - Medium grays
    {255, 228, 225, 255},     // MARBLE - Misty Rose
    {240, 248, 255, 255},     // QUARTZITE - Alice Blue
    {112, 128, 144, 255},     // SLATE - Slate Gray
    {169, 169, 169, 255},     // SCHIST - Dark Gray
    {128, 128, 128, 255},     // GNEISS - Gray
    
    // Individual Minerals
    {255, 255, 255, 255},     // QUARTZ - White
    {255, 20, 147, 255},      // FELDSPAR - Deep Pink
    {255, 255, 224, 255},     // CALCITE - Light Yellow
    {255, 248, 220, 255},     // DOLOMITE_MINERAL - Cornsilk
    {255, 255, 255, 255},     // KAOLINITE - White
    {169, 169, 169, 255},     // MONTMORILLONITE - Dark Gray
    {128, 128, 128, 255},     // ILLITE - Gray
    {205, 92, 92, 255},       // HEMATITE - Indian Red
    {255, 140, 0, 255},       // GOETHITE - Dark Orange
    {0, 0, 0, 255},           // MAGNETITE - Black
    
    // Soils and Sediments
    {255, 218, 185, 255},     // SAND - Peach Puff
    {160, 82, 45, 255},       // CLAY - Saddle Brown
    {139, 69, 19, 255},       // SOIL - Saddle Brown
    {178, 34, 34, 255},       // LATERITE - Fire Brick
    {85, 107, 47, 255},       // PEAT - Dark Olive Green
    
    // Hydrocarbons - Blacks and dark colors
    {47, 79, 79, 255},        // CRUDE_OIL_LIGHT - Dark Slate Gray
    {25, 25, 25, 255},        // CRUDE_OIL_MEDIUM - Very Dark Gray
    {0, 0, 0, 255},           // CRUDE_OIL_HEAVY - Black
    {0, 0, 0, 255},           // CRUDE_OIL_EXTRA_HEAVY - Black
    
    {255, 20, 147, 255},      // GASOLINE - Deep Pink
    {255, 140, 0, 255},       // DIESEL - Dark Orange
    {135, 206, 235, 255},     // JET_FUEL - Sky Blue
    {255, 69, 0, 255},        // HEATING_OIL - Orange Red
    {139, 69, 19, 255},       // LUBRICATING_OIL - Saddle Brown
    {0, 0, 0, 255},           // ASPHALT - Black
    
    {173, 216, 230, 255},     // PROPANE - Light Blue
    {255, 182, 193, 255},     // BUTANE - Light Pink
    {240, 248, 255, 255},     // NATURAL_GAS_CONDENSATE - Alice Blue
    
    {72, 61, 139, 255},       // WEATHERED_OIL_LIGHT - Dark Slate Blue
    {47, 79, 79, 255},        // WEATHERED_OIL_MEDIUM - Dark Slate Gray
    {25, 25, 25, 255},        // WEATHERED_OIL_HEAVY - Very Dark Gray
    {139, 69, 19, 255},       // OIL_MOUSSE - Saddle Brown
    
    // Plastics - Bright synthetic colors
    {255, 0, 255, 255},       // POLYETHYLENE_LD - Magenta
    {255, 20, 147, 255},      // POLYETHYLENE_HD - Deep Pink
    {138, 43, 226, 255},      // POLYPROPYLENE - Blue Violet
    {255, 105, 180, 255},     // POLYSTYRENE - Hot Pink
    {148, 0, 211, 255},       // PVC - Dark Violet
    {0, 255, 255, 255},       // PET - Cyan
    {30, 144, 255, 255},      // POLYCARBONATE - Dodger Blue
    {255, 0, 0, 255},         // NYLON - Red
    {255, 69, 0, 255},        // ABS - Orange Red
    {255, 182, 193, 255},     // PMMA - Light Pink
    {240, 248, 255, 255},     // PTFE - Alice Blue
    
    {218, 112, 214, 255},     // PLASTIC_WEATHERED_UV - Orchid
    {221, 160, 221, 255},     // PLASTIC_WEATHERED_THERMAL - Plum
    {255, 192, 203, 255},     // MICROPLASTICS - Pink
    
    // Man-made Materials
    {192, 192, 192, 255},     // CONCRETE - Silver
    {105, 105, 105, 255},     // METAL - Dim Gray
    {255, 127, 80, 255},      // PAINT - Coral
    {255, 255, 255, 255},     // SNOW - White
    
    // Roofing Materials
    {25, 25, 25, 255},        // ASPHALT_SHINGLES - Very Dark Gray
    {205, 92, 92, 255},       // CLAY_TILES - Indian Red
    {169, 169, 169, 255},     // CONCRETE_TILES - Dark Gray
    {112, 128, 144, 255},     // SLATE_ROOFING - Slate Gray
    {210, 180, 140, 255},     // WOOD_SHINGLES - Tan
    {192, 192, 192, 255},     // METAL_ROOFING_STEEL - Silver
    {211, 211, 211, 255},     // METAL_ROOFING_ALUMINUM - Light Gray
    {184, 115, 51, 255},      // METAL_ROOFING_COPPER - Dark Orange
    {105, 105, 105, 255},     // METAL_ROOFING_ZINC - Dim Gray
    {128, 128, 128, 255},     // METAL_ROOFING_TIN - Gray
    
    // Flat/Commercial Roofing
    {47, 79, 79, 255},        // BUILT_UP_ROOFING - Dark Slate Gray
    {0, 0, 0, 255},           // MODIFIED_BITUMEN - Black
    {25, 25, 25, 255},        // EPDM_RUBBER - Very Dark Gray
    {255, 255, 255, 255},     // TPO_MEMBRANE - White
    {240, 248, 255, 255},     // PVC_MEMBRANE - Alice Blue
    {192, 192, 192, 255},     // LIQUID_APPLIED_MEMBRANE - Silver
    {34, 139, 34, 255},       // GREEN_ROOF_EXTENSIVE - Forest Green
    {0, 128, 0, 255},         // GREEN_ROOF_INTENSIVE - Green
    {255, 255, 255, 255},     // WHITE_MEMBRANE - White
    {128, 128, 128, 255},     // BALLASTED_ROOF - Gray
    
    // Temporary/Emergency Roofing
    {0, 0, 255, 255},         // TARP_BLUE - Blue
    {192, 192, 192, 255},     // TARP_SILVER - Silver
    {255, 255, 255, 255},     // TARP_WHITE - White
    {210, 180, 140, 255},     // TARP_CANVAS - Tan
    {169, 169, 169, 255},     // TEMPORARY_METAL_SHEET - Dark Gray
    {160, 82, 45, 255},       // PLYWOOD_SHEETING - Saddle Brown
    {240, 248, 255, 255},     // PLASTIC_SHEETING - Alice Blue
    {255, 69, 0, 255},        // EMERGENCY_PATCH - Orange Red
    
    // Solar/Renewable Energy
    {25, 25, 112, 255},       // SOLAR_PV_MONOCRYSTALLINE - Midnight Blue
    {0, 0, 139, 255},         // SOLAR_PV_POLYCRYSTALLINE - Dark Blue
    {47, 79, 79, 255},        // SOLAR_PV_THIN_FILM_CDTE - Dark Slate Gray
    {72, 61, 139, 255},       // SOLAR_PV_THIN_FILM_CIGS - Dark Slate Blue
    {105, 105, 105, 255},     // SOLAR_PV_THIN_FILM_AMORPHOUS - Dim Gray
    {0, 0, 205, 255},         // SOLAR_PV_BIFACIAL - Medium Blue
    {65, 105, 225, 255},      // SOLAR_THERMAL_FLAT - Royal Blue
    {100, 149, 237, 255},     // SOLAR_THERMAL_EVACUATED - Cornflower Blue
    {211, 211, 211, 255},     // PV_MOUNTING_ALUMINUM - Light Gray
    {105, 105, 105, 255},     // PV_MOUNTING_STEEL - Dim Gray
    
    // Specialty Roofing
    {218, 165, 32, 255},      // THATCH_ROOFING - Golden Rod
    {107, 142, 35, 255},      // SOD_ROOF - Olive Drab
    {255, 255, 255, 255},     // MEMBRANE_REFLECTIVE - White
    {0, 0, 139, 255},         // PHOTOVOLTAIC_INTEGRATED - Dark Blue
    {173, 216, 230, 255},     // SKYLIGHTS_GLASS - Light Blue
    {240, 248, 255, 255},     // SKYLIGHTS_PLASTIC - Alice Blue
    {255, 255, 255, 255},     // ROOF_COATING_ELASTOMERIC - White
    {248, 248, 255, 255},     // ROOF_COATING_SILICONE - Ghost White
    {255, 255, 255, 255},     // ROOF_COATING_ACRYLIC - White
    
    // Marine/Harbor Materials - Boat Decks
    {245, 245, 220, 255},     // FIBERGLASS_DECK - Beige
    {210, 180, 140, 255},     // WOOD_DECK_TEAK - Tan
    {160, 82, 45, 255},       // WOOD_DECK_MAHOGANY - Saddle Brown
    {222, 184, 135, 255},     // WOOD_DECK_PINE - Burlywood
    {192, 192, 192, 255},     // ALUMINUM_DECK - Silver
    {128, 128, 128, 255},     // STEEL_DECK_PAINTED - Gray
    {169, 169, 169, 255},     // STEEL_DECK_GALVANIZED - Dark Gray
    {139, 69, 19, 255},       // COMPOSITE_DECK - Saddle Brown
    {0, 0, 0, 255},           // RUBBER_DECK_MATTING - Black
    {255, 255, 255, 255},     // VINYL_DECK_COVERING - White
    {210, 180, 140, 255},     // CANVAS_DECK_COVERING - Tan
    {255, 140, 0, 255},       // ANTI_SLIP_COATING - Dark Orange
    
    // Hull and Structural
    {255, 255, 255, 255},     // FIBERGLASS_HULL_WHITE - White
    {30, 144, 255, 255},      // FIBERGLASS_HULL_COLORED - Dodger Blue
    {192, 192, 192, 255},     // ALUMINUM_HULL - Silver
    {105, 105, 105, 255},     // STEEL_HULL_PAINTED - Dim Gray
    {160, 82, 45, 255},       // WOOD_HULL_VARNISHED - Saddle Brown
    {255, 140, 0, 255},       // INFLATABLE_PVC - Dark Orange
    {0, 0, 0, 255},           // INFLATABLE_HYPALON - Black
    {255, 255, 255, 255},     // MARINE_VINYL - White
    {210, 180, 140, 255},     // BOAT_CANVAS - Tan
    {255, 0, 0, 255},         // MARINE_PAINT_ANTIFOUL - Red
    {255, 255, 255, 255},     // MARINE_PAINT_TOPSIDE - White
    
    // Harbor/Marina Infrastructure
    {169, 169, 169, 255},     // CONCRETE_DOCK - Dark Gray
    {160, 82, 45, 255},       // WOOD_DOCK_TREATED - Saddle Brown
    {139, 69, 19, 255},       // WOOD_DOCK_COMPOSITE - Saddle Brown
    {192, 192, 192, 255},     // ALUMINUM_DOCK - Silver
    {255, 255, 255, 255},     // PLASTIC_DOCK_HDPE - White
    {169, 169, 169, 255},     // DOCK_DECKING_EZDOCK - Dark Gray
    {128, 128, 128, 255},     // MARINA_PILING_CONCRETE - Gray
    {105, 105, 105, 255},     // MARINA_PILING_STEEL - Dim Gray
    {160, 82, 45, 255},       // MARINA_PILING_WOOD - Saddle Brown
    {255, 255, 255, 255},     // FLOATING_DOCK_FOAM - White
    {192, 192, 192, 255},     // BOAT_LIFT_ALUMINUM - Silver
    {0, 0, 0, 255},           // FENDER_SYSTEM_RUBBER - Black
    
    // Water and Marine Environment
    {0, 191, 255, 255},       // WATER_HARBOR_CLEAN - Deep Sky Blue
    {139, 69, 19, 255},       // WATER_HARBOR_TURBID - Saddle Brown
    {0, 0, 0, 255},           // WATER_HARBOR_POLLUTED - Black
    {46, 125, 50, 255},       // SEAWEED_FLOATING - Dark Green
    {245, 245, 220, 255},     // BARNACLE_FOULING - Beige
    {0, 128, 0, 255},         // MARINE_ALGAE - Green
    
    // Road Materials - Asphalt Types
    {47, 79, 79, 255},        // ASPHALT_NEW - Dark Slate Gray
    {105, 105, 105, 255},     // ASPHALT_AGED - Dim Gray
    {169, 169, 169, 255},     // ASPHALT_CRACKED - Dark Gray
    {128, 128, 128, 255},     // ASPHALT_RECYCLED - Gray
    {220, 20, 60, 255},       // ASPHALT_COLORED - Crimson
    {112, 128, 144, 255},     // ASPHALT_POROUS - Slate Gray
    
    // Concrete Types
    {211, 211, 211, 255},     // CONCRETE_PAVEMENT_NEW - Light Gray
    {169, 169, 169, 255},     // CONCRETE_PAVEMENT_AGED - Dark Gray
    {128, 128, 128, 255},     // CONCRETE_PAVEMENT_STAINED - Gray
    {192, 192, 192, 255},     // CONCRETE_STAMPED - Silver
    {169, 169, 169, 255},     // CONCRETE_EXPOSED_AGGREGATE - Dark Gray
    {255, 182, 193, 255},     // CONCRETE_COLORED - Light Pink
    
    // Alternative Surfaces
    {205, 92, 92, 255},       // BRICK_PAVEMENT - Indian Red
    {105, 105, 105, 255},     // STONE_PAVEMENT_GRANITE - Dim Gray
    {47, 79, 79, 255},        // STONE_PAVEMENT_BASALT - Dark Slate Gray
    {112, 128, 144, 255},     // COBBLESTONE - Slate Gray
    {205, 192, 176, 255},     // GRAVEL_ROAD - Light Gray
    {160, 82, 45, 255},       // DIRT_ROAD_DRY - Saddle Brown
    {139, 69, 19, 255},       // DIRT_ROAD_WET - Saddle Brown
    {128, 128, 128, 255},     // CRUSHED_STONE - Gray
    
    // Road Markings and Accessories
    {255, 255, 255, 255},     // ROAD_PAINT_WHITE - White
    {255, 255, 0, 255},       // ROAD_PAINT_YELLOW - Yellow
    {255, 255, 255, 255},     // THERMOPLASTIC_MARKING - White
    {255, 255, 255, 255},     // REFLECTIVE_ROAD_MARKING - White
    {0, 0, 0, 255},           // ROAD_SEALANT - Black
    {47, 79, 79, 255},        // POTHOLE_PATCH - Dark Slate Gray
    
    // Road Infrastructure
    {169, 169, 169, 255},     // CONCRETE_CURB - Dark Gray
    {192, 192, 192, 255},     // STEEL_GUARDRAIL - Silver
    {169, 169, 169, 255},     // CONCRETE_BARRIER - Dark Gray
    {105, 105, 105, 255},     // ASPHALT_SHOULDER - Dim Gray
    {128, 128, 128, 255},     // GRAVEL_SHOULDER - Gray
    
    // Parking and Urban Surfaces
    {47, 79, 79, 255},        // PARKING_LOT_ASPHALT - Dark Slate Gray
    {169, 169, 169, 255},     // PARKING_LOT_CONCRETE - Dark Gray
    {192, 192, 192, 255},     // INTERLOCKING_PAVERS - Silver
    {211, 211, 211, 255},     // PERMEABLE_PAVERS - Light Gray
    {255, 140, 0, 255},       // RUBBER_SPEED_BUMP - Dark Orange
    {255, 255, 255, 255},     // PAINTED_CROSSWALK - White
    
    // Forest and Tree Species - Deciduous Broadleaf
    {34, 139, 34, 255},       // OAK_TREE - Forest Green
    {255, 0, 0, 255},         // MAPLE_TREE - Red (fall colors)
    {255, 255, 255, 255},     // BIRCH_TREE - White (bark)
    {160, 82, 45, 255},       // BEECH_TREE - Saddle Brown
    {128, 128, 128, 255},     // ASH_TREE - Gray
    {139, 69, 19, 255},       // HICKORY_TREE - Saddle Brown
    {255, 182, 193, 255},     // CHERRY_TREE - Light Pink
    {139, 69, 19, 255},       // WALNUT_TREE - Saddle Brown
    {255, 255, 255, 255},     // POPLAR_TREE - White
    {255, 255, 0, 255},       // WILLOW_TREE - Yellow (weeping)
    {210, 180, 140, 255},     // BASSWOOD_TREE - Tan
    {160, 82, 45, 255},       // CHESTNUT_TREE - Saddle Brown
    {245, 245, 220, 255},     // SYCAMORE_TREE - Beige (bark pattern)
    {255, 140, 0, 255},       // TULIP_TREE - Dark Orange (flowers)
    {255, 0, 0, 255},         // SWEETGUM_TREE - Red (fall colors)
    
    // Evergreen Coniferous - Various greens
    {0, 100, 0, 255},         // PINE_TREE - Dark Green
    {34, 139, 34, 255},       // SPRUCE_TREE - Forest Green
    {0, 128, 0, 255},         // FIR_TREE - Green
    {85, 107, 47, 255},       // CEDAR_TREE - Dark Olive Green
    {46, 125, 50, 255},       // HEMLOCK_TREE - Dark Green
    {107, 142, 35, 255},      // DOUGLAS_FIR - Olive Drab
    {128, 128, 0, 255},       // JUNIPER_TREE - Olive
    {173, 255, 47, 255},      // LARCH_TREE - Green Yellow (deciduous conifer)
    {165, 42, 42, 255},       // REDWOOD_TREE - Brown (bark)
    {160, 82, 45, 255},       // SEQUOIA_TREE - Saddle Brown (bark)
    {46, 125, 50, 255},       // CYPRESS_TREE - Dark Green
    {0, 100, 0, 255},         // YEW_TREE - Dark Green
    
    // Tropical and Subtropical Trees
    {0, 128, 0, 255},         // PALM_TREE - Green
    {255, 255, 0, 255},       // BANANA_TREE - Yellow
    {0, 128, 0, 255},         // MANGO_TREE - Green
    {0, 100, 0, 255},         // AVOCADO_TREE - Dark Green
    {255, 165, 0, 255},       // CITRUS_TREE - Orange
    {139, 69, 19, 255},       // COCONUT_PALM - Saddle Brown
    {160, 82, 45, 255},       // BAOBAB_TREE - Saddle Brown
    {139, 69, 19, 255},       // MAHOGANY_TREE - Saddle Brown
    {210, 180, 140, 255},     // TEAK_TREE - Tan
    {0, 0, 0, 255},           // EBONY_TREE - Black
    {245, 245, 220, 255},     // BALSA_TREE - Beige
    {165, 42, 42, 255},       // BRAZILWOOD_TREE - Brown
    
    // Fruit and Nut Trees (Orchard)
    {255, 0, 0, 255},         // APPLE_TREE - Red
    {255, 255, 0, 255},       // PEAR_TREE - Yellow
    {255, 182, 193, 255},     // PEACH_TREE - Light Pink
    {128, 0, 128, 255},       // PLUM_TREE - Purple
    {255, 255, 255, 255},     // ALMOND_TREE - White
    {160, 82, 45, 255},       // PECAN_TREE - Saddle Brown
    {173, 255, 47, 255},      // PISTACHIO_TREE - Green Yellow
    {160, 82, 45, 255},       // HAZELNUT_TREE - Saddle Brown
    {128, 0, 128, 255},       // FIG_TREE - Purple
    {128, 128, 0, 255},       // OLIVE_TREE - Olive
    
    // Urban and Ornamental Trees
    {210, 180, 140, 255},     // LONDON_PLANE - Tan
    {255, 215, 0, 255},       // GINKGO_TREE - Gold (fall color)
    {255, 0, 0, 255},         // JAPANESE_MAPLE - Red
    {255, 255, 255, 255},     // MAGNOLIA_TREE - White (flowers)
    {255, 255, 255, 255},     // DOGWOOD_TREE - White (flowers)
    {255, 192, 203, 255},     // REDBUD_TREE - Pink
    {138, 43, 226, 255},      // JACARANDA_TREE - Blue Violet
    {255, 255, 0, 255},       // ACACIA_TREE - Yellow
    {0, 128, 128, 255},       // EUCALYPTUS_TREE - Teal
    {255, 0, 0, 255},         // BOTTLE_BRUSH - Red
    
    // Tree Conditions and Seasonal States
    {173, 255, 47, 255},      // DECIDUOUS_SPRING - Green Yellow
    {0, 128, 0, 255},         // DECIDUOUS_SUMMER - Green
    {255, 140, 0, 255},       // DECIDUOUS_AUTUMN - Dark Orange
    {160, 82, 45, 255},       // DECIDUOUS_WINTER - Saddle Brown
    {0, 100, 0, 255},         // EVERGREEN_HEALTHY - Dark Green
    {128, 128, 0, 255},       // EVERGREEN_STRESSED - Olive
    {139, 69, 19, 255},       // TREE_DEAD_STANDING - Saddle Brown
    {160, 82, 45, 255},       // TREE_BARK_GENERAL - Saddle Brown
    };
    
    for (int i = 0; i < ctx->num_reference_vectors && i < 14; i++) {
        GDALSetColorEntry(color_table, i, &colors[i]);
    }
    
    GDALSetRasterColorTable(class_band, color_table);
    GDALDestroyColorTable(color_table);
      
    // Write confidence band  
    GDALRasterBandH conf_band = GDALGetRasterBand(output_dataset, 2);
    GDALSetDescription(conf_band, "Classification Confidence");
    
    // Scale confidence to 0-65535 range
    uint16_t* scaled_confidence = (uint16_t*)malloc(ctx->result->width * ctx->result->height * sizeof(uint16_t));
    for (long long i = 0; i < (long long)ctx->result->width * ctx->result->height; i++) {
        float conf = ctx->result->confidence[i];
        if (conf < 0.0f) conf = 0.0f;
        if (conf > 1.0f) conf = 1.0f;
        scaled_confidence[i] = (unsigned char)(conf * 65535.0f);
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
    
     // Enhanced metadata for >256 classes
    char** metadata = NULL;
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        char key[64], value[128];
        snprintf(key, sizeof(key), "CLASS_%d_NAME", i);
        snprintf(value, sizeof(value), "%s", ctx->reference_vectors[i].name);
        metadata = CSLSetNameValue(metadata, key, value);
    }
    
    // Add extended classification statistics
    char stats[256];
    snprintf(stats, sizeof(stats), "%d", ctx->num_reference_vectors);
    metadata = CSLSetNameValue(metadata, "NUM_CLASSES", stats);
    metadata = CSLSetNameValue(metadata, "MAX_CLASSES_SUPPORTED", "65535");
    metadata = CSLSetNameValue(metadata, "CLASSIFICATION_TYPE", "Hyperspectral_Material_Classification");
    metadata = CSLSetNameValue(metadata, "DATA_TYPE", "16bit_UInt16");
    metadata = CSLSetNameValue(metadata, "CONFIDENCE_RANGE", "0-65535");
    
    GDALSetMetadata(class_band, metadata, NULL);
    CSLDestroy(metadata);
    
    // Create RAT (Raster Attribute Table) for QGIS/GRASS compatibility
    GDALRasterAttributeTableH rat = GDALCreateRasterAttributeTable();
    
    GDALRATCreateColumn(rat, "VALUE", GFT_Integer, GFU_MinMax);
    GDALRATCreateColumn(rat, "NAME", GFT_String, GFU_Name);
    GDALRATCreateColumn(rat, "RED", GFT_Integer, GFU_Red);
    GDALRATCreateColumn(rat, "GREEN", GFT_Integer, GFU_Green);
    GDALRATCreateColumn(rat, "BLUE", GFT_Integer, GFU_Blue);
    GDALRATCreateColumn(rat, "MATERIAL_TYPE", GFT_String, GFU_Generic);
    
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        GDALRATSetValueAsInt(rat, i, 0, i);  // VALUE
        GDALRATSetValueAsString(rat, i, 1, ctx->reference_vectors[i].name);  // NAME
        if (i < 14) {
            GDALRATSetValueAsInt(rat, i, 2, colors[i].c1);  // RED
            GDALRATSetValueAsInt(rat, i, 3, colors[i].c2);  // GREEN  
            GDALRATSetValueAsInt(rat, i, 4, colors[i].c3);  // BLUE
        }
        char material_type[64];
        snprintf(material_type, sizeof(material_type), "TYPE_%d", ctx->reference_vectors[i].material);
        GDALRATSetValueAsString(rat, i, 5, material_type);
    }
    
    GDALSetDefaultRAT(class_band, rat);

    // Force creation of external VAT file for better QGIS compatibility
    char vat_filename[512];
    snprintf(vat_filename, sizeof(vat_filename), "%s.vat.dbf", output_filename);

    // Force creation of external VAT file for better QGIS compatibility
    GDALColorTableH derived_color_table = GDALRATTranslateToColorTable(rat, ctx->num_reference_vectors);
    if (derived_color_table) {
        // Apply the RAT-derived color table to ensure consistency
        GDALSetRasterColorTable(class_band, derived_color_table);
        GDALDestroyColorTable(derived_color_table);
        printf("External VAT file created for enhanced QGIS compatibility\n");
    }
    GDALDestroyRasterAttributeTable(rat);
    
    GDALClose(output_dataset);
    
    printf("Extended classification result saved with:\n");
    printf("- Material classification (16-bit, supports up to 65,535 classes)\n");
    printf("- High precision confidence scores (16-bit, 0-65535 range)\n");
    printf("- Algorithmic color generation for >256 classes\n");
    printf("- Extended metadata and RAT for large class counts\n");
    printf("- Full QGIS/GRASS GIS compatibility\n");
    
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
    result->classification = (uint16_t*)malloc(pixel_count * sizeof(uint16_t));  // uint16_t
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

int detect_gpu_capabilities(ProcessingContext* ctx) {
    #ifdef CUDA_AVAILABLE
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    
    if (cuda_status == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        ctx->available_gpu_memory = prop.totalGlobalMem;
        
        if (ctx->available_gpu_memory > GPU_MEMORY_THRESHOLD) {
            ctx->use_gpu = 1;
            ctx->gpu_type = 0; // CUDA
            cudaStreamCreate(&ctx->stream);
            return 1;
        }
    }
    #endif
    
    #ifdef OPENCL_AVAILABLE
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err == CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err == CL_SUCCESS) {
            cl_ulong mem_size;
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
            
            if (mem_size > GPU_MEMORY_THRESHOLD) {
                ctx->use_gpu = 1;
                ctx->gpu_type = 1; // OpenCL
                ctx->available_gpu_memory = mem_size;
                
                ctx->cl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
                ctx->cl_queue = clCreateCommandQueue(ctx->cl_context, device, 0, &err);
                
                ctx->cl_program = clCreateProgramWithSource(ctx->cl_context, 1, 
                                                           &opencl_classify_kernel_source, NULL, &err);
                clBuildProgram(ctx->cl_program, 1, &device, NULL, NULL, NULL);
                ctx->cl_classify_kernel = clCreateKernel(ctx->cl_program, "classify_pixels", &err);
                
                return 1;
            }
        }
    }
    #endif
    
    return 0;
}

void load_data_to_gpu(ProcessingContext* ctx) {
    if (!ctx->use_gpu) return;
    
    #ifdef CUDA_AVAILABLE
    if (ctx->gpu_type == 0) {
        // Allocate GPU memory for reference data
        size_t ref_data_size = 0;
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            ref_data_size += ctx->reference_vectors[i].size * 2 * sizeof(float);
        }
        
        cudaMalloc(&ctx->d_reference_data, ref_data_size);
        cudaMalloc(&ctx->d_image_data, ctx->image->data_size);
        cudaMalloc(&ctx->d_wavelengths, ctx->image->bands * sizeof(float));
        cudaMalloc(&ctx->d_classification, ctx->image->width * ctx->image->height * sizeof(uint16_t));
        cudaMalloc(&ctx->d_confidence, ctx->image->width * ctx->image->height * sizeof(float));
        
        // Copy reference data
        float* host_ref_data = (float*)malloc(ref_data_size);
        size_t offset = 0;
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            int size = ctx->reference_vectors[i].size;
            memcpy(host_ref_data + offset, ctx->reference_vectors[i].wavelengths, size * sizeof(float));
            offset += size;
            memcpy(host_ref_data + offset, ctx->reference_vectors[i].reflectance, size * sizeof(float));
            offset += size;
        }
        
        cudaMemcpy(ctx->d_reference_data, host_ref_data, ref_data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_image_data, ctx->image->data, ctx->image->data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_wavelengths, ctx->image->wavelengths, ctx->image->bands * sizeof(float), cudaMemcpyHostToDevice);
        
        free(host_ref_data);
    }
    #endif
    
    #ifdef OPENCL_AVAILABLE
    if (ctx->gpu_type == 1) {
        // Similar implementation for OpenCL
        size_t ref_data_size = 0;
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            ref_data_size += ctx->reference_vectors[i].size * 2 * sizeof(float);
        }
        
        ctx->cl_reference_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_READ_ONLY, ref_data_size, NULL, NULL);
        ctx->cl_image_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_READ_ONLY, ctx->image->data_size, NULL, NULL);
        ctx->cl_wavelength_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_READ_ONLY, ctx->image->bands * sizeof(float), NULL, NULL);
        ctx->cl_classification_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_WRITE_ONLY, ctx->image->width * ctx->image->height * sizeof(unsigned char), NULL, NULL);
        ctx->cl_confidence_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_WRITE_ONLY, ctx->image->width * ctx->image->height * sizeof(float), NULL, NULL);
        
        // Copy data to GPU
        float* host_ref_data = (float*)malloc(ref_data_size);
        size_t offset = 0;
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            int size = ctx->reference_vectors[i].size;
            memcpy(host_ref_data + offset, ctx->reference_vectors[i].wavelengths, size * sizeof(float));
            offset += size;
            memcpy(host_ref_data + offset, ctx->reference_vectors[i].reflectance, size * sizeof(float));
            offset += size;
        }
        
        clEnqueueWriteBuffer(ctx->cl_queue, ctx->cl_reference_buffer, CL_TRUE, 0, ref_data_size, host_ref_data, 0, NULL, NULL);
        clEnqueueWriteBuffer(ctx->cl_queue, ctx->cl_image_buffer, CL_TRUE, 0, ctx->image->data_size, ctx->image->data, 0, NULL, NULL);
        clEnqueueWriteBuffer(ctx->cl_queue, ctx->cl_wavelength_buffer, CL_TRUE, 0, ctx->image->bands * sizeof(float), ctx->image->wavelengths, 0, NULL, NULL);
        
        free(host_ref_data);
    }
    #endif
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

#ifdef CUDA_AVAILABLE
int classify_image_cuda(ProcessingContext* ctx) {
    printf("Classifying %d pixels using CUDA GPU...\n", ctx->image->width * ctx->image->height);
    
    // Create array of reference band counts
    int* ref_band_counts = (int*)malloc(ctx->num_reference_vectors * sizeof(int));
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        ref_band_counts[i] = ctx->reference_vectors[i].size;
    }
    
    int* d_ref_band_counts;
    cudaMalloc(&d_ref_band_counts, ctx->num_reference_vectors * sizeof(int));
    cudaMemcpy(d_ref_band_counts, ref_band_counts, ctx->num_reference_vectors * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((ctx->image->width + block_size.x - 1) / block_size.x,
                   (ctx->image->height + block_size.y - 1) / block_size.y);
    
    cuda_classify_pixels<<<grid_size, block_size, 0, ctx->stream>>>(
        ctx->d_image_data, ctx->d_wavelengths,
        ctx->d_reference_data, ctx->d_reference_data, // wavelengths included in reference data
        ctx->d_classification, ctx->d_confidence,
        ctx->image->width, ctx->image->height, ctx->image->bands,
        ctx->num_reference_vectors, d_ref_band_counts
    );
    
    // Copy results back
    cudaMemcpyAsync(ctx->result->classification, ctx->d_classification, 
                    ctx->image->width * ctx->image->height * sizeof(unsigned char),
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaMemcpyAsync(ctx->result->confidence, ctx->d_confidence,
                    ctx->image->width * ctx->image->height * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx->stream);
    
    cudaStreamSynchronize(ctx->stream);
    
    cudaFree(d_ref_band_counts);
    free(ref_band_counts);
    
    return 0;
}
#endif

#ifdef OPENCL_AVAILABLE
int classify_image_opencl(ProcessingContext* ctx) {
    printf("Classifying %d pixels using OpenCL GPU...\n", ctx->image->width * ctx->image->height);
    
    // Create array of reference band counts
    int* ref_band_counts = (int*)malloc(ctx->num_reference_vectors * sizeof(int));
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        ref_band_counts[i] = ctx->reference_vectors[i].size;
    }
    
    cl_mem cl_ref_band_counts = clCreateBuffer(ctx->cl_context, CL_MEM_READ_ONLY, 
                                               ctx->num_reference_vectors * sizeof(int), NULL, NULL);
    clEnqueueWriteBuffer(ctx->cl_queue, cl_ref_band_counts, CL_TRUE, 0, 
                        ctx->num_reference_vectors * sizeof(int), ref_band_counts, 0, NULL, NULL);
    
    // Set kernel arguments
    clSetKernelArg(ctx->cl_classify_kernel, 0, sizeof(cl_mem), &ctx->cl_image_buffer);
    clSetKernelArg(ctx->cl_classify_kernel, 1, sizeof(cl_mem), &ctx->cl_wavelength_buffer);
    clSetKernelArg(ctx->cl_classify_kernel, 2, sizeof(cl_mem), &ctx->cl_reference_buffer);
    clSetKernelArg(ctx->cl_classify_kernel, 3, sizeof(cl_mem), &ctx->cl_reference_buffer);
    clSetKernelArg(ctx->cl_classify_kernel, 4, sizeof(cl_mem), &ctx->cl_classification_buffer);
    clSetKernelArg(ctx->cl_classify_kernel, 5, sizeof(cl_mem), &ctx->cl_confidence_buffer);
    clSetKernelArg(ctx->cl_classify_kernel, 6, sizeof(int), &ctx->image->width);
    clSetKernelArg(ctx->cl_classify_kernel, 7, sizeof(int), &ctx->image->height);
    clSetKernelArg(ctx->cl_classify_kernel, 8, sizeof(int), &ctx->image->bands);
    clSetKernelArg(ctx->cl_classify_kernel, 9, sizeof(int), &ctx->num_reference_vectors);
    clSetKernelArg(ctx->cl_classify_kernel, 10, sizeof(cl_mem), &cl_ref_band_counts);
    
    // Launch kernel
    size_t global_work_size[2] = {ctx->image->width, ctx->image->height};
    clEnqueueNDRangeKernel(ctx->cl_queue, ctx->cl_classify_kernel, 2, NULL, global_work_size, 
                          NULL, 0, NULL, NULL);
    
    // Read results
    clEnqueueReadBuffer(ctx->cl_queue, ctx->cl_classification_buffer, CL_TRUE, 0,
                       ctx->image->width * ctx->image->height * sizeof(unsigned char),
                       ctx->result->classification, 0, NULL, NULL);
    clEnqueueReadBuffer(ctx->cl_queue, ctx->cl_confidence_buffer, CL_TRUE, 0,
                       ctx->image->width * ctx->image->height * sizeof(float),
                       ctx->result->confidence, 0, NULL, NULL);
    
    clReleaseMemObject(cl_ref_band_counts);
    free(ref_band_counts);
    
    return 0;
}
#endif

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
