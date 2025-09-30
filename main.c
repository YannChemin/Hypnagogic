#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "reference_materials.h"

// Conditional includes based on available GPU support
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef OPENCL_AVAILABLE
#include <CL/cl.h>
#endif

#define MAX_VECTOR_SIZE 4096
#define SIMILARITY_THRESHOLD 0.8
#define GPU_MEMORY_THRESHOLD (1024 * 1024 * 1024) // 1GB

// Processing context
typedef struct {
    HyperspectralVector* reference_vectors;
    int num_reference_vectors;
    int use_gpu;
    int gpu_type; // 0: CUDA, 1: OpenCL
    size_t available_gpu_memory;
    
    #ifdef CUDA_AVAILABLE
    float* d_reference_data;
    float* d_input_data;
    cudaStream_t stream;
    #endif
    
    #ifdef OPENCL_AVAILABLE
    cl_context cl_context;
    cl_command_queue cl_queue;
    cl_program cl_program;
    cl_kernel cl_kernel;
    cl_mem cl_reference_buffer;
    cl_mem cl_input_buffer;
    cl_mem cl_result_buffer;
    #endif
} ProcessingContext;

// Function prototypes
void initialize_context(ProcessingContext* ctx);
void cleanup_context(ProcessingContext* ctx);
int load_reference_vectors_from_header(ProcessingContext* ctx);
HyperspectralVector* create_hyperspectral_vector(int size, MaterialType material, const char* name);
void destroy_hyperspectral_vector(HyperspectralVector* vec);
int detect_gpu_capabilities(ProcessingContext* ctx);
void load_data_to_gpu(ProcessingContext* ctx);
float* match_vectors_cpu(ProcessingContext* ctx, HyperspectralVector* input, float* similarities);
float* match_vectors_cuda(ProcessingContext* ctx, HyperspectralVector* input, float* similarities);
float* match_vectors_opencl(ProcessingContext* ctx, HyperspectralVector* input, float* similarities);
float calculate_similarity(float* vec1_wav, float* vec1_ref, float* vec2_wav, float* vec2_ref, int size1, int size2);
void interpolate_vector(float* src_wav, float* src_ref, int src_size, 
                       float* dst_wav, float* dst_ref, int dst_size,
                       float* interp_ref);

// OpenCL kernel source for similarity calculation
#ifdef OPENCL_AVAILABLE
const char* opencl_kernel_source = 
"__kernel void calculate_similarities(__global float* reference_data,\n"
"                                   __global float* input_data,\n"
"                                   __global float* results,\n"
"                                   int vector_size,\n"
"                                   int num_references) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_references) return;\n"
"    \n"
"    float dot_product = 0.0f;\n"
"    float norm_ref = 0.0f;\n"
"    float norm_input = 0.0f;\n"
"    \n"
"    int ref_offset = gid * vector_size * 2;\n"
"    \n"
"    for (int i = 0; i < vector_size; i++) {\n"
"        float ref_val = reference_data[ref_offset + vector_size + i];\n"
"        float input_val = input_data[vector_size + i];\n"
"        \n"
"        dot_product += ref_val * input_val;\n"
"        norm_ref += ref_val * ref_val;\n"
"        norm_input += input_val * input_val;\n"
"    }\n"
"    \n"
"    results[gid] = dot_product / (sqrt(norm_ref) * sqrt(norm_input));\n"
"}\n";
#endif

// CUDA kernel for similarity calculation
#ifdef CUDA_AVAILABLE
__global__ void cuda_calculate_similarities(float* reference_data, float* input_data,
                                          float* results, int vector_size, int num_references) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_references) return;
    
    float dot_product = 0.0f;
    float norm_ref = 0.0f;
    float norm_input = 0.0f;
    
    int ref_offset = idx * vector_size * 2;
    
    for (int i = 0; i < vector_size; i++) {
        float ref_val = reference_data[ref_offset + vector_size + i];
        float input_val = input_data[vector_size + i];
        
        dot_product += ref_val * input_val;
        norm_ref += ref_val * ref_val;
        norm_input += input_val * input_val;
    }
    
    results[idx] = dot_product / (sqrtf(norm_ref) * sqrtf(norm_input));
}
#endif

int main() {
    ProcessingContext ctx;
    initialize_context(&ctx);
    
    printf("Hyperspectral Vector Processing System V3\n");
    printf("==========================================\n");
    
    // Load reference vectors from header file - NO MORE SYNTHETIC DATA
    printf("Loading reference materials from header file...\n");
    if (load_reference_vectors_from_header(&ctx) != 0) {
        printf("Error: Failed to load reference vectors from header file\n");
        cleanup_context(&ctx);
        return -1;
    }
    
    // Detect GPU capabilities
    int gpu_detected = detect_gpu_capabilities(&ctx);
    
    if (gpu_detected) {
        printf("GPU detected: %s\n", ctx.gpu_type == 0 ? "CUDA" : "OpenCL");
        printf("Available GPU memory: %.2f GB\n", ctx.available_gpu_memory / (1024.0 * 1024.0 * 1024.0));
        load_data_to_gpu(&ctx);
    } else {
        printf("Using CPU processing with OpenMP\n");
        printf("Number of CPU cores: %d\n", omp_get_max_threads());
    }
    
    // Create a test input vector (simulating received data with vegetation-like characteristics)
    HyperspectralVector* input = create_hyperspectral_vector(200, VEGETATION, "received_input");
    
    if (!input) {
        printf("Error: Failed to create test input vector\n");
        cleanup_context(&ctx);
        return -1;
    }
    
    // Generate test input data that resembles vegetation spectrum for demonstration
    printf("\nGenerating test input vector (simulating received hyperspectral data)...\n");
    srand(time(NULL));
    for (int i = 0; i < input->size; i++) {
        input->wavelengths[i] = 400.0f + i * 2.0f; // 400-798nm range
        // Simulate vegetation-like spectrum with NIR plateau for testing
        if (input->wavelengths[i] < 680) {
            input->reflectance[i] = 0.05f + 0.05f * sinf(i * 0.1f) + (rand() % 100) / 2000.0f;
        } else {
            input->reflectance[i] = 0.4f + 0.3f * (1.0f + sinf(i * 0.05f)) + (rand() % 100) / 2000.0f;
        }
        // Ensure reflectance stays in valid range
        if (input->reflectance[i] > 1.0f) input->reflectance[i] = 1.0f;
        if (input->reflectance[i] < 0.0f) input->reflectance[i] = 0.0f;
    }
    
    printf("Input vector: %d spectral bands (%.1f - %.1f nm)\n", 
           input->size, input->wavelengths[0], input->wavelengths[input->size-1]);
    
    // Perform similarity matching using REAL reference data from header file
    printf("\nMatching against %d reference materials from header file:\n", ctx.num_reference_vectors);
    float* similarities = (float*)malloc(ctx.num_reference_vectors * sizeof(float));
    if (!similarities) {
        printf("Error: Failed to allocate similarity array\n");
        destroy_hyperspectral_vector(input);
        cleanup_context(&ctx);
        return -1;
    }
    
    clock_t start = clock();
    
    if (ctx.use_gpu && ctx.gpu_type == 0) {
        #ifdef CUDA_AVAILABLE
        match_vectors_cuda(&ctx, input, similarities);
        #endif
    } else if (ctx.use_gpu && ctx.gpu_type == 1) {
        #ifdef OPENCL_AVAILABLE
        match_vectors_opencl(&ctx, input, similarities);
        #endif
    } else {
        match_vectors_cpu(&ctx, input, similarities);
    }
    
    clock_t end = clock();
    double processing_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("\nProcessing completed in %.4f seconds\n", processing_time);
    printf("\nSimilarity Results (using real reference materials):\n");
    printf("===================================================\n");
    
    for (int i = 0; i < ctx.num_reference_vectors; i++) {
        printf("%-12s: %.4f %s\n", 
               ctx.reference_vectors[i].name, 
               similarities[i],
               similarities[i] > SIMILARITY_THRESHOLD ? "(MATCH)" : "");
    }
    
    // Find best match
    int best_match = 0;
    float best_similarity = similarities[0];
    for (int i = 1; i < ctx.num_reference_vectors; i++) {
        if (similarities[i] > best_similarity) {
            best_similarity = similarities[i];
            best_match = i;
        }
    }
    
    printf("\nBest match: %s (similarity: %.4f)\n", 
           ctx.reference_vectors[best_match].name, best_similarity);
    printf("Reference spectrum: %d bands (%.1f - %.1f nm)\n",
           ctx.reference_vectors[best_match].size,
           ctx.reference_vectors[best_match].wavelengths[0],
           ctx.reference_vectors[best_match].wavelengths[ctx.reference_vectors[best_match].size-1]);
    
    // Cleanup
    free(similarities);
    destroy_hyperspectral_vector(input);
    cleanup_context(&ctx);
    
    return 0;
}

void initialize_context(ProcessingContext* ctx) {
    memset(ctx, 0, sizeof(ProcessingContext));
    ctx->num_reference_vectors = 0;
    ctx->reference_vectors = NULL;  // Initialize to NULL for dynamic allocation
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
    
    #ifdef CUDA_AVAILABLE
    if (ctx->use_gpu && ctx->gpu_type == 0) {
        if (ctx->d_reference_data) cudaFree(ctx->d_reference_data);
        if (ctx->d_input_data) cudaFree(ctx->d_input_data);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
    }
    #endif
    
    #ifdef OPENCL_AVAILABLE
    if (ctx->use_gpu && ctx->gpu_type == 1) {
        if (ctx->cl_reference_buffer) clReleaseMemObject(ctx->cl_reference_buffer);
        if (ctx->cl_input_buffer) clReleaseMemObject(ctx->cl_input_buffer);
        if (ctx->cl_result_buffer) clReleaseMemObject(ctx->cl_result_buffer);
        if (ctx->cl_kernel) clReleaseKernel(ctx->cl_kernel);
        if (ctx->cl_program) clReleaseProgram(ctx->cl_program);
        if (ctx->cl_queue) clReleaseCommandQueue(ctx->cl_queue);
        if (ctx->cl_context) clReleaseContext(ctx->cl_context);
    }
    #endif
}

// MAIN FUNCTION: Load reference vectors from header file (NO synthetic generation)
int load_reference_vectors_from_header(ProcessingContext* ctx) {
    // Get the number of reference materials from the header
    ctx->num_reference_vectors = NUM_REFERENCE_MATERIALS;
    
    if (ctx->num_reference_vectors == 0) {
        printf("Error: No reference materials found in header file\n");
        return -1;
    }
    
    printf("Found %d reference materials in header file\n", ctx->num_reference_vectors);
    
    // Allocate memory for reference vectors
    ctx->reference_vectors = (HyperspectralVector*)malloc(
        ctx->num_reference_vectors * sizeof(HyperspectralVector));
    
    if (!ctx->reference_vectors) {
        printf("Error: Failed to allocate memory for reference vectors\n");
        return -1;
    }
    
    // Load each reference material from the header file arrays
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        const ReferenceMaterialData* mat_data = &reference_materials[i];
        HyperspectralVector* vec = &ctx->reference_vectors[i];
        
        vec->size = mat_data->size;
        vec->material = mat_data->material;
        strncpy(vec->name, mat_data->name, sizeof(vec->name) - 1);
        vec->name[sizeof(vec->name) - 1] = '\0'; // Ensure null termination
        
        // Validate data from header
        if (vec->size <= 0 || vec->size > MAX_VECTOR_SIZE) {
            printf("Error: Invalid size %d for material %s\n", vec->size, vec->name);
            return -1;
        }
        
        if (!mat_data->wavelengths || !mat_data->reflectance) {
            printf("Error: NULL data pointers for material %s\n", vec->name);
            return -1;
        }
        
        // Allocate memory for wavelengths and reflectance
        vec->wavelengths = (float*)malloc(vec->size * sizeof(float));
        vec->reflectance = (float*)malloc(vec->size * sizeof(float));
        
        if (!vec->wavelengths || !vec->reflectance) {
            printf("Error: Failed to allocate memory for vector %s\n", vec->name);
            return -1;
        }
        
        // Copy REAL data from header file (const arrays) - NO SYNTHETIC GENERATION
        for (int j = 0; j < vec->size; j++) {
            vec->wavelengths[j] = mat_data->wavelengths[j];
            vec->reflectance[j] = mat_data->reflectance[j];
        }
        
        // Validate wavelength ordering
        for (int j = 1; j < vec->size; j++) {
            if (vec->wavelengths[j] <= vec->wavelengths[j-1]) {
                printf("Warning: Non-monotonic wavelengths in %s at index %d (%.1f -> %.1f)\n", 
                       vec->name, j, vec->wavelengths[j-1], vec->wavelengths[j]);
            }
        }
        
        printf("Loaded %s: %d bands (%.1f-%.1f nm) - REAL DATA FROM HEADER\n", 
               vec->name, vec->size, vec->wavelengths[0], vec->wavelengths[vec->size-1]);
    }
    
    printf("Successfully loaded %d REAL reference vectors from header file\n", ctx->num_reference_vectors);
    return 0;
}

HyperspectralVector* create_hyperspectral_vector(int size, MaterialType material, const char* name) {
    if (size <= 0 || size > MAX_VECTOR_SIZE) {
        printf("Error: Invalid vector size %d\n", size);
        return NULL;
    }
    
    HyperspectralVector* vec = (HyperspectralVector*)malloc(sizeof(HyperspectralVector));
    if (!vec) {
        printf("Error: Failed to allocate memory for hyperspectral vector\n");
        return NULL;
    }
    
    vec->size = size;
    vec->material = material;
    strncpy(vec->name, name, sizeof(vec->name) - 1);
    vec->name[sizeof(vec->name) - 1] = '\0';  // Ensure null termination
    
    vec->wavelengths = (float*)malloc(size * sizeof(float));
    vec->reflectance = (float*)malloc(size * sizeof(float));
    
    if (!vec->wavelengths || !vec->reflectance) {
        printf("Error: Failed to allocate memory for vector data\n");
        destroy_hyperspectral_vector(vec);
        return NULL;
    }
    
    // Initialize to zero
    memset(vec->wavelengths, 0, size * sizeof(float));
    memset(vec->reflectance, 0, size * sizeof(float));
    
    return vec;
}

void destroy_hyperspectral_vector(HyperspectralVector* vec) {
    if (vec) {
        if (vec->wavelengths) free(vec->wavelengths);
        if (vec->reflectance) free(vec->reflectance);
        free(vec);
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
                
                // Initialize OpenCL context
                ctx->cl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
                ctx->cl_queue = clCreateCommandQueue(ctx->cl_context, device, 0, &err);
                
                // Compile kernel
                ctx->cl_program = clCreateProgramWithSource(ctx->cl_context, 1, 
                                                           &opencl_kernel_source, NULL, &err);
                clBuildProgram(ctx->cl_program, 1, &device, NULL, NULL, NULL);
                ctx->cl_kernel = clCreateKernel(ctx->cl_program, "calculate_similarities", &err);
                
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
        size_t total_size = 0;
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            total_size += ctx->reference_vectors[i].size * 2 * sizeof(float);
        }
        
        cudaMalloc(&ctx->d_reference_data, total_size);
        cudaMalloc(&ctx->d_input_data, MAX_VECTOR_SIZE * 2 * sizeof(float));
        
        // Copy reference data to GPU
        float* host_data = (float*)malloc(total_size);
        size_t offset = 0;
        
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            int size = ctx->reference_vectors[i].size;
            memcpy(host_data + offset, ctx->reference_vectors[i].wavelengths, size * sizeof(float));
            offset += size;
            memcpy(host_data + offset, ctx->reference_vectors[i].reflectance, size * sizeof(float));
            offset += size;
        }
        
        cudaMemcpy(ctx->d_reference_data, host_data, total_size, cudaMemcpyHostToDevice);
        free(host_data);
    }
    #endif
    
    #ifdef OPENCL_AVAILABLE
    if (ctx->gpu_type == 1) {
        size_t total_size = 0;
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            total_size += ctx->reference_vectors[i].size * 2 * sizeof(float);
        }
        
        ctx->cl_reference_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_READ_ONLY, 
                                                 total_size, NULL, NULL);
        ctx->cl_input_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_READ_ONLY, 
                                             MAX_VECTOR_SIZE * 2 * sizeof(float), NULL, NULL);
        ctx->cl_result_buffer = clCreateBuffer(ctx->cl_context, CL_MEM_WRITE_ONLY, 
                                              ctx->num_reference_vectors * sizeof(float), NULL, NULL);
        
        // Copy reference data to GPU
        float* host_data = (float*)malloc(total_size);
        size_t offset = 0;
        
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            int size = ctx->reference_vectors[i].size;
            memcpy(host_data + offset, ctx->reference_vectors[i].wavelengths, size * sizeof(float));
            offset += size;
            memcpy(host_data + offset, ctx->reference_vectors[i].reflectance, size * sizeof(float));
            offset += size;
        }
        
        clEnqueueWriteBuffer(ctx->cl_queue, ctx->cl_reference_buffer, CL_TRUE, 0, 
                            total_size, host_data, 0, NULL, NULL);
        free(host_data);
    }
    #endif
}

float* match_vectors_cpu(ProcessingContext* ctx, HyperspectralVector* input, float* similarities) {
    printf("Starting CPU similarity matching with %d REAL reference vectors...\n", ctx->num_reference_vectors);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        similarities[i] = calculate_similarity(
            ctx->reference_vectors[i].wavelengths,
            ctx->reference_vectors[i].reflectance,
            input->wavelengths,
            input->reflectance,
            ctx->reference_vectors[i].size,
            input->size
        );
    }
    
    return similarities;
}

#ifdef CUDA_AVAILABLE
float* match_vectors_cuda(ProcessingContext* ctx, HyperspectralVector* input, float* similarities) {
    // Copy input data to GPU
    float* input_data = (float*)malloc(input->size * 2 * sizeof(float));
    memcpy(input_data, input->wavelengths, input->size * sizeof(float));
    memcpy(input_data + input->size, input->reflectance, input->size * sizeof(float));
    
    cudaMemcpyAsync(ctx->d_input_data, input_data, input->size * 2 * sizeof(float), 
                    cudaMemcpyHostToDevice, ctx->stream);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (ctx->num_reference_vectors + block_size - 1) / block_size;
    
    float* d_results;
    cudaMalloc(&d_results, ctx->num_reference_vectors * sizeof(float));
    
    cuda_calculate_similarities<<<grid_size, block_size, 0, ctx->stream>>>(
        ctx->d_reference_data, ctx->d_input_data, d_results, 
        input->size, ctx->num_reference_vectors);
    
    // Copy results back
    cudaMemcpyAsync(similarities, d_results, ctx->num_reference_vectors * sizeof(float), 
                    cudaMemcpyDeviceToHost, ctx->stream);
    
    cudaStreamSynchronize(ctx->stream);
    
    cudaFree(d_results);
    free(input_data);
    
    return similarities;
}
#endif

#ifdef OPENCL_AVAILABLE
float* match_vectors_opencl(ProcessingContext* ctx, HyperspectralVector* input, float* similarities) {
    // Copy input data to GPU
    float* input_data = (float*)malloc(input->size * 2 * sizeof(float));
    memcpy(input_data, input->wavelengths, input->size * sizeof(float));
    memcpy(input_data + input->size, input->reflectance, input->size * sizeof(float));
    
    clEnqueueWriteBuffer(ctx->cl_queue, ctx->cl_input_buffer, CL_TRUE, 0, 
                        input->size * 2 * sizeof(float), input_data, 0, NULL, NULL);
    
    // Set kernel arguments
    clSetKernelArg(ctx->cl_kernel, 0, sizeof(cl_mem), &ctx->cl_reference_buffer);
    clSetKernelArg(ctx->cl_kernel, 1, sizeof(cl_mem), &ctx->cl_input_buffer);
    clSetKernelArg(ctx->cl_kernel, 2, sizeof(cl_mem), &ctx->cl_result_buffer);
    clSetKernelArg(ctx->cl_kernel, 3, sizeof(int), &input->size);
    clSetKernelArg(ctx->cl_kernel, 4, sizeof(int), &ctx->num_reference_vectors);
    
    // Launch kernel
    size_t global_work_size = ctx->num_reference_vectors;
    clEnqueueNDRangeKernel(ctx->cl_queue, ctx->cl_kernel, 1, NULL, &global_work_size, 
                          NULL, 0, NULL, NULL);
    
    // Read results
    clEnqueueReadBuffer(ctx->cl_queue, ctx->cl_result_buffer, CL_TRUE, 0, 
                       ctx->num_reference_vectors * sizeof(float), similarities, 0, NULL, NULL);
    
    free(input_data);
    return similarities;
}
#endif

float calculate_similarity(float* vec1_wav, float* vec1_ref, float* vec2_wav, float* vec2_ref, 
                          int size1, int size2) {
    // Validate inputs
    if (!vec1_wav || !vec1_ref || !vec2_wav || !vec2_ref || size1 <= 0 || size2 <= 0) {
        printf("Error: Invalid input to calculate_similarity\n");
        return 0.0f;
    }
    
    // For different sized vectors, interpolate to common wavelength grid
    int common_size = (size1 < size2) ? size1 : size2;
    if (common_size > MAX_VECTOR_SIZE / 2) {
        common_size = MAX_VECTOR_SIZE / 2;  // Safety limit
    }
    
    float* interp_ref1 = (float*)malloc(common_size * sizeof(float));
    float* interp_ref2 = (float*)malloc(common_size * sizeof(float));
    float* common_wav = (float*)malloc(common_size * sizeof(float));
    
    if (!interp_ref1 || !interp_ref2 || !common_wav) {
        printf("Error: Memory allocation failed in calculate_similarity\n");
        if (interp_ref1) free(interp_ref1);
        if (interp_ref2) free(interp_ref2);
        if (common_wav) free(common_wav);
        return 0.0f;
    }
    
    // Create common wavelength grid based on overlap
    float min_wav = fmaxf(vec1_wav[0], vec2_wav[0]);
    float max_wav = fminf(vec1_wav[size1-1], vec2_wav[size2-1]);
    
    // Check for valid overlap
    if (min_wav >= max_wav) {
        printf("Warning: No spectral overlap between vectors\n");
        free(interp_ref1);
        free(interp_ref2);
        free(common_wav);
        return 0.0f;
    }
    
    float step = (max_wav - min_wav) / (common_size - 1);
    
    for (int i = 0; i < common_size; i++) {
        common_wav[i] = min_wav + i * step;
    }
    
    // Interpolate both vectors to common grid
    interpolate_vector(vec1_wav, vec1_ref, size1, common_wav, interp_ref1, common_size, interp_ref1);
    interpolate_vector(vec2_wav, vec2_ref, size2, common_wav, interp_ref2, common_size, interp_ref2);
    
    // Calculate cosine similarity
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (int i = 0; i < common_size; i++) {
        dot_product += interp_ref1[i] * interp_ref2[i];
        norm1 += interp_ref1[i] * interp_ref1[i];
        norm2 += interp_ref2[i] * interp_ref2[i];
    }
    
    float similarity = 0.0f;
    if (norm1 > 0.0f && norm2 > 0.0f) {
        similarity = dot_product / (sqrtf(norm1) * sqrtf(norm2));
    }
    
    free(interp_ref1);
    free(interp_ref2);
    free(common_wav);
    
    return similarity;
}

void interpolate_vector(float* src_wav, float* src_ref, int src_size, 
                       float* dst_wav, float* dst_ref, int dst_size,
                       float* interp_ref) {
    // Validate inputs
    if (!src_wav || !src_ref || !dst_wav || !interp_ref || src_size <= 0 || dst_size <= 0) {
        printf("Error: Invalid input to interpolate_vector\n");
        return;
    }
    
    for (int i = 0; i < dst_size; i++) {
        float target_wav = dst_wav[i];
        
        // Handle boundary cases
        if (target_wav <= src_wav[0]) {
            interp_ref[i] = src_ref[0];
            continue;
        }
        if (target_wav >= src_wav[src_size-1]) {
            interp_ref[i] = src_ref[src_size-1];
            continue;
        }
        
        // Find surrounding points for interpolation
        int left = 0, right = src_size - 1;
        
        for (int j = 0; j < src_size - 1; j++) {
            if (src_wav[j] <= target_wav && src_wav[j+1] >= target_wav) {
                left = j;
                right = j + 1;
                break;
            }
        }
        
        // Linear interpolation
        if (left == right || fabsf(src_wav[right] - src_wav[left]) < 1e-6f) {
            interp_ref[i] = src_ref[left];
        } else {
            float t = (target_wav - src_wav[left]) / (src_wav[right] - src_wav[left]);
            // Clamp t to [0, 1] for safety
            t = fmaxf(0.0f, fminf(1.0f, t));
            interp_ref[i] = src_ref[left] + t * (src_ref[right] - src_ref[left]);
        }
    }
}