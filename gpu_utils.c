#include "common_types.h"
#include <string.h>  // For memcpy warning

#ifdef OPENCL_AVAILABLE
#include "opencl_kernels.h"
#endif

int detect_gpu_capabilities(ProcessingContext* ctx) {
    #ifdef CUDA_AVAILABLE
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    
    if (cuda_status == cudaSuccess && device_count > 0) {
        struct cudaDeviceProp prop;  // Add 'struct' keyword
        cudaGetDeviceProperties(&prop, 0);
        ctx->available_gpu_memory = prop.totalGlobalMem;
        
        if (ctx->available_gpu_memory > GPU_MEMORY_THRESHOLD) {
            ctx->use_gpu = 1;
            ctx->gpu_type = 0;
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
        
        cudaMalloc((void**)&ctx->d_reference_data, ref_data_size);
        cudaMalloc((void**)&ctx->d_image_data, ctx->image->data_size);
        cudaMalloc((void**)&ctx->d_wavelengths, ctx->image->bands * sizeof(float));
        cudaMalloc((void**)&ctx->d_classification, ctx->image->width * ctx->image->height * sizeof(uint16_t));
        cudaMalloc((void**)&ctx->d_confidence, ctx->image->width * ctx->image->height * sizeof(float));
        
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

#ifdef CUDA_AVAILABLE
#include "cuda_kernels.h"

int classify_image_cuda(ProcessingContext* ctx) {
    printf("Classifying %d pixels using CUDA GPU...\n", ctx->image->width * ctx->image->height);
    
    int* ref_band_counts = (int*)malloc(ctx->num_reference_vectors * sizeof(int));
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        ref_band_counts[i] = ctx->reference_vectors[i].size;
    }
    
    int* d_ref_band_counts;
    cudaMalloc((void**)&d_ref_band_counts, ctx->num_reference_vectors * sizeof(int));
    cudaMemcpy(d_ref_band_counts, ref_band_counts, ctx->num_reference_vectors * sizeof(int), cudaMemcpyHostToDevice);
    
    // Call wrapper function
    launch_cuda_classify_pixels(
        ctx->d_image_data, ctx->d_wavelengths,
        ctx->d_reference_data, ctx->d_reference_data,
        ctx->d_classification, ctx->d_confidence,
        ctx->image->width, ctx->image->height, ctx->image->bands,
        ctx->num_reference_vectors, d_ref_band_counts,
        ctx->stream
    );
    
    // Copy results back
    cudaMemcpyAsync(ctx->result->classification, ctx->d_classification, 
                    ctx->image->width * ctx->image->height * sizeof(uint16_t),
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
    // Check if Fourier mode is requested but not available in OpenCL yet
    if (ctx->classification_mode != MODE_SPATIAL_CPU) {
        printf("Warning: OpenCL currently only supports spatial domain classification\n");
        printf("Falling back to CPU Fourier classification...\n");
        
        // Call appropriate CPU Fourier function based on mode
        switch (ctx->classification_mode) {
            case 1: // MODE_FOURIER_CPU
                return classify_image_fourier_cpu(ctx);
            case 2: // MODE_FOURIER_CPU_COHERENCE_QUALITY
                return classify_image_fourier_cpu_optimized(ctx);
            case 3: // MODE_FOURIER_CPU_COHERENCE_FASTEST
                return classify_image_fourier_cpu_light(ctx);
            default:
                break;
        }
    }
    
    printf("Classifying %d pixels using OpenCL GPU (spatial domain)...\n", 
           ctx->image->width * ctx->image->height);
    
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
