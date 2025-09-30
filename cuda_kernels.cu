#include <cuda_runtime.h>
#include <stdint.h>

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
        
        float dot_product = 0.0f;
        float norm_pixel = 0.0f;
        float norm_ref = 0.0f;
        
        for (int i = 0; i < image_bands; i++) {
            float pixel_val = image_data[pixel_offset + i];
            float img_wav = image_wavelengths[i];
            
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

// Wrapper function callable from C
extern "C" void launch_cuda_classify_pixels(
    float* d_image_data, float* d_image_wavelengths,
    float* d_reference_data, float* d_reference_wavelengths,
    uint16_t* d_classification, float* d_confidence,
    int width, int height, int image_bands,
    int num_references, int* d_ref_band_counts,
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    cuda_classify_pixels<<<grid_size, block_size, 0, stream>>>(
        d_image_data, d_image_wavelengths,
        d_reference_data, d_reference_wavelengths,
        d_classification, d_confidence,
        width, height, image_bands,
        num_references, d_ref_band_counts
    );
}
