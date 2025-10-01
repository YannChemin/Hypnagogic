#include "fourier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/stat.h>

// Platform-specific includes for directory creation
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/types.h>
#include <unistd.h>
#endif

int next_power_of_2(int n) {
    int power = 1;
    while (power < n) power *= 2;
    return power;
}

char* generate_fourier_cache_filename(int fft_size, int num_materials) {
    char* filename = (char*)malloc(512);
    if (!filename) return NULL;
    
    // Create cache directory if it doesn't exist
    struct stat st = {0};
    if (stat(FOURIER_CACHE_DIR, &st) == -1) {
        mkdir(FOURIER_CACHE_DIR, 0755);
    }
    
    snprintf(filename, 512, "%s/fourier_fft%d_mat%d_v%d.cache",
             FOURIER_CACHE_DIR, fft_size, num_materials, FOURIER_CACHE_VERSION);
    return filename;
}

int save_fourier_cache(ProcessingContext* ctx, const char* cache_file) {
    printf("Saving Fourier cache to: %s\n", cache_file);
    
    FILE* f = fopen(cache_file, "wb");
    if (!f) {
        printf("Warning: Cannot create cache file\n");
        return -1;
    }
    
    // Write header
    uint32_t version = FOURIER_CACHE_VERSION;
    uint32_t fft_size = ctx->fft_size;
    uint32_t num_materials = ctx->num_reference_vectors;
    
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&fft_size, sizeof(uint32_t), 1, f);
    fwrite(&num_materials, sizeof(uint32_t), 1, f);
    
    // Write each reference material's Fourier data
    int fft_complex_size = fft_size / 2 + 1;
    
    for (int i = 0; i < num_materials; i++) {
        // Write material ID and name
        int material_id = ctx->reference_vectors[i].material;
        fwrite(&material_id, sizeof(int), 1, f);
        fwrite(ctx->reference_vectors[i].name, sizeof(char), 64, f);
        
        // Write original spectrum info
        int original_size = ctx->reference_vectors[i].size;
        fwrite(&original_size, sizeof(int), 1, f);
        fwrite(ctx->reference_vectors[i].wavelengths, sizeof(float), original_size, f);
        fwrite(ctx->reference_vectors[i].reflectance, sizeof(float), original_size, f);
        
        // Write Fourier data
        fwrite(ctx->reference_fft[i], sizeof(float complex), fft_complex_size, f);
        fwrite(ctx->reference_fft_mag[i], sizeof(float), fft_complex_size, f);
        fwrite(&ctx->reference_energies[i], sizeof(float), 1, f);
        
        if (i % 10000 == 0 && i > 0) {
            printf("  Saved %d/%d materials to cache\n", i, num_materials);
        }
    }
    
    fclose(f);
    printf("Fourier cache saved successfully (%d materials, FFT size %d)\n", 
           num_materials, fft_size);
    return 0;
}

int load_fourier_cache(ProcessingContext* ctx, const char* cache_file) {
    FILE* f = fopen(cache_file, "rb");
    if (!f) {
        return -1;  // Cache doesn't exist
    }
    
    printf("Loading Fourier cache from: %s\n", cache_file);
    
    // Read and validate header
    uint32_t version, fft_size, num_materials;
    
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&fft_size, sizeof(uint32_t), 1, f) != 1 ||
        fread(&num_materials, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    
    // Validate parameters
    if (version != FOURIER_CACHE_VERSION) {
        printf("Cache version mismatch (expected %d, got %u)\n", 
               FOURIER_CACHE_VERSION, version);
        fclose(f);
        return -1;
    }
    
    if (fft_size != ctx->fft_size) {
        printf("FFT size mismatch (expected %d, got %u)\n", ctx->fft_size, fft_size);
        fclose(f);
        return -1;
    }
    
    if (num_materials != ctx->num_reference_vectors) {
        printf("Material count mismatch (expected %d, got %u)\n", 
               ctx->num_reference_vectors, num_materials);
        fclose(f);
        return -1;
    }
    
    printf("Cache validation passed. Loading %u materials...\n", num_materials);
    
    // Load each reference material's Fourier data
    int fft_complex_size = fft_size / 2 + 1;
    
    for (int i = 0; i < num_materials; i++) {
        // Read material ID and name (for validation)
        int material_id;
        char name[64];
        fread(&material_id, sizeof(int), 1, f);
        fread(name, sizeof(char), 64, f);
        
        // Read original spectrum info (skip, we already have it)
        int original_size;
        fread(&original_size, sizeof(int), 1, f);
        fseek(f, original_size * sizeof(float) * 2, SEEK_CUR);  // Skip wavelengths and reflectance
        
        // Read Fourier data
        fread(ctx->reference_fft[i], sizeof(float complex), fft_complex_size, f);
        fread(ctx->reference_fft_mag[i], sizeof(float), fft_complex_size, f);
        fread(&ctx->reference_energies[i], sizeof(float), 1, f);
        
        if (i % 10000 == 0 && i > 0) {
            printf("  Loaded %d/%d materials from cache\n", i, num_materials);
        }
    }
    
    fclose(f);
    printf("Successfully loaded %d materials from Fourier cache\n", num_materials);
    return 0;
}

int setup_fourier_processing(ProcessingContext* ctx) {
    printf("\n=== Setting up Fourier Processing ===\n");
    
    // Determine FFT size (next power of 2)
    ctx->fft_size = next_power_of_2(ctx->image->bands);
    printf("Image bands: %d -> FFT size: %d (%.1f%% padding)\n",
           ctx->image->bands, ctx->fft_size,
           100.0 * (ctx->fft_size - ctx->image->bands) / ctx->image->bands);
    
    // Allocate reference FFT storage
    int fft_complex_size = ctx->fft_size / 2 + 1;
    
    ctx->reference_fft = (float complex**)malloc(ctx->num_reference_vectors * sizeof(float complex*));
    ctx->reference_fft_mag = (float**)malloc(ctx->num_reference_vectors * sizeof(float*));
    ctx->reference_energies = (float*)malloc(ctx->num_reference_vectors * sizeof(float));
    
    if (!ctx->reference_fft || !ctx->reference_fft_mag || !ctx->reference_energies) {
        printf("Error: Failed to allocate reference FFT storage\n");
        return -1;
    }
    
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        ctx->reference_fft[i] = fftwf_alloc_complex(fft_complex_size);
        ctx->reference_fft_mag[i] = (float*)malloc(fft_complex_size * sizeof(float));
        
        if (!ctx->reference_fft[i] || !ctx->reference_fft_mag[i]) {
            printf("Error: Failed to allocate FFT arrays for material %d\n", i);
            return -1;
        }
    }
    
    // Setup thread-local FFT resources
    ctx->num_threads = omp_get_max_threads();
    printf("Initializing %d thread-local FFT plans\n", ctx->num_threads);
    
    ctx->thread_fft_plans = (fftwf_plan*)malloc(ctx->num_threads * sizeof(fftwf_plan));
    ctx->thread_fft_input = (float**)malloc(ctx->num_threads * sizeof(float*));
    ctx->thread_fft_output = (float complex**)malloc(ctx->num_threads * sizeof(float complex*));
    ctx->thread_fft_mag = (float**)malloc(ctx->num_threads * sizeof(float*));
    
    if (!ctx->thread_fft_plans || !ctx->thread_fft_input || 
        !ctx->thread_fft_output || !ctx->thread_fft_mag) {
        printf("Error: Failed to allocate thread-local FFT resources\n");
        return -1;
    }
    
    // Initialize FFTW with threading support
    fftwf_init_threads();
    fftwf_plan_with_nthreads(1);  // Each thread gets its own single-threaded plan
    
    for (int t = 0; t < ctx->num_threads; t++) {
        ctx->thread_fft_input[t] = fftwf_alloc_real(ctx->fft_size);
        ctx->thread_fft_output[t] = fftwf_alloc_complex(fft_complex_size);
        ctx->thread_fft_mag[t] = (float*)malloc(fft_complex_size * sizeof(float));
        
        if (!ctx->thread_fft_input[t] || !ctx->thread_fft_output[t] || !ctx->thread_fft_mag[t]) {
            printf("Error: Failed to allocate FFT workspace for thread %d\n", t);
            return -1;
        }
        
        ctx->thread_fft_plans[t] = fftwf_plan_dft_r2c_1d(ctx->fft_size,
                                                         ctx->thread_fft_input[t],
                                                         ctx->thread_fft_output[t],
                                                         FFTW_ESTIMATE);
        
        if (!ctx->thread_fft_plans[t]) {
            printf("Error: Failed to create FFT plan for thread %d\n", t);
            return -1;
        }
    }
    
    printf("Fourier processing setup complete\n");
    return 0;
}

void precompute_fourier_references(ProcessingContext* ctx) {
    printf("\nPre-computing Fourier transforms for %d reference materials...\n",
           ctx->num_reference_vectors);
    
    // Use first thread's resources for pre-computation
    float* input = ctx->thread_fft_input[0];
    float complex* output = ctx->thread_fft_output[0];
    fftwf_plan plan = ctx->thread_fft_plans[0];
    
    int fft_complex_size = ctx->fft_size / 2 + 1;
    
    for (int i = 0; i < ctx->num_reference_vectors; i++) {
        // Clear input buffer
        memset(input, 0, ctx->fft_size * sizeof(float));
        
        // Interpolate reference spectrum to image wavelength grid
        HyperspectralVector* ref = &ctx->reference_vectors[i];
        
        for (int b = 0; b < ctx->image->bands && b < ctx->fft_size; b++) {
            float img_wav = ctx->image->wavelengths[b];
            
            // Find closest reference wavelength and interpolate
            if (img_wav <= ref->wavelengths[0]) {
                input[b] = ref->reflectance[0];
            } else if (img_wav >= ref->wavelengths[ref->size - 1]) {
                input[b] = ref->reflectance[ref->size - 1];
            } else {
                // Linear interpolation
                for (int j = 0; j < ref->size - 1; j++) {
                    if (ref->wavelengths[j] <= img_wav && ref->wavelengths[j+1] >= img_wav) {
                        float t = (img_wav - ref->wavelengths[j]) /
                                 (ref->wavelengths[j+1] - ref->wavelengths[j]);
                        input[b] = ref->reflectance[j] + t * (ref->reflectance[j+1] - ref->reflectance[j]);
                        break;
                    }
                }
            }
        }
        
        // Compute FFT
        fftwf_execute(plan);
        
        // Store FFT, magnitude, and compute energy
        ctx->reference_energies[i] = 0.0f;
        
        for (int j = 0; j < fft_complex_size; j++) {
            ctx->reference_fft[i][j] = output[j];
            ctx->reference_fft_mag[i][j] = cabsf(output[j]);
            ctx->reference_energies[i] += ctx->reference_fft_mag[i][j] * ctx->reference_fft_mag[i][j];
        }
        
        ctx->reference_energies[i] = sqrtf(ctx->reference_energies[i]);
        
        if (i % 100 == 0 || i == ctx->num_reference_vectors - 1) {
            printf("  Computed FFT for %d/%d materials (%.1f%%)\n",
                   i + 1, ctx->num_reference_vectors,
                   100.0 * (i + 1) / ctx->num_reference_vectors);
        }
    }
    
    printf("All reference Fourier transforms computed\n");
}

float calculate_fourier_similarity_optimized(float complex* pixel_fft, float complex* ref_fft,
                                            float* pixel_mag, float* ref_mag, int fft_size) {
    // Optimized similarity using magnitude correlation
    // Skip DC component (index 0) for better material discrimination
    float correlation = 0.0f;
    float pixel_energy = 0.0f;
    float ref_energy = 0.0f;
    
    for (int i = 1; i < fft_size; i++) {
        correlation += pixel_mag[i] * ref_mag[i];
        pixel_energy += pixel_mag[i] * pixel_mag[i];
        ref_energy += ref_mag[i] * ref_mag[i];
    }
    
    // Handle zero-energy cases
    if (pixel_energy < 1e-6f || ref_energy < 1e-6f) {
        return 0.0f;
    }
    
    // Normalized correlation (cosine similarity in frequency domain)
    return correlation / (sqrtf(pixel_energy) * sqrtf(ref_energy));
}

int classify_image_fourier_cpu(ProcessingContext* ctx) {
    printf("\nClassifying %lld pixels using Fourier transforms (CPU with OpenMP)...\n",
           (long long)ctx->image->width * ctx->image->height);
    
    int fft_complex_size = ctx->fft_size / 2 + 1;
    //long long total_pixels = (long long)ctx->image->width * ctx->image->height;
    
    #pragma omp parallel for schedule(dynamic, 100)
    for (int y = 0; y < ctx->image->height; y++) {
        int thread_id = omp_get_thread_num();
        float* input = ctx->thread_fft_input[thread_id];
        float complex* output = ctx->thread_fft_output[thread_id];
        float* mag = ctx->thread_fft_mag[thread_id];
        fftwf_plan plan = ctx->thread_fft_plans[thread_id];
        
        for (int x = 0; x < ctx->image->width; x++) {
            int pixel_idx = y * ctx->image->width + x;
            
            // Extract and zero-pad pixel spectrum
            memset(input, 0, ctx->fft_size * sizeof(float));
            for (int b = 0; b < ctx->image->bands; b++) {
                input[b] = ctx->image->data[b * ctx->image->width * ctx->image->height + pixel_idx];
            }
            
            // Compute pixel FFT
            fftwf_execute(plan);
            
            // Compute magnitude spectrum
            for (int j = 0; j < fft_complex_size; j++) {
                mag[j] = cabsf(output[j]);
            }
            
            // Find best matching reference material
            float best_similarity = -1.0f;
            int best_material = 0;
            
            for (int ref = 0; ref < ctx->num_reference_vectors; ref++) {
                float similarity = calculate_fourier_similarity_optimized(
                    output, ctx->reference_fft[ref],
                    mag, ctx->reference_fft_mag[ref],
                    fft_complex_size
                );
                
                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    best_material = ref;
                }
            }
            
            ctx->result->classification[pixel_idx] = (uint16_t)best_material;
            ctx->result->confidence[pixel_idx] = best_similarity;
        }
        
        if (y % 100 == 0) {
            printf("  Processed %d/%d rows (%.1f%%)\n",
                   y + 1, ctx->image->height,
                   100.0 * (y + 1) / ctx->image->height);
        }
    }
    
    printf("Fourier classification complete\n");
    return 0;
}

void cleanup_fourier_resources(ProcessingContext* ctx) {
    if (!ctx->use_fourier) return;
    
    printf("Cleaning up Fourier resources...\n");
    
    // Cleanup reference FFT data
    if (ctx->reference_fft) {
        for (int i = 0; i < ctx->num_reference_vectors; i++) {
            if (ctx->reference_fft[i]) fftwf_free(ctx->reference_fft[i]);
            if (ctx->reference_fft_mag && ctx->reference_fft_mag[i]) {
                free(ctx->reference_fft_mag[i]);
            }
        }
        free(ctx->reference_fft);
    }
    
    if (ctx->reference_fft_mag) free(ctx->reference_fft_mag);
    if (ctx->reference_energies) free(ctx->reference_energies);
    
    // Cleanup thread-local resources
    if (ctx->thread_fft_plans) {
        for (int t = 0; t < ctx->num_threads; t++) {
            if (ctx->thread_fft_plans[t]) fftwf_destroy_plan(ctx->thread_fft_plans[t]);
            if (ctx->thread_fft_input && ctx->thread_fft_input[t]) {
                fftwf_free(ctx->thread_fft_input[t]);
            }
            if (ctx->thread_fft_output && ctx->thread_fft_output[t]) {
                fftwf_free(ctx->thread_fft_output[t]);
            }
            if (ctx->thread_fft_mag && ctx->thread_fft_mag[t]) {
                free(ctx->thread_fft_mag[t]);
            }
        }
        free(ctx->thread_fft_plans);
    }
    
    if (ctx->thread_fft_input) free(ctx->thread_fft_input);
    if (ctx->thread_fft_output) free(ctx->thread_fft_output);
    if (ctx->thread_fft_mag) free(ctx->thread_fft_mag);
    
    // Cleanup FFTW threading
    fftwf_cleanup_threads();
    
    printf("Fourier resources cleaned up\n");
}

// ------ Fourier with spatial coherence boost --------
int classify_image_fourier_cpu_optimized(ProcessingContext* ctx) {
    printf("\nClassifying %lld pixels using Fourier transforms with spatial coherence...\n",
           (long long)ctx->image->width * ctx->image->height);
    
    int fft_complex_size = ctx->fft_size / 2 + 1;
    
    // Allocate per-pixel top-3 candidates (very small memory overhead)
    CandidateMatch* top_candidates = (CandidateMatch*)malloc(
        ctx->image->width * ctx->image->height * 3 * sizeof(CandidateMatch)
    );
    
    if (!top_candidates) {
        printf("Warning: Cannot allocate candidate buffer, using standard method\n");
        return classify_image_fourier_cpu(ctx);
    }
    
    // Phase 1: Find top 3 candidates for each pixel
    printf("Phase 1: Computing top-3 candidates per pixel...\n");
    
    #pragma omp parallel for schedule(dynamic, 100)
    for (int y = 0; y < ctx->image->height; y++) {
        int thread_id = omp_get_thread_num();
        float* input = ctx->thread_fft_input[thread_id];
        float complex* output = ctx->thread_fft_output[thread_id];
        float* mag = ctx->thread_fft_mag[thread_id];
        fftwf_plan plan = ctx->thread_fft_plans[thread_id];
        
        for (int x = 0; x < ctx->image->width; x++) {
            int pixel_idx = y * ctx->image->width + x;
            
            // Extract and zero-pad pixel spectrum
            memset(input, 0, ctx->fft_size * sizeof(float));
            for (int b = 0; b < ctx->image->bands; b++) {
                input[b] = ctx->image->data[b * ctx->image->width * ctx->image->height + pixel_idx];
            }
            
            // Compute pixel FFT
            fftwf_execute(plan);
            
            // Compute magnitude spectrum
            for (int j = 0; j < fft_complex_size; j++) {
                mag[j] = cabsf(output[j]);
            }
            
            // Find top 3 matching materials (minimal overhead)
            CandidateMatch top3[3] = {
                {0, -1.0f}, {0, -1.0f}, {0, -1.0f}
            };
            
            for (int ref = 0; ref < ctx->num_reference_vectors; ref++) {
                float similarity = calculate_fourier_similarity_optimized(
                    output, ctx->reference_fft[ref],
                    mag, ctx->reference_fft_mag[ref],
                    fft_complex_size
                );
                
                // Insert into top 3 if better
                if (similarity > top3[2].similarity) {
                    if (similarity > top3[0].similarity) {
                        top3[2] = top3[1];
                        top3[1] = top3[0];
                        top3[0] = (CandidateMatch){ref, similarity};
                    } else if (similarity > top3[1].similarity) {
                        top3[2] = top3[1];
                        top3[1] = (CandidateMatch){ref, similarity};
                    } else {
                        top3[2] = (CandidateMatch){ref, similarity};
                    }
                }
            }
            
            // Store top 3 candidates
            int cand_base = pixel_idx * 3;
            top_candidates[cand_base] = top3[0];
            top_candidates[cand_base + 1] = top3[1];
            top_candidates[cand_base + 2] = top3[2];
            
            // Initial classification (will be refined)
            ctx->result->classification[pixel_idx] = (uint16_t)top3[0].material_id;
            ctx->result->confidence[pixel_idx] = top3[0].similarity;
        }
        
        if (y % 100 == 0) {
            printf("  Phase 1: Processed %d/%d rows (%.1f%%)\n",
                   y + 1, ctx->image->height,
                   100.0 * (y + 1) / ctx->image->height);
        }
    }
    
    // Phase 2: Spatial coherence refinement (very fast - no FFT needed)
    printf("Phase 2: Applying spatial coherence refinement...\n");
    
    int refinements = 0;
    
    #pragma omp parallel for schedule(dynamic, 100) reduction(+:refinements)
    for (int y = 1; y < ctx->image->height - 1; y++) {
        for (int x = 1; x < ctx->image->width - 1; x++) {
            int pixel_idx = y * ctx->image->width + x;
            int cand_base = pixel_idx * 3;
            
            // Count neighbor materials (3x3 window, excluding center)
            int neighbor_votes[3] = {0, 0, 0}; // Votes for each of our top-3
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    int neighbor_idx = ny * ctx->image->width + nx;
                    int neighbor_material = ctx->result->classification[neighbor_idx];
                    
                    // Check if neighbor matches any of our top-3 candidates
                    for (int c = 0; c < 3; c++) {
                        if (top_candidates[cand_base + c].material_id == neighbor_material) {
                            neighbor_votes[c]++;
                            break;
                        }
                    }
                }
            }
            
            // Find best candidate considering spatial coherence
            int best_candidate = 0;
            float best_score = top_candidates[cand_base].similarity;
            
            for (int c = 1; c < 3; c++) {
                // Boost score based on neighbor agreement
                // Weight: 80% similarity + 20% spatial coherence
                float spatial_boost = neighbor_votes[c] / 8.0f; // Max 8 neighbors
                float combined_score = 0.8f * top_candidates[cand_base + c].similarity + 
                                      0.2f * spatial_boost;
                
                if (combined_score > best_score && neighbor_votes[c] >= 3) {
                    best_score = combined_score;
                    best_candidate = c;
                }
            }
            
            // Apply refinement if a better candidate was found
            if (best_candidate != 0) {
                ctx->result->classification[pixel_idx] = 
                    (uint16_t)top_candidates[cand_base + best_candidate].material_id;
                ctx->result->confidence[pixel_idx] = 
                    top_candidates[cand_base + best_candidate].similarity;
                refinements++;
            }
        }
        
        if (y % 100 == 0) {
            printf("  Phase 2: Processed %d/%d rows\n", y + 1, ctx->image->height);
        }
    }
    
    free(top_candidates);
    
    printf("Spatial coherence refinement: %d pixels reclassified (%.2f%%)\n",
           refinements, 100.0 * refinements / (ctx->image->width * ctx->image->height));
    printf("Fourier classification with spatial coherence complete\n");
    
    return 0;
}

// Alternative: Even lighter approach - neighbor-aware candidate selection
// Only refines when there's strong disagreement with neighbors
int classify_image_fourier_cpu_light(ProcessingContext* ctx) {
    printf("\nClassifying with lightweight neighbor checking...\n");
    
    int fft_complex_size = ctx->fft_size / 2 + 1;
    
    #pragma omp parallel for schedule(dynamic, 100)
    for (int y = 0; y < ctx->image->height; y++) {
        int thread_id = omp_get_thread_num();
        float* input = ctx->thread_fft_input[thread_id];
        float complex* output = ctx->thread_fft_output[thread_id];
        float* mag = ctx->thread_fft_mag[thread_id];
        fftwf_plan plan = ctx->thread_fft_plans[thread_id];
        
        for (int x = 0; x < ctx->image->width; x++) {
            int pixel_idx = y * ctx->image->width + x;
            
            // Extract and compute FFT
            memset(input, 0, ctx->fft_size * sizeof(float));
            for (int b = 0; b < ctx->image->bands; b++) {
                input[b] = ctx->image->data[b * ctx->image->width * ctx->image->height + pixel_idx];
            }
            fftwf_execute(plan);
            
            for (int j = 0; j < fft_complex_size; j++) {
                mag[j] = cabsf(output[j]);
            }
            
            // Get neighbor consensus (if available)
            int neighbor_material = -1;
            int neighbor_count = 0;
            
            if (x > 0 && y > 0) {
                int left_material = ctx->result->classification[pixel_idx - 1];
                int top_material = ctx->result->classification[pixel_idx - ctx->image->width];
                
                if (left_material == top_material) {
                    neighbor_material = left_material;
                    neighbor_count = 2;
                }
            }
            
            // Find best match - compare only top candidates
            float best_similarity = -1.0f;
            int best_material = 0;
            float neighbor_similarity = -1.0f;
            
            // If we have strong neighbor agreement, check that material first
            if (neighbor_count >= 2 && neighbor_material >= 0) {
                neighbor_similarity = calculate_fourier_similarity_optimized(
                    output, ctx->reference_fft[neighbor_material],
                    mag, ctx->reference_fft_mag[neighbor_material],
                    fft_complex_size
                );
                
                // If neighbor match is good (>0.85), use it immediately
                if (neighbor_similarity > 0.85f) {
                    ctx->result->classification[pixel_idx] = (uint16_t)neighbor_material;
                    ctx->result->confidence[pixel_idx] = neighbor_similarity;
                    continue; // Skip full search
                }
                
                best_similarity = neighbor_similarity;
                best_material = neighbor_material;
            }
            
            // Full search through all references
            for (int ref = 0; ref < ctx->num_reference_vectors; ref++) {
                if (ref == neighbor_material) continue; // Already checked
                
                float similarity = calculate_fourier_similarity_optimized(
                    output, ctx->reference_fft[ref],
                    mag, ctx->reference_fft_mag[ref],
                    fft_complex_size
                );
                
                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    best_material = ref;
                }
            }
            
            ctx->result->classification[pixel_idx] = (uint16_t)best_material;
            ctx->result->confidence[pixel_idx] = best_similarity;
        }
        
        if (y % 100 == 0) {
            printf("  Processed %d/%d rows (%.1f%%)\n",
                   y + 1, ctx->image->height,
                   100.0 * (y + 1) / ctx->image->height);
        }
    }
    
    printf("Lightweight neighbor-aware classification complete\n");
    return 0;
}

