# Hypnagogic

A GRASS GIS Module to deal with pesky multi-bands imagery

~~~shell
./hypna_cpu --help
Hyperspectral Image Classification System V6
============================================
Usage: ./hypna_cpu input.tif output.tif [options]

Options:
  --diagnose, -d              : Only diagnose the input file structure
  --mode=<mode>               : Classification algorithm mode
  --help, -h                  : Show this help message

Available modes:
  fourier_cpu                 : Baseline Fourier transform method
                                 - Standard accuracy
                                 - Good for validation/comparison

  fourier_cpu_coherence_quality : Spatial coherence refinement (DEFAULT)
                                 - Best accuracy (~5-10% slower)
                                 - Reduces salt-and-pepper noise
                                 - Preserves field boundaries
                                 - Recommended for production

  fourier_cpu_coherence_fastest : Lightweight neighbor prediction
                                 - Fastest method (30-50% speedup)
                                 - Good accuracy on homogeneous scenes
                                 - Ideal for large datasets

  spatial_cpu                 : Original spatial domain method
                                 - No Fourier transform
                                 - Direct spectral comparison
                                 - Baseline for benchmarking

Examples:
  ./hypna_cpu field.tif result.tif
  ./hypna_cpu field.tif result.tif --mode=fourier_cpu_coherence_fastest
  ./hypna_cpu field.tif result.tif --mode=fourier_cpu
  ./hypna_cpu field.tif --diagnose
~~~
