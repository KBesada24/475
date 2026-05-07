# HW5: Image Processing

This homework submission is a self-contained Python project. It generates every input image procedurally, then applies several image-processing techniques without relying on external image files.

## Requirements

- Python 3
- NumPy
- Pillow

Install the dependencies listed in `requirements.txt` if needed.

## Run

```bash
python3 generate_images.py
```

The script writes these files into this `hw5` directory:

1. `01_generated_still_life.png` - a procedural RGB source image with gradients, shapes, shadows, texture, and noise.
2. `02_convolution_filter_grid.png` - Gaussian blur, unsharp masking, Sobel magnitude, and emboss filtering from custom convolution code.
3. `03_frequency_filtering.png` - FFT magnitude visualization, circular low-pass reconstruction, high-pass detail, and a hybrid result.
4. `04_edge_pipeline.png` - grayscale conversion, smoothed gradient magnitude, non-maximum suppression, and final thresholded edges.
5. `05_segmentation_morphology.png` - color-based segmentation, opening/closing morphology, connected-component labeling, and object overlays.
6. `06_laplacian_pyramid_blend.png` - a generated moon/planet and sunset scene blended with a Laplacian pyramid.

## Methods Used

- Procedural image generation
- 2D convolution with explicitly defined kernels
- Gaussian blur and unsharp masking
- Sobel edge detection
- Non-maximum suppression
- Fourier-domain low-pass and high-pass filtering
- Binary erosion, dilation, opening, and closing
- Connected-component labeling
- Laplacian pyramid blending

The project is deterministic: running the script again produces the same images.
