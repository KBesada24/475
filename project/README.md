# Midterm Project: Object Detection and Image Processing

This project is a self-contained Python image-processing submission. It generates every source image procedurally, then applies object-detection and image-processing techniques related to the assignment slides: filtering, edges, segmentation, morphology, connected components, and brightness/saturation/darkness feature extraction.

## Requirements

- Python 3
- NumPy
- Pillow

The dependencies are listed in `requirements.txt`.

## Run

```bash
python3 generate_images.py
```

Running the script writes seven PNG files into this `project` directory.

## Generated Images

1. `01_generated_scene.png` - a procedural scene with multiple colored objects, shadows, texture, and noise.
2. `02_convolution_filters.png` - Gaussian blur, unsharp masking, Sobel response, Sobel magnitude, and emboss filtering using custom convolution code.
3. `03_frequency_filtering.png` - FFT magnitude visualization, circular low-pass reconstruction, high-pass detail, and a hybrid frequency result.
4. `04_edge_detection_pipeline.png` - grayscale conversion, smoothing, gradient magnitude, non-maximum suppression, binary thresholding, and edge overlay.
5. `05_object_detection_morphology.png` - color segmentation, morphological opening and closing, connected-component labeling, and bounding-box object overlays.
6. `06_opening_closing_challenge.png` - a generated binary text challenge showing how opening removes small noise and how closing fills holes and gaps.
7. `07_naturalness_features.png` - brightness, saturation, darkness, brightness-darkness difference, histograms, and a compact feature-vector summary.

## Methods Demonstrated

- Procedural image generation
- RGB to grayscale conversion
- 2D convolution with explicit kernels
- Gaussian blur and unsharp masking
- Sobel edge detection
- Non-maximum suppression
- Fourier-domain low-pass and high-pass filtering
- Binary erosion and dilation
- Morphological opening and closing
- Connected-component labeling for simple object detection
- HSV saturation extraction
- Brightness and darkness feature channels
- Histogram-based naturalness feature summaries

## Morphology Notes

Opening is implemented as erosion followed by dilation. In the challenge image, it removes thin scratches and isolated white noise from the binary text.

Closing is implemented as dilation followed by erosion. In the challenge image, it repairs small dark gaps and holes inside the foreground text.

The object-detection panel uses the same operations after color segmentation: opening removes small false-positive pixels, closing reconnects nearby object regions, and connected components convert the cleaned mask into labeled detections with bounding boxes.

## Brightness, Saturation, and Darkness Notes

The naturalness panel implements the slide's channel definitions directly from RGB values:

- Brightness channel: `Ibr(x, y) = max(R(x, y), G(x, y), B(x, y))`
- Darkness channel: `Ida(x, y) = min(R(x, y), G(x, y), B(x, y))`
- Difference channel: `Idi(x, y) = Ibr(x, y) - Ida(x, y)`
- Saturation channel: extracted from HSV/HSB as chroma divided by brightness

For brightness, saturation, and darkness, the script computes histograms plus simple feature statistics: mean, standard deviation, third central moment, and contrast/range. This mirrors the feature-vector idea from the slide without requiring an external image dataset.
