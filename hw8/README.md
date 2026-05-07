# HW8: Histogram Recognition and DFT Image Processing

This submission is a self-contained Python project for the HW8 image-processing assignment. It generates every input image procedurally, then applies histogram-based recognition/classification and Discrete Fourier Transform processing.

## Requirements

- Python 3
- NumPy
- Pillow

The dependencies are listed in `requirements.txt`.

## Run

```bash
python3 generate_images.py
```

The script writes all generated images into this `hw8` directory and prints classification accuracy plus DFT validation results.

## Output Files

1. `01_database_examples.png` - generated image database with good and noisy traffic-sign scenes.
2. `02_enhancement_comparison.png` - noisy input, median restoration, Gaussian smoothing, unsharp enhancement, contrast stretch, and selected best-enhanced image.
3. `03_segmentation_comparison.png` - segmentation candidates, morphology cleanup, and selected object mask.
4. `04_segmented_image.png` - best-enhanced image, binary mask, and the segmented image obtained by multiplying the image by the mask.
5. `05_histograms.png` - segmented gray-level histogram and oriented-gradient histogram features.
6. `06_classification_results.png` - template/test predictions, similarity scores, and accuracy.
7. `07_manual_dft_validation.png` - manual 2D DFT/IDFT on a small image and comparison against NumPy FFT.
8. `08_fftshift_explanation.png` - unshifted and shifted DFT magnitude visualizations showing why `fftshift` centers low frequencies.
9. `09_dft_properties.png` - numerical demonstrations of DFT linearity, shift, convolution, Parseval energy, and conjugate symmetry.
10. `10_dft_noise_filtering.png` - periodic noise removal using a frequency-domain notch filter.
11. `11_dft_encryption.png` - reversible Fourier-domain image encryption and decryption.

## Assignment Mapping

### Histogram Recognition / Classification Slide

- Input database images, including good and noisy images: `01_database_examples.png`.
- Restoration/enhancement methods and selected best method: `02_enhancement_comparison.png`.
- Segmentation methods and selected best method: `03_segmentation_comparison.png`.
- Multiplication of enhanced image by binary mask: `04_segmented_image.png`.
- Histograms of segmented gray image and oriented gradients: `05_histograms.png`.
- Histogram similarity classification with chi-squared distance and correlation: `06_classification_results.png`.
- Accuracy percentage: printed by the script and shown in `06_classification_results.png`.

### DFT Slide

- Implement the DFT: manual 2D DFT and IDFT functions in `generate_images.py`, visualized in `07_manual_dft_validation.png`.
- Show why to use `fftshift`: `08_fftshift_explanation.png`.
- Prove basic DFT properties: numerical property checks in `09_dft_properties.png`.
- Apply the DFT to filter noise from an image: `10_dft_noise_filtering.png`.
- Apply the DFT to encrypt an image: `11_dft_encryption.png`.

## Methods Used

- Procedural image generation
- Median, Gaussian, unsharp, and contrast enhancement
- Binary segmentation, erosion, dilation, opening, and closing
- Gray-level histograms
- HOG-style oriented-gradient histograms
- Chi-squared histogram distance
- Histogram correlation coefficient
- Manual 2D DFT and IDFT
- Frequency spectrum visualization
- Frequency-domain notch filtering
- Fourier-domain reversible encryption

The project is deterministic: running the script again produces the same outputs.
