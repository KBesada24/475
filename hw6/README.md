# HW6: Image Enhancement and Segmentation

This submission is a self-contained Python project for the two HW6 screenshots. It generates deterministic grayscale and color images, then implements the requested enhancement and segmentation algorithms directly with NumPy and Pillow.

## Requirements

- Python 3
- NumPy
- Pillow

Install the dependencies listed in `requirements.txt` if needed.

## Run

From the repository root:

```bash
python3 hw6/generate_images.py
```

Or from this folder:

```bash
python3 generate_images.py
```

The script writes all PNG outputs into this `hw6` directory.

## Task Mapping

1. Contrast stretching: `03_contrast_stretching.png`
2. Image quality measure and local filter: `04_quality_measure_filter.png`, with numeric values in `summary.txt`
3. Histogram equalization: `05_histogram_equalization.png`
4. Equalization transformation function `u vs. v`: `06_equalization_transformation_curve.png`
5. Contrast stretching comparison and spatial enhancement discussion: `03_contrast_stretching.png`, `07_gamma_correction.png`
6. Gamma correction from image `f` to image `g`: `07_gamma_correction.png`
7. Color image enhancement: `08_color_enhancement.png`
8. Histogram-based segmentation, peaks, ranges, and identified objects: `09_histogram_segmentation_plot.png`, `10_histogram_identified_objects.png`, `11_histogram_segmentation_masks.png`
9. Global thresholding for two-mode and three-mode images using different initial estimates: `12_global_threshold_two_mode_hist.png`, `13_global_threshold_three_mode_hist.png`, `14_global_thresholding.png`
10. Multilevel and adaptive thresholding with different initial estimates/settings: `15_multilevel_threshold_hist.png`, `16_multilevel_and_adaptive_thresholding.png`
11. Niblack, Bernsen, and Sauvola thresholding on gray and color images: `17_niblack_bernsen_sauvola_gray.png`, `18_niblack_bernsen_sauvola_color.png`
12. Bonus mean histogram stretching for enhancement and segmentation: `19_bonus_mean_histogram_stretch_segmentation.png`

## Methods

The contrast stretching function uses the 2nd and 98th percentiles as robust low/high bounds, then maps the clipped range to `[0, 1]`. This improves the intentionally low-contrast source image while avoiding excessive influence from noise.

The quality measure follows the local block idea from EME/EEME. For each block, the code computes local max/min contrast with a small constant `c`, applies `alpha`, averages the block scores, and also produces a local enhanced image from the same block statistics.

Histogram equalization is implemented manually from the 256-bin histogram and cumulative distribution function. The plotted transformation curve shows the mapping from input intensity `u` to output intensity `v`.

Gamma correction uses `g = f^gamma` with `gamma = 0.55`, which brightens midtones. The spatial enhancement example applies mean histogram stretching after gamma correction to increase contrast around the image mean.

Color enhancement works through luminance. The script equalizes and stretches luminance, rescales RGB channels by the luminance ratio, then applies a modest saturation boost so colors remain natural.

Segmentation uses generated images with two and three clear intensity modes. The histogram segmentation task marks the object ranges directly on the histogram and overlays bounding boxes around the identified objects. Global thresholding uses the iterative intermeans algorithm from multiple starting thresholds. Multilevel thresholding extends that idea to three classes. Adaptive thresholding, Niblack, Bernsen, and Sauvola use local window statistics.

## Discussion

Contrast stretching and histogram equalization both improve visibility, but they behave differently. Contrast stretching preserves the original ordering and relative spacing of intensities after clipping, while histogram equalization redistributes intensities more aggressively. In the generated image, equalization reveals more texture in the darker regions, but contrast stretching gives a smoother and less harsh result.

Gamma correction brightens the image without needing a local window, so it is useful when the main issue is global underexposure. The mean histogram stretch bonus gives a stronger segmentation-friendly result because it increases separation around the image mean before thresholding.

For segmentation, global thresholding works best on the two-mode image because the histogram naturally separates foreground and background. The three-mode image benefits from multilevel thresholding because one threshold cannot preserve both object classes. Adaptive, Niblack, Bernsen, and Sauvola methods are better when illumination varies across the image, although their output depends on local window size and threshold constants.
