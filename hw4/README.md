# HW4: Image Processing and Rendering

This submission is a self-contained Python project that generates five images with no external image assets.

## Requirements

- Python 3
- Pillow
- NumPy

The Python dependencies are listed in `requirements.txt`.

## Run

```bash
python3 generate_images.py
```

The script writes these files into this `hw4` directory:

1. `01_procedural_scene.png` - a procedural landscape rendered from gradients, waves, masks, and noise.
2. `02_sobel_edge_overlay.png` - a Sobel edge-detection pass composited over the procedural image.
3. `03_floyd_steinberg_dither.png` - palette quantization with Floyd-Steinberg error diffusion.
4. `04_halftone_render.png` - a layered dot-screen halftone rendering.
5. `05_raytraced_spheres.png` - a small ray tracer with spheres, a checker plane, lighting, reflections, and gamma correction.

## Notes

The implementation is intentionally asset-free and deterministic. All pixels are generated from Python code in `generate_images.py`, making the results reproducible for grading.
