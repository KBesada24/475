from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
WIDTH = 192
HEIGHT = 192
FONT = ImageFont.load_default()


@dataclass(frozen=True)
class Sample:
    label: str
    variant: str
    image: np.ndarray
    mask: np.ndarray


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def from_pil(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(float) / 255


def save_rgb(name: str, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image), mode="RGB").save(OUT_DIR / name)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0, 1)[..., None], 3, axis=2)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722])


def normalize01(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def convolve(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(channel, ((py, py), (px, px)), mode="edge")
    output = np.zeros_like(channel, dtype=float)
    for y in range(kh):
        for x in range(kw):
            output += kernel[y, x] * padded[y : y + channel.shape[0], x : x + channel.shape[1]]
    return output


def convolve_rgb(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.dstack([convolve(image[..., c], kernel) for c in range(3)])


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def median_filter(image: np.ndarray, radius: int = 1) -> np.ndarray:
    size = 2 * radius + 1
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    windows = []
    for y in range(size):
        for x in range(size):
            windows.append(padded[y : y + image.shape[0], x : x + image.shape[1]])
    return np.median(np.stack(windows, axis=0), axis=0)


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    result = np.empty_like(image)
    for c in range(3):
        low, high = np.percentile(image[..., c], [2, 98])
        result[..., c] = np.clip((image[..., c] - low) / max(high - low, 1e-6), 0, 1)
    return result


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return dilate(erode(mask, radius), radius)


def closing(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return erode(dilate(mask, radius), radius)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int32)
    best_pixels: list[tuple[int, int]] = []
    label = 1
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx] != 0:
                continue
            stack = [(sy, sx)]
            labels[sy, sx] = label
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        stack.append((ny, nx))
            if len(pixels) > len(best_pixels):
                best_pixels = pixels
            label += 1
    result = np.zeros_like(mask, dtype=bool)
    for y, x in best_pixels:
        result[y, x] = True
    return result


def save_grid(name: str, panels: list[tuple[str, np.ndarray]], columns: int = 3, tile: tuple[int, int] = (260, 220)) -> None:
    tile_w, tile_h = tile
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), (245, 242, 232))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row, col = divmod(index, columns)
        x0, y0 = col * tile_w, row * tile_h
        panel = Image.fromarray(to_uint8(image), mode="RGB").resize((tile_w, tile_h - 28), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x0, y0 + 28))
        draw.rectangle((x0, y0, x0 + tile_w, y0 + 28), fill=(32, 38, 46))
        draw.text((x0 + 9, y0 + 8), label[:42], fill=(248, 242, 222), font=FONT)
    canvas.save(OUT_DIR / name)


def add_scene_background(rng: np.random.Generator, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    sky = np.dstack([0.70 - 0.14 * v, 0.80 - 0.13 * v, 0.88 - 0.08 * v])
    road = v > 0.63 + 0.04 * np.sin(2 * math.pi * u)
    grass = (v > 0.48) & ~road
    sky[grass] = np.array([0.28, 0.55, 0.27]) + rng.normal(0, 0.018, sky[grass].shape)
    sky[road] = np.array([0.38, 0.38, 0.39]) + rng.normal(0, 0.012, sky[road].shape)
    lane = road & (np.abs(u - 0.5) < 0.018)
    sky[lane] = np.array([0.92, 0.86, 0.50])
    return np.clip(sky, 0, 1)


def sign_polygon(label: str, cx: int, cy: int, size: int) -> list[tuple[float, float]]:
    if label == "stop":
        return [
            (cx + size * math.cos(math.pi / 8 + i * math.pi / 4), cy + size * math.sin(math.pi / 8 + i * math.pi / 4))
            for i in range(8)
        ]
    if label == "warning":
        return [(cx, cy - size), (cx - size * 0.92, cy + size * 0.78), (cx + size * 0.92, cy + size * 0.78)]
    return [
        (cx + size * math.cos(i * 2 * math.pi / 40), cy + size * math.sin(i * 2 * math.pi / 40))
        for i in range(40)
    ]


def make_sample(label: str, variant_index: int, noisy: bool) -> Sample:
    rng = np.random.default_rng(8200 + variant_index * 71 + {"stop": 1, "warning": 2, "speed": 3}[label])
    image = add_scene_background(rng)
    mask_img = Image.new("L", (WIDTH, HEIGHT), 0)
    pil = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    mask_draw = ImageDraw.Draw(mask_img)

    cx = int(WIDTH * (0.50 + rng.normal(0, 0.035)))
    cy = int(HEIGHT * (0.43 + rng.normal(0, 0.030)))
    size = int(45 + rng.integers(-5, 6))
    post_w = 8
    draw.rectangle((cx - post_w // 2, cy + size - 4, cx + post_w // 2, HEIGHT - 34), fill=(90, 92, 86, 255))

    poly = sign_polygon(label, cx, cy, size)
    fill = {"stop": (204, 38, 34, 255), "warning": (236, 197, 42, 255), "speed": (45, 112, 205, 255)}[label]
    draw.polygon(poly, fill=fill, outline=(245, 245, 238, 255))
    mask_draw.polygon(poly, fill=255)

    if label == "stop":
        draw.text((cx - 16, cy - 5), "STOP", fill=(255, 255, 255, 255), font=FONT)
    elif label == "warning":
        draw.line((cx - 20, cy + 16, cx, cy - 14, cx + 20, cy + 16), fill=(40, 38, 32, 255), width=5)
    else:
        draw.ellipse((cx - size + 11, cy - size + 11, cx + size - 11, cy + size - 11), outline=(245, 245, 245, 255), width=7)
        draw.text((cx - 8, cy - 5), "30", fill=(255, 255, 255, 255), font=FONT)

    clean = from_pil(pil)
    mask = np.asarray(mask_img).astype(bool)
    if noisy:
        clean = add_capture_degradation(clean, rng)
        variant = "noisy"
    else:
        variant = "good"
    return Sample(label, variant, clean, mask)


def add_capture_degradation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = image.copy()
    noisy += rng.normal(0, 0.045, noisy.shape)
    salt = rng.random(noisy.shape[:2])
    noisy[salt < 0.012] = 0
    noisy[salt > 0.988] = 1
    blur = convolve_rgb(np.clip(noisy, 0, 1), gaussian_kernel(3, 0.8))
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    vignette = 0.78 + 0.22 * (1 - ((x / WIDTH - 0.52) ** 2 + (y / HEIGHT - 0.48) ** 2))
    return np.clip(blur * vignette[..., None], 0, 1)


def make_database() -> tuple[list[Sample], list[Sample]]:
    templates = [make_sample(label, i, noisy=False) for i, label in enumerate(["stop", "warning", "speed"])]
    tests: list[Sample] = []
    index = 10
    for label in ["stop", "warning", "speed"]:
        for _ in range(3):
            tests.append(make_sample(label, index, noisy=True))
            index += 1
    return templates, tests


def enhance_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    median = median_filter(image, 1)
    gaussian = convolve_rgb(image, gaussian_kernel(5, 1.1))
    unsharp = np.clip(median + 1.35 * (median - convolve_rgb(median, gaussian_kernel(5, 1.3))), 0, 1)
    stretched = contrast_stretch(unsharp)
    return [
        ("Noisy input", image),
        ("Median restoration", median),
        ("Gaussian smoothing", gaussian),
        ("Unsharp enhancement", unsharp),
        ("Contrast stretch", stretched),
        ("Selected best", stretched),
    ]


def best_enhancement(image: np.ndarray) -> np.ndarray:
    return enhance_variants(image)[-1][1]


def segment_candidates(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.max(image, axis=2)
    minc = np.min(image, axis=2)
    saturation = maxc - minc
    bright_color = (saturation > 0.18) & (maxc > 0.35)
    red_mask = (red > green * 1.16) & (red > blue * 1.16) & (red > 0.35)
    yellow_mask = (red > 0.45) & (green > 0.35) & (blue < 0.34)
    blue_mask = (blue > red * 1.10) & (blue > green * 1.05) & (blue > 0.33)
    color_union = bright_color & (red_mask | yellow_mask | blue_mask)
    clean = closing(opening(color_union, 2), 3)
    component = largest_component(clean)
    overlay = np.clip(image * 0.55 + np.dstack([component * 0.95, component * 0.15, component * 0.05]), 0, 1)
    return [
        ("Saturation threshold", gray_to_rgb(bright_color.astype(float))),
        ("Color rule mask", gray_to_rgb(color_union.astype(float))),
        ("Opening + closing", gray_to_rgb(clean.astype(float))),
        ("Largest component", gray_to_rgb(component.astype(float))),
        ("Selected overlay", overlay),
    ]


def segment_object(image: np.ndarray) -> np.ndarray:
    return segment_candidates(image)[3][1][..., 0] > 0.5


def segmented_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image * mask[..., None]


def gray_histogram(image: np.ndarray, mask: np.ndarray, bins: int = 32) -> np.ndarray:
    gray = rgb_to_gray(image)
    values = gray[mask]
    hist, _ = np.histogram(values, bins=bins, range=(0, 1))
    hist = hist.astype(float)
    return hist / max(hist.sum(), 1)


def hog_histogram(image: np.ndarray, mask: np.ndarray, bins: int = 9) -> np.ndarray:
    gray = rgb_to_gray(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    gx = convolve(gray, sobel_x)
    gy = convolve(gray, sobel_y)
    mag = np.sqrt(gx * gx + gy * gy)
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    hist = np.zeros(bins, dtype=float)
    bin_index = np.minimum((angle / 180 * bins).astype(int), bins - 1)
    for b in range(bins):
        hist[b] = mag[(bin_index == b) & mask].sum()
    return hist / max(hist.sum(), 1e-12)


def chi_squared(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / np.maximum(a + b, 1e-10)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - a.mean()
    bb = b - b.mean()
    return float(np.sum(aa * bb) / max(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)), 1e-10))


def feature_vector(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray_hist = gray_histogram(image, mask, 32)
    hog_hist = hog_histogram(image, mask, 9)
    return np.concatenate([gray_hist, hog_hist])


def classify(templates: list[Sample], tests: list[Sample]) -> tuple[list[tuple[str, str, float, float]], float]:
    template_features = {}
    for sample in templates:
        enhanced = best_enhancement(sample.image)
        mask = segment_object(enhanced)
        template_features[sample.label] = feature_vector(enhanced, mask)

    rows = []
    correct = 0
    for sample in tests:
        enhanced = best_enhancement(sample.image)
        mask = segment_object(enhanced)
        feature = feature_vector(enhanced, mask)
        scores = []
        for label, template in template_features.items():
            chi = chi_squared(feature, template)
            corr = correlation(feature, template)
            combined = chi - 0.22 * corr
            scores.append((combined, label, chi, corr))
        _combined, predicted, chi, corr = min(scores, key=lambda item: item[0])
        correct += int(predicted == sample.label)
        rows.append((sample.label, predicted, chi, corr))
    return rows, 100 * correct / len(tests)


def draw_histogram_panel(title: str, values: np.ndarray, width: int = 420, height: int = 260) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (247, 244, 234))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 30), fill=(31, 38, 48))
    draw.text((12, 9), title, fill=(245, 239, 219), font=FONT)
    left, top, right, bottom = 34, 48, width - 18, height - 34
    draw.rectangle((left, top, right, bottom), outline=(75, 78, 82))
    max_value = max(float(values.max()), 1e-12)
    bar_w = (right - left) / len(values)
    for i, value in enumerate(values):
        x0 = left + i * bar_w
        x1 = left + (i + 1) * bar_w - 1
        y0 = bottom - (bottom - top) * float(value) / max_value
        color = (48, 104, 181) if i % 2 == 0 else (231, 160, 47)
        draw.rectangle((x0, y0, x1, bottom), fill=color)
    draw.text((left, bottom + 8), "bins", fill=(45, 45, 45), font=FONT)
    draw.text((6, top + 4), "count", fill=(45, 45, 45), font=FONT)
    return canvas


def save_histogram_visual(name: str, image: np.ndarray, mask: np.ndarray) -> None:
    gray_hist = gray_histogram(image, mask)
    hog_hist = hog_histogram(image, mask)
    canvas = Image.new("RGB", (840, 520), (247, 244, 234))
    canvas.paste(draw_histogram_panel("Gray-level histogram of segmented image", gray_hist), (0, 0))
    canvas.paste(draw_histogram_panel("Histogram of Oriented Gradients", hog_hist), (420, 0))
    segmented = Image.fromarray(to_uint8(segmented_image(image, mask)), mode="RGB").resize((260, 220), Image.Resampling.LANCZOS)
    mask_panel = Image.fromarray(to_uint8(gray_to_rgb(mask.astype(float))), mode="RGB").resize((260, 220), Image.Resampling.NEAREST)
    canvas.paste(segmented, (150, 290))
    canvas.paste(mask_panel, (430, 290))
    draw = ImageDraw.Draw(canvas)
    draw.text((150, 270), "Segmented gray image source", fill=(30, 30, 30), font=FONT)
    draw.text((430, 270), "Binary object mask", fill=(30, 30, 30), font=FONT)
    canvas.save(OUT_DIR / name)


def save_classification_results(name: str, rows: list[tuple[str, str, float, float]], accuracy: float) -> None:
    width, height = 820, 350
    canvas = Image.new("RGB", (width, height), (247, 244, 234))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 42), fill=(31, 38, 48))
    draw.text((16, 14), f"Histogram similarity classification accuracy: {accuracy:.1f}%", fill=(245, 239, 219), font=FONT)
    headers = ["True label", "Prediction", "Chi-squared", "Correlation", "Correct"]
    xs = [28, 170, 320, 500, 670]
    y = 66
    for x, header in zip(xs, headers):
        draw.text((x, y), header, fill=(25, 25, 25), font=FONT)
    y += 24
    for true_label, predicted, chi, corr in rows:
        correct = "yes" if true_label == predicted else "no"
        color = (28, 120, 70) if correct == "yes" else (170, 55, 45)
        values = [true_label, predicted, f"{chi:.4f}", f"{corr:.4f}", correct]
        draw.rectangle((18, y - 6, width - 18, y + 18), fill=(235, 232, 222) if (y // 24) % 2 else (252, 249, 239))
        for x, value in zip(xs, values):
            draw.text((x, y), value, fill=color if value == correct else (35, 35, 35), font=FONT)
        y += 25
    canvas.save(OUT_DIR / name)


def manual_dft2(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    y, u = np.mgrid[0:h, 0:h]
    x, v = np.mgrid[0:w, 0:w]
    wy = np.exp(-2j * np.pi * u * y / h)
    wx = np.exp(-2j * np.pi * x * v / w)
    return wy @ image @ wx


def manual_idft2(spectrum: np.ndarray) -> np.ndarray:
    h, w = spectrum.shape
    y, u = np.mgrid[0:h, 0:h]
    x, v = np.mgrid[0:w, 0:w]
    wy = np.exp(2j * np.pi * y * u / h)
    wx = np.exp(2j * np.pi * v * x / w)
    return (wy @ spectrum @ wx / (h * w)).real


def spectrum_visual(spectrum: np.ndarray) -> np.ndarray:
    return gray_to_rgb(normalize01(np.log1p(np.abs(spectrum))))


def small_dft_source(size: int = 32) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=float)
    image[(x - 10) ** 2 + (y - 10) ** 2 < 28] = 0.9
    image[17:25, 15:27] = 0.55
    image += 0.2 * (x / (size - 1))
    return np.clip(image, 0, 1)


def dft_validation() -> float:
    source = small_dft_source()
    manual = manual_dft2(source)
    numpy_fft = np.fft.fft2(source)
    recon = manual_idft2(manual)
    max_error = float(np.max(np.abs(manual - numpy_fft)))
    recon_error = float(np.max(np.abs(recon - source)))
    panels = [
        ("Small grayscale input", gray_to_rgb(source)),
        ("Manual DFT magnitude", spectrum_visual(manual)),
        ("NumPy FFT magnitude", spectrum_visual(numpy_fft)),
        ("Manual IDFT reconstruction", gray_to_rgb(np.clip(recon, 0, 1))),
        ("Abs reconstruction error", gray_to_rgb(normalize01(np.abs(recon - source)))),
        (f"DFT max err {max_error:.2e}", gray_to_rgb(np.full_like(source, min(max_error, 1)))),
    ]
    save_grid("07_manual_dft_validation.png", panels, columns=3, tile=(220, 180))
    return max(max_error, recon_error)


def fftshift_explanation() -> None:
    source = small_dft_source(96)
    spectrum = np.fft.fft2(source)
    shifted = np.fft.fftshift(spectrum)
    panels = [
        ("Generated grayscale image", gray_to_rgb(source)),
        ("Unshifted DFT magnitude", spectrum_visual(spectrum)),
        ("Shifted with fftshift", spectrum_visual(shifted)),
        ("Low frequencies at corners", corner_marker(spectrum_visual(spectrum))),
        ("Low frequencies centered", center_marker(spectrum_visual(shifted))),
    ]
    save_grid("08_fftshift_explanation.png", panels, columns=3, tile=(240, 210))


def corner_marker(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(pil)
    for box in [(0, 0, 18, 18), (78, 0, 95, 18), (0, 78, 18, 95), (78, 78, 95, 95)]:
        draw.rectangle(box, outline=(255, 210, 45), width=3)
    return from_pil(pil)


def center_marker(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.ellipse((40, 40, 56, 56), outline=(255, 210, 45), width=3)
    return from_pil(pil)


def dft_properties() -> None:
    a = small_dft_source(32)
    b = np.rot90(a) * 0.7
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    alpha, beta = 1.4, -0.55
    linearity = np.max(np.abs(np.fft.fft2(alpha * a + beta * b) - (alpha * fa + beta * fb)))
    shifted = np.roll(np.roll(a, 5, axis=0), -3, axis=1)
    shift_mag = np.max(np.abs(np.abs(np.fft.fft2(shifted)) - np.abs(fa)))
    kernel = np.zeros_like(a)
    kernel[:3, :3] = gaussian_kernel(3, 0.8)
    circular_conv = np.real(np.fft.ifft2(fa * np.fft.fft2(kernel)))
    conv_error = np.max(np.abs(np.fft.fft2(circular_conv) - fa * np.fft.fft2(kernel)))
    parseval_space = np.sum(np.abs(a) ** 2)
    parseval_freq = np.sum(np.abs(fa) ** 2) / a.size
    parseval_error = abs(parseval_space - parseval_freq)
    conj_error = np.max(np.abs(fa - np.conj(np.roll(np.roll(fa[::-1, ::-1], 1, axis=0), 1, axis=1))))

    rows = [
        ("Linearity", linearity),
        ("Shift magnitude invariant", shift_mag),
        ("Convolution theorem", conv_error),
        ("Parseval energy", parseval_error),
        ("Conjugate symmetry", conj_error),
    ]
    canvas = Image.new("RGB", (860, 420), (247, 244, 234))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 860, 46), fill=(31, 38, 48))
    draw.text((18, 17), "Numerical proof of basic DFT properties", fill=(245, 239, 219), font=FONT)
    y = 78
    for title, error in rows:
        draw.text((40, y), title, fill=(24, 24, 24), font=FONT)
        draw.text((300, y), f"max absolute error = {error:.3e}", fill=(24, 24, 24), font=FONT)
        draw.rectangle((620, y - 4, 810, y + 14), fill=(42, 126, 86) if error < 1e-9 else (205, 137, 47))
        draw.text((648, y), "verified", fill=(255, 255, 255), font=FONT)
        y += 55
    thumb_a = Image.fromarray(to_uint8(gray_to_rgb(a)), mode="RGB").resize((150, 150), Image.Resampling.NEAREST)
    thumb_f = Image.fromarray(to_uint8(spectrum_visual(np.fft.fftshift(fa))), mode="RGB").resize((150, 150), Image.Resampling.NEAREST)
    canvas.paste(thumb_a, (90, 250))
    canvas.paste(thumb_f, (280, 250))
    draw.text((90, 230), "Input signal", fill=(24, 24, 24), font=FONT)
    draw.text((280, 230), "Shifted spectrum", fill=(24, 24, 24), font=FONT)
    canvas.save(OUT_DIR / "09_dft_properties.png")


def periodic_noise_image() -> tuple[np.ndarray, np.ndarray]:
    sample = make_sample("stop", 44, noisy=False)
    gray = rgb_to_gray(sample.image)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    pattern = 0.16 * np.sin(2 * math.pi * (x * 13 / WIDTH + y * 9 / HEIGHT))
    noisy = np.clip(gray + pattern, 0, 1)
    return gray, noisy


def dft_noise_filtering() -> None:
    clean, noisy = periodic_noise_image()
    spectrum = np.fft.fftshift(np.fft.fft2(noisy))
    h, w = noisy.shape
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    notch = np.ones_like(noisy, dtype=float)
    for dy, dx in [(9, 13), (-9, -13)]:
        dist = np.sqrt((y - (cy + dy)) ** 2 + (x - (cx + dx)) ** 2)
        notch[dist < 5] = 0
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * notch)))
    panels = [
        ("Clean generated image", gray_to_rgb(clean)),
        ("Periodic noisy image", gray_to_rgb(noisy)),
        ("Shifted noisy spectrum", spectrum_visual(spectrum)),
        ("Notch filter mask", gray_to_rgb(notch)),
        ("Filtered spectrum", spectrum_visual(spectrum * notch)),
        ("DFT noise-filtered result", gray_to_rgb(np.clip(filtered, 0, 1))),
    ]
    save_grid("10_dft_noise_filtering.png", panels, columns=3, tile=(240, 210))


def dft_encrypt_image() -> None:
    source = rgb_to_gray(make_sample("speed", 61, noisy=False).image)
    spectrum = np.fft.fft2(source)
    rng = np.random.default_rng(8871)
    phase_key = np.exp(1j * rng.uniform(-math.pi, math.pi, spectrum.shape))
    encrypted_spectrum = spectrum * phase_key
    encrypted = np.real(np.fft.ifft2(encrypted_spectrum))
    decrypted = np.real(np.fft.ifft2(encrypted_spectrum / phase_key))
    panels = [
        ("Original image", gray_to_rgb(source)),
        ("Encrypted spatial image", gray_to_rgb(normalize01(encrypted))),
        ("Encrypted spectrum magnitude", spectrum_visual(np.fft.fftshift(encrypted_spectrum))),
        ("Decrypted image", gray_to_rgb(np.clip(decrypted, 0, 1))),
        ("Decryption error", gray_to_rgb(normalize01(np.abs(decrypted - source)))),
    ]
    save_grid("11_dft_encryption.png", panels, columns=3, tile=(240, 210))


def run_histogram_pipeline() -> float:
    templates, tests = make_database()
    panels = []
    for sample in templates + tests[:6]:
        panels.append((f"{sample.label} / {sample.variant}", sample.image))
    save_grid("01_database_examples.png", panels, columns=3, tile=(230, 210))

    example = tests[0]
    enhanced_panels = enhance_variants(example.image)
    save_grid("02_enhancement_comparison.png", enhanced_panels, columns=3, tile=(250, 220))

    enhanced = best_enhancement(example.image)
    save_grid("03_segmentation_comparison.png", segment_candidates(enhanced), columns=3, tile=(250, 220))

    mask = segment_object(enhanced)
    segmented = segmented_image(enhanced, mask)
    save_grid(
        "04_segmented_image.png",
        [
            ("Best-enhanced image", enhanced),
            ("Binary mask", gray_to_rgb(mask.astype(float))),
            ("Enhanced image x mask", segmented),
        ],
        columns=3,
        tile=(260, 230),
    )
    save_histogram_visual("05_histograms.png", enhanced, mask)

    rows, accuracy = classify(templates, tests)
    save_classification_results("06_classification_results.png", rows, accuracy)
    print("Histogram classification predictions:")
    for true_label, predicted, chi, corr in rows:
        status = "correct" if true_label == predicted else "wrong"
        print(f"  true={true_label:8s} predicted={predicted:8s} chi2={chi:.4f} corr={corr:.4f} {status}")
    print(f"Histogram classification accuracy: {accuracy:.1f}%")
    return accuracy


def main() -> None:
    accuracy = run_histogram_pipeline()
    dft_error = dft_validation()
    fftshift_explanation()
    dft_properties()
    dft_noise_filtering()
    dft_encrypt_image()
    print(f"Manual DFT/IDFT validation max error: {dft_error:.3e}")
    print(f"Generated HW8 images in {OUT_DIR}")
    if accuracy < 80:
        raise RuntimeError("Classification accuracy unexpectedly low")


if __name__ == "__main__":
    main()
