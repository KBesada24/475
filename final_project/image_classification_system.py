from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "generated_images"
OUT_DIR = ROOT / "output"
SIZE = 192
CLASSES = ("coin", "leaf", "car", "traffic_sign")
FONT = ImageFont.load_default()


@dataclass
class Sample:
    label: str
    index: int
    quality: str
    image: np.ndarray
    truth_mask: np.ndarray
    filename: str


@dataclass
class Processed:
    sample: Sample
    enhanced: np.ndarray
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    segmented: np.ndarray
    lbp: np.ndarray
    features: dict[str, float]


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722])


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0, 1)[..., None], 3, axis=2)


def normalize01(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image), mode="RGB").save(path)


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
    for channel in range(3):
        low, high = np.percentile(image[..., channel], [2, 98])
        result[..., channel] = np.clip((image[..., channel] - low) / max(high - low, 1e-6), 0, 1)
    return result


def histogram_equalize_gray(gray: np.ndarray) -> np.ndarray:
    values = to_uint8(gray_to_rgb(gray))[..., 0]
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    cdf = hist.cumsum()
    cdf = (cdf - cdf[cdf > 0][0]) / max(cdf[-1] - cdf[cdf > 0][0], 1.0)
    return cdf[values]


def histogram_equalize_rgb(image: np.ndarray) -> np.ndarray:
    gray_eq = histogram_equalize_gray(rgb_to_gray(image))
    gray = rgb_to_gray(image)
    scale = gray_eq / np.maximum(gray, 1e-4)
    return np.clip(image * scale[..., None], 0, 1)


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


def background(rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    u = x / (SIZE - 1)
    v = y / (SIZE - 1)
    base = np.dstack([0.70 + 0.08 * u, 0.72 + 0.06 * v, 0.68 + 0.05 * (1 - u)])
    texture = 0.018 * np.sin(2 * math.pi * (u * 7 + v * 5))
    return np.clip(base + texture[..., None] + rng.normal(0, 0.01, (SIZE, SIZE, 3)), 0, 1)


def ellipse_mask(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


def polygon_mask(points: list[tuple[float, float]]) -> np.ndarray:
    image = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(image).polygon(points, fill=255)
    return np.asarray(image) > 0


def rectangle_mask(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    return mask


def add_object_texture(image: np.ndarray, mask: np.ndarray, color: np.ndarray, texture: np.ndarray) -> np.ndarray:
    result = image.copy()
    result[mask] = np.clip(color + texture[..., None], 0, 1)[mask]
    return result


def make_coin(index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = background(rng)
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    cx = SIZE * (0.50 + rng.normal(0, 0.025))
    cy = SIZE * (0.51 + rng.normal(0, 0.02))
    radius = SIZE * (0.29 + rng.normal(0, 0.012))
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    texture = 0.04 * np.sin(radial * 24 + index) + 0.05 * (1 - radial)
    image = add_object_texture(image, mask, np.array([0.84, 0.63, 0.20]), texture)
    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(box, outline=(92, 64, 18, 230), width=4)
    draw.ellipse((cx - radius * 0.68, cy - radius * 0.68, cx + radius * 0.68, cy + radius * 0.68), outline=(255, 220, 90, 150), width=3)
    draw.text((cx - 9, cy - 8), "$", fill=(74, 50, 14, 200), font=FONT)
    return np.asarray(canvas).astype(float) / 255, mask


def make_leaf(index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = background(rng)
    cx = SIZE * (0.50 + rng.normal(0, 0.025))
    cy = SIZE * (0.50 + rng.normal(0, 0.025))
    points = [(cx, cy - 68), (cx + 50, cy - 23), (cx + 38, cy + 25), (cx, cy + 68), (cx - 42, cy + 28), (cx - 50, cy - 20)]
    mask = polygon_mask(points)
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    veins = 0.03 * np.sin((x - cx) * 0.22 + (y - cy) * 0.15 + index)
    gradient = 0.05 * (1 - np.abs(x - cx) / 72)
    image = add_object_texture(image, mask, np.array([0.18, 0.58, 0.25]), veins + gradient)
    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.line((cx, cy - 60, cx, cy + 61), fill=(222, 244, 156, 180), width=2)
    for offset in (-42, -25, -8, 10, 28):
        draw.line((cx, cy + offset, cx + 35, cy + offset - 19), fill=(216, 239, 151, 115), width=1)
        draw.line((cx, cy + offset, cx - 35, cy + offset - 19), fill=(216, 239, 151, 115), width=1)
    return np.asarray(canvas).astype(float) / 255, mask


def make_car(index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = background(rng)
    body = rectangle_mask(39 + rng.integers(-4, 4), 88 + rng.integers(-3, 3), 153 + rng.integers(-4, 4), 132 + rng.integers(-2, 3))
    cabin = polygon_mask([(65, 88), (86, 56), (123, 56), (143, 88)])
    wheels = ellipse_mask(70, 132, 14, 14) | ellipse_mask(128, 132, 14, 14)
    mask = body | cabin | wheels
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    image = add_object_texture(image, body | cabin, np.array([0.66, 0.13, 0.15]), 0.025 * np.sin(x * 0.35 + index) + 0.05 * np.exp(-((y - 82) ** 2) / 300))
    image[wheels] = np.array([0.06, 0.06, 0.07])
    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.polygon([(68, 84), (88, 60), (121, 60), (139, 84)], fill=(145, 207, 226, 210), outline=(70, 42, 44, 180))
    draw.rectangle((45, 91, 150, 125), outline=(74, 32, 31, 220), width=3)
    draw.ellipse((56, 118, 84, 146), fill=(26, 27, 30, 255))
    draw.ellipse((114, 118, 142, 146), fill=(26, 27, 30, 255))
    draw.ellipse((64, 126, 76, 138), fill=(158, 160, 166, 255))
    draw.ellipse((122, 126, 134, 138), fill=(158, 160, 166, 255))
    return np.asarray(canvas).astype(float) / 255, mask


def make_traffic_sign(index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = background(rng)
    cx = SIZE * (0.50 + rng.normal(0, 0.02))
    cy = SIZE * (0.49 + rng.normal(0, 0.02))
    radius = SIZE * 0.31
    points = [(cx + radius * math.cos(math.pi / 8 + i * math.pi / 4), cy + radius * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]
    sign = polygon_mask(points)
    pole = rectangle_mask(cx - 4, cy + radius * 0.75, cx + 4, SIZE - 22)
    mask = sign | pole
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    image = add_object_texture(image, sign, np.array([0.80, 0.05, 0.05]), 0.02 * np.sin(x * 0.25 + y * 0.08 + index))
    image[pole] = np.array([0.46, 0.46, 0.43])
    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.polygon(points, outline=(245, 245, 240, 255), width=6)
    bbox = draw.textbbox((0, 0), "STOP", font=FONT)
    draw.text((cx - (bbox[2] - bbox[0]) / 2, cy - 5), "STOP", fill=(255, 255, 255, 240), font=FONT)
    return np.asarray(canvas).astype(float) / 255, mask


def degrade_image(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, index: int) -> np.ndarray:
    degraded = image.copy()
    if index % 3 == 0:
        degraded = convolve_rgb(degraded, gaussian_kernel(5, 1.2))
    degraded = np.clip(degraded + rng.normal(0, 0.045, degraded.shape), 0, 1)
    salt = rng.random(degraded.shape[:2])
    degraded[salt > 0.992] = 1
    degraded[salt < 0.008] = 0
    degraded = np.clip((degraded - 0.5) * 0.72 + 0.5, 0, 1)
    if index % 2 == 1:
        ys, xs = np.where(mask)
        y0 = int(np.percentile(ys, 40))
        x0 = int(np.percentile(xs, 52))
        degraded[y0 : y0 + 20, x0 : x0 + 34] = np.array([0.42, 0.42, 0.39])
    return degraded


def generate_samples() -> list[Sample]:
    IMAGE_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    generators = {"coin": make_coin, "leaf": make_leaf, "car": make_car, "traffic_sign": make_traffic_sign}
    samples = []
    for class_index, label in enumerate(CLASSES):
        for index in range(12):
            rng = np.random.default_rng(47500 + class_index * 1000 + index)
            image, mask = generators[label](index, rng)
            quality = "clean" if index < 6 else "noisy"
            if quality == "noisy":
                image = degrade_image(image, mask, rng, index)
            filename = f"{label}_{quality}_{index + 1:02d}.png"
            save_rgb(IMAGE_DIR / filename, image)
            samples.append(Sample(label, index, quality, image, mask, filename))
    return samples


def restoration_candidates(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    median = median_filter(image, 1)
    gaussian = convolve_rgb(image, gaussian_kernel(5, 1.1))
    unsharp = np.clip(median + 1.25 * (median - convolve_rgb(median, gaussian_kernel(5, 1.0))), 0, 1)
    stretched = contrast_stretch(unsharp)
    equalized = histogram_equalize_rgb(stretched)
    return [
        ("Original", image),
        ("Median denoising", median),
        ("Gaussian smoothing", gaussian),
        ("Unsharp enhancement", unsharp),
        ("Contrast stretch", stretched),
        ("Histogram equalization", equalized),
    ]


def edge_strength(gray: np.ndarray) -> float:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    mag = np.sqrt(convolve(gray, sobel_x) ** 2 + convolve(gray, sobel_y) ** 2)
    return float(np.mean(mag))


def quality_score(image: np.ndarray) -> float:
    gray = rgb_to_gray(image)
    contrast = float(np.std(gray))
    edges = edge_strength(gray)
    noise_penalty = float(np.mean(np.abs(gray - convolve(gray, gaussian_kernel(5, 1.0)))))
    return contrast + 0.55 * edges - 0.35 * noise_penalty


def enhance_best(image: np.ndarray) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    candidates = restoration_candidates(image)
    best_name, best_image = max(candidates[1:], key=lambda item: quality_score(item[1]))
    return best_image, candidates + [(f"Selected: {best_name}", best_image)]


def segment_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = rgb_to_gray(image)
    border = np.concatenate([image[:12].reshape(-1, 3), image[-12:].reshape(-1, 3), image[:, :12].reshape(-1, 3), image[:, -12:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    color_distance = np.linalg.norm(image - bg, axis=2)
    saturation = image.max(axis=2) - image.min(axis=2)
    dark_object = gray < np.percentile(gray, 28)
    raw = (color_distance > np.percentile(color_distance, 70)) | ((saturation > 0.12) & (color_distance > 0.12)) | dark_object
    raw[:4, :] = raw[-4:, :] = raw[:, :4] = raw[:, -4:] = False
    clean = closing(opening(raw, 2), 3)
    clean = largest_component(clean)
    clean = closing(clean, 2)
    return raw, clean


def local_binary_pattern(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = [gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2]]
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    padded = np.zeros(gray.shape, dtype=np.uint8)
    padded[1:-1, 1:-1] = code
    return padded


def glcm_features(levels_image: np.ndarray, mask: np.ndarray, levels: int = 16) -> dict[str, float]:
    glcm = np.zeros((levels, levels), dtype=float)
    h, w = levels_image.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (-1, 1)):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        src = levels_image[y0:y1, x0:x1]
        dst = levels_image[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        valid = mask[y0:y1, x0:x1] & mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        for i, j in zip(src[valid].ravel(), dst[valid].ravel()):
            glcm[int(i), int(j)] += 1
            glcm[int(j), int(i)] += 1
    if glcm.sum() == 0:
        return {"contrast": 0.0, "energy": 0.0, "homogeneity": 0.0, "correlation": 0.0}
    p = glcm / glcm.sum()
    i, j = np.indices(p.shape)
    contrast = float(np.sum((i - j) ** 2 * p))
    energy = float(np.sum(p * p))
    homogeneity = float(np.sum(p / (1 + np.abs(i - j))))
    mean_i = float(np.sum(i * p))
    mean_j = float(np.sum(j * p))
    std_i = float(np.sqrt(np.sum(((i - mean_i) ** 2) * p)))
    std_j = float(np.sqrt(np.sum(((j - mean_j) ** 2) * p)))
    correlation = float(np.sum((i - mean_i) * (j - mean_j) * p) / (std_i * std_j + 1e-9))
    return {"contrast": contrast, "energy": energy, "homogeneity": homogeneity, "correlation": correlation}


def normalized_hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(float)
    return hist / max(hist.sum(), 1.0)


def boundary_points(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1]
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, :-2] & padded[:-2, 2:] & padded[2:, :-2] & padded[2:, 2:]
    return np.argwhere(core & ~interior)


def chain_code_histogram(mask: np.ndarray) -> np.ndarray:
    points = boundary_points(mask)
    if len(points) < 2:
        return np.zeros(8)
    center = points.mean(axis=0)
    ordered = points[np.argsort(np.arctan2(points[:, 0] - center[0], points[:, 1] - center[1]))]
    deltas = np.diff(np.vstack([ordered, ordered[0]]), axis=0)
    dirs = {(0, 1): 0, (-1, 1): 1, (-1, 0): 2, (-1, -1): 3, (0, -1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7}
    hist = np.zeros(8, dtype=float)
    for dy, dx in deltas:
        hist[dirs.get((int(np.sign(dy)), int(np.sign(dx))), 0)] += 1
    return hist / max(hist.sum(), 1.0)


def box_counting_dimension(mask: np.ndarray) -> float:
    sizes = np.array([2, 4, 8, 16, 32])
    counts = []
    for size in sizes:
        h = (mask.shape[0] // size) * size
        w = (mask.shape[1] // size) * size
        blocks = mask[:h, :w].reshape(h // size, size, w // size, size)
        counts.append(np.count_nonzero(blocks.any(axis=(1, 3))))
    counts = np.maximum(np.array(counts, dtype=float), 1)
    slope, _ = np.polyfit(np.log(1 / sizes), np.log(counts), 1)
    return float(slope)


def projection_features(mask: np.ndarray, bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    horizontal = mask.sum(axis=1).astype(float)
    vertical = mask.sum(axis=0).astype(float)
    h_chunks = np.array_split(horizontal, bins)
    v_chunks = np.array_split(vertical, bins)
    h = np.array([chunk.mean() for chunk in h_chunks]) / mask.shape[1]
    v = np.array([chunk.mean() for chunk in v_chunks]) / mask.shape[0]
    return h, v


def extract_features(sample: Sample, enhanced: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    gray = rgb_to_gray(enhanced)
    lbp = local_binary_pattern(gray)
    features: dict[str, float] = {
        "area_ratio": float(mask.mean()),
        "box_counting_dimension": box_counting_dimension(mask),
        "aspect_ratio": float((np.ptp(np.where(mask)[1]) + 1) / max(np.ptp(np.where(mask)[0]) + 1, 1)) if mask.any() else 0.0,
    }
    hproj, vproj = projection_features(mask)
    for i, value in enumerate(hproj):
        features[f"projection_horizontal_{i:02d}"] = float(value)
    for i, value in enumerate(vproj):
        features[f"projection_vertical_{i:02d}"] = float(value)
    for i, value in enumerate(chain_code_histogram(mask)):
        features[f"chain_code_{i}"] = float(value)
    for i, value in enumerate(normalized_hist(lbp[mask], 16, (0, 256))):
        features[f"lbp_hist_{i:02d}"] = float(value)
    for channel, name in enumerate(("red", "green", "blue")):
        values = enhanced[..., channel][mask]
        for i, value in enumerate(normalized_hist(values, 8, (0, 1))):
            features[f"{name}_hist_{i:02d}"] = float(value)
        levels = np.clip(local_binary_pattern(enhanced[..., channel]) // 16, 0, 15)
        for key, value in glcm_features(levels, mask).items():
            features[f"haralick_{name}_{key}"] = value
    for i, value in enumerate(normalized_hist(gray[mask], 12, (0, 1))):
        features[f"intensity_hist_{i:02d}"] = float(value)
    return features, lbp


def process_sample(sample: Sample) -> tuple[Processed, list[tuple[str, np.ndarray]]]:
    enhanced, candidates = enhance_best(sample.image)
    raw_mask, clean_mask = segment_image(enhanced)
    segmented = enhanced * clean_mask[..., None]
    features, lbp = extract_features(sample, enhanced, clean_mask)
    return Processed(sample, enhanced, raw_mask, clean_mask, segmented, lbp, features), candidates


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.dstack([mask * 0.95, mask * 0.85, mask * 0.18]).astype(float)


def lbp_to_rgb(lbp: np.ndarray) -> np.ndarray:
    normalized = lbp.astype(float) / 255
    return np.dstack([normalized, np.sqrt(normalized), 1 - normalized * 0.45])


def save_grid(path: Path, panels: list[tuple[str, np.ndarray]], columns: int = 4, tile: tuple[int, int] = (230, 210)) -> None:
    tile_w, tile_h = tile
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), (246, 245, 238))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row, col = divmod(index, columns)
        x, y = col * tile_w, row * tile_h
        panel = Image.fromarray(to_uint8(image), mode="RGB").resize((tile_w, tile_h - 28), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, y + 28))
        draw.rectangle((x, y, x + tile_w, y + 28), fill=(31, 37, 43))
        draw.text((x + 8, y + 8), label[:36], fill=(248, 244, 232), font=FONT)
    canvas.save(path)


def save_dataset_sheet(samples: list[Sample]) -> None:
    panels = [(f"{s.label} / {s.quality}", s.image) for s in samples]
    save_grid(OUT_DIR / "01_dataset_contact_sheet.png", panels, columns=6, tile=(170, 196))


def save_restoration_sheet(candidates: list[tuple[str, np.ndarray]]) -> None:
    save_grid(OUT_DIR / "02_restoration_enhancement.png", candidates, columns=4, tile=(235, 210))


def save_segmentation_sheet(processed: Processed) -> None:
    panels = [
        ("Best enhanced image", processed.enhanced),
        ("Raw foreground mask", mask_to_rgb(processed.raw_mask)),
        ("Opening/closing + largest object", mask_to_rgb(processed.clean_mask)),
        ("Segmented image = enhanced x mask", processed.segmented),
    ]
    save_grid(OUT_DIR / "03_segmentation_pipeline.png", panels, columns=4, tile=(235, 210))


def save_projection_image(processed: Processed) -> None:
    hproj, vproj = projection_features(processed.clean_mask)
    width, height = 760, 420
    canvas = Image.new("RGB", (width, height), (247, 246, 240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 36), fill=(31, 37, 43))
    draw.text((16, 12), "Binary object feature extraction: mask projections", fill=(248, 244, 232), font=FONT)
    mask_img = Image.fromarray(to_uint8(mask_to_rgb(processed.clean_mask)), mode="RGB").resize((240, 240), Image.Resampling.NEAREST)
    canvas.paste(mask_img, (55, 85))
    draw.text((118, 335), "Binary object mask", fill=(31, 37, 43), font=FONT)
    base_x, base_y = 380, 310
    bar_w = 14
    for i, value in enumerate(hproj):
        x = base_x + i * (bar_w + 4)
        draw.rectangle((x, base_y - int(value * 190), x + bar_w, base_y), fill=(35, 112, 164))
    draw.text((425, 335), "Horizontal projection", fill=(31, 37, 43), font=FONT)
    base_x, base_y = 380, 160
    for i, value in enumerate(vproj):
        y = base_y + i * 12
        draw.rectangle((base_x, y, base_x + int(value * 260), y + 8), fill=(184, 64, 62))
    draw.text((425, 70), "Vertical projection", fill=(31, 37, 43), font=FONT)
    canvas.save(OUT_DIR / "04_binary_projection_features.png")


def save_feature_concatenation(processed: Processed, feature_names: list[str]) -> None:
    width, height = 980, 470
    canvas = Image.new("RGB", (width, height), (247, 246, 240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 40), fill=(31, 37, 43))
    draw.text((16, 14), "Feature concatenation for SVM classification", fill=(248, 244, 232), font=FONT)
    source = Image.fromarray(to_uint8(processed.segmented), mode="RGB").resize((150, 150), Image.Resampling.LANCZOS)
    lbp_img = Image.fromarray(to_uint8(lbp_to_rgb(processed.lbp)), mode="RGB").resize((150, 150), Image.Resampling.LANCZOS)
    canvas.paste(source, (45, 95))
    canvas.paste(lbp_img, (260, 95))
    draw.text((82, 260), "Segmented image", fill=(31, 37, 43), font=FONT)
    draw.text((298, 260), "LBP texture", fill=(31, 37, 43), font=FONT)
    groups = [
        ("Projections", 32, (45, 330), (46, 124, 74)),
        ("Haralick", 12, (250, 330), (160, 85, 165)),
        ("Color + intensity histograms", 36, (440, 330), (214, 158, 49)),
        ("Box count + chain code", 9, (710, 330), (184, 64, 62)),
    ]
    for title, count, (x0, y0), color in groups:
        draw.text((x0, y0 - 24), title, fill=(31, 37, 43), font=FONT)
        for i in range(count):
            x = x0 + i * 6
            h = 12 + (i * 17 % 58)
            draw.rectangle((x, y0 + 70 - h, x + 4, y0 + 70), fill=color)
    draw.line((205, 170, 250, 170), fill=(31, 37, 43), width=2)
    draw.line((420, 170, 770, 170), fill=(31, 37, 43), width=2)
    draw.rectangle((760, 125, 925, 215), outline=(31, 37, 43), width=2)
    draw.text((785, 145), f"Concatenated vector", fill=(31, 37, 43), font=FONT)
    draw.text((800, 170), f"{len(feature_names)} features", fill=(31, 37, 43), font=FONT)
    draw.text((804, 195), "input to SVM", fill=(31, 37, 43), font=FONT)
    canvas.save(OUT_DIR / "05_feature_concatenation.png")


def save_classification_montage(processed: list[Processed], predictions: dict[str, str], test_files: set[str]) -> None:
    panels = []
    for item in processed:
        if item.sample.filename not in test_files:
            continue
        img = Image.fromarray(to_uint8(item.sample.image), mode="RGB")
        draw = ImageDraw.Draw(img)
        pred = predictions[item.sample.filename]
        color = (30, 122, 62) if pred == item.sample.label else (185, 45, 42)
        draw.rectangle((0, 0, SIZE, 24), fill=color)
        draw.text((6, 8), f"{item.sample.label} -> {pred}", fill=(255, 255, 255), font=FONT)
        panels.append((item.sample.filename, np.asarray(img).astype(float) / 255))
    save_grid(OUT_DIR / "06_classification_results.png", panels, columns=4, tile=(220, 210))


def save_feature_summary(rows: list[dict[str, object]]) -> None:
    metrics = [
        ("Area", "area_ratio"),
        ("Box dim.", "box_counting_dimension"),
        ("H proj 7", "projection_horizontal_07"),
        ("LBP high", "lbp_hist_15"),
        ("R contrast", "haralick_red_contrast"),
        ("Intensity 6", "intensity_hist_06"),
    ]
    values = {label: [np.mean([float(row[key]) for row in rows if row["label"] == label]) for _, key in metrics] for label in CLASSES}
    maxima = [max(values[label][i] for label in CLASSES) or 1 for i in range(len(metrics))]
    width, height = 980, 430
    canvas = Image.new("RGB", (width, height), (247, 246, 240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 40), fill=(31, 37, 43))
    draw.text((16, 14), "Class-level averages from concatenated feature vectors", fill=(248, 244, 232), font=FONT)
    colors = {"coin": (210, 158, 47), "leaf": (62, 151, 73), "car": (185, 50, 52), "traffic_sign": (44, 120, 180)}
    left, baseline, scale_h, group_w, bar_w = 70, 330, 220, 140, 22
    for metric_i, (metric_label, _key) in enumerate(metrics):
        gx = left + metric_i * group_w
        draw.text((gx, baseline + 24), metric_label, fill=(31, 37, 43), font=FONT)
        for class_i, label in enumerate(CLASSES):
            value = values[label][metric_i]
            x0 = gx + class_i * (bar_w + 6)
            y0 = baseline - int((value / maxima[metric_i]) * scale_h)
            draw.rectangle((x0, y0, x0 + bar_w, baseline), fill=colors[label])
    for i, label in enumerate(CLASSES):
        y = 80 + i * 26
        draw.rectangle((850, y, 868, y + 18), fill=colors[label])
        draw.text((878, y + 4), label, fill=(31, 37, 43), font=FONT)
    canvas.save(OUT_DIR / "07_feature_summary.png")


def write_features_csv(rows: list[dict[str, object]], feature_names: list[str]) -> None:
    with (OUT_DIR / "features.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "label", "quality", "split", "prediction"] + feature_names)
        writer.writeheader()
        writer.writerows(rows)


def train_and_report(processed: list[Processed], rows: list[dict[str, object]], feature_names: list[str]) -> tuple[dict[str, str], set[str]]:
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.array([str(row["label"]) for row in rows])
    filenames = np.array([str(row["filename"]) for row in rows])
    x_train, x_test, y_train, y_test, files_train, files_test = train_test_split(x, y, filenames, test_size=0.34, random_state=475, stratify=y)
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=8.0, gamma="scale"))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_pred = model.predict(x_train)
    predictions = {file: pred for file, pred in zip(files_test, y_pred)}
    predictions.update({file: pred for file, pred in zip(files_train, train_pred)})
    test_files = set(files_test.tolist())
    for row in rows:
        row["split"] = "test" if row["filename"] in test_files else "train"
        row["prediction"] = predictions[str(row["filename"])]
    correct = int(np.sum(y_pred == y_test))
    total = len(y_test)
    accuracy = 100 * correct / total
    matrix = confusion_matrix(y_test, y_pred, labels=list(CLASSES))
    report = classification_report(y_test, y_pred, labels=list(CLASSES), zero_division=0)
    lines = [
        "Final Project: Image Classification System",
        "==========================================",
        "",
        "Pipeline:",
        "1. Image database assembly: 48 generated images, four classes, clean and noisy/degraded samples.",
        "2. Image restoration/enhancement: median denoising, Gaussian smoothing, unsharp masking, contrast stretching, and histogram equalization.",
        "3. Image segmentation: color-distance foreground thresholding, grayscale thresholding, morphological opening/closing, and largest connected object selection.",
        "4. Binary object extraction: selected enhanced image multiplied by the binary mask.",
        "5. Binary object features: horizontal and vertical projections.",
        "6. Concatenated feature vector: projections, Haralick texture, box-counting dimension, chain-code histogram, LBP histogram, RGB histograms, and intensity histogram.",
        "7. Classifier: scikit-learn SVM, SVC(kernel='rbf') inside a StandardScaler pipeline.",
        "",
        f"Test accuracy: {accuracy:.2f}% ({correct}/{total})",
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        "actual/predicted," + ",".join(CLASSES),
    ]
    for label, row in zip(CLASSES, matrix):
        lines.append(label + "," + ",".join(str(int(value)) for value in row))
    lines.extend(["", "Class-level precision/recall/F1:", report, "Per-image predictions:"])
    for row in rows:
        lines.append(f"{row['filename']}: split={row['split']}, actual={row['label']}, predicted={row['prediction']}")
    lines.extend(
        [
            "",
            "Performance discussion:",
            "The generated classes are intentionally separated by both shape and texture, so the concatenated vector gives the SVM multiple independent cues. Noisy images are restored before segmentation, which keeps binary masks stable enough for projections, box counting, and chain-code features.",
            "",
            "Potential improvements:",
            "For real images, the same system could be adapted by replacing the generated database with labeled folders, adding data augmentation, tuning SVM C/gamma by cross-validation, and using more robust segmentation for cluttered backgrounds.",
        ]
    )
    (OUT_DIR / "classification_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return predictions, test_files


def main() -> None:
    samples = generate_samples()
    processed: list[Processed] = []
    restoration_example: list[tuple[str, np.ndarray]] | None = None
    for sample in samples:
        item, candidates = process_sample(sample)
        processed.append(item)
        if sample.quality == "noisy" and restoration_example is None:
            restoration_example = candidates

    rows: list[dict[str, object]] = []
    for item in processed:
        row: dict[str, object] = {
            "filename": item.sample.filename,
            "label": item.sample.label,
            "quality": item.sample.quality,
            "split": "",
            "prediction": "",
        }
        row.update(item.features)
        rows.append(row)
    feature_names = [key for key in rows[0].keys() if key not in {"filename", "label", "quality", "split", "prediction"}]

    predictions, test_files = train_and_report(processed, rows, feature_names)
    write_features_csv(rows, feature_names)

    save_dataset_sheet(samples)
    save_restoration_sheet(restoration_example or restoration_candidates(samples[0].image))
    save_segmentation_sheet(next(item for item in processed if item.sample.quality == "noisy"))
    save_projection_image(next(item for item in processed if item.sample.quality == "noisy"))
    save_feature_concatenation(processed[0], feature_names)
    save_classification_montage(processed, predictions, test_files)
    save_feature_summary(rows)

    correct = sum(row["label"] == row["prediction"] for row in rows if row["split"] == "test")
    total = sum(row["split"] == "test" for row in rows)
    print(f"Generated {len(samples)} images in {IMAGE_DIR}")
    print(f"Extracted {len(feature_names)} concatenated features per image")
    print(f"SVM test accuracy: {100 * correct / total:.2f}% ({correct}/{total})")
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
