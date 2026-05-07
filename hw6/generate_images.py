from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
WIDTH = 640
HEIGHT = 420
EPS = 1e-6


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def from_uint8(image: np.ndarray) -> np.ndarray:
    return image.astype(float) / 255.0


def save_rgb(name: str, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image), mode="RGB").save(OUT_DIR / name)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0, 1)[..., None], 3, axis=2)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722])


def panel_image(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    if image.ndim == 2:
        image = gray_to_rgb(image)
    return Image.fromarray(to_uint8(image), mode="RGB").resize(size, Image.Resampling.LANCZOS)


def save_grid(name: str, panels: list[tuple[str, np.ndarray]], columns: int = 2) -> None:
    tile_w, tile_h = 420, 292
    label_h = 30
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), (246, 244, 236))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, image) in enumerate(panels):
        row, col = divmod(index, columns)
        x = col * tile_w
        y = row * tile_h
        canvas.paste(panel_image(image, (tile_w, tile_h - label_h)), (x, y + label_h))
        draw.rectangle((x, y, x + tile_w, y + label_h), fill=(28, 36, 42))
        draw.text((x + 10, y + 9), label, fill=(246, 239, 219), font=font)

    canvas.save(OUT_DIR / name)


def draw_histogram(
    name: str,
    gray: np.ndarray,
    title: str,
    thresholds: list[float] | None = None,
    ranges: list[tuple[float, float, str, tuple[int, int, int]]] | None = None,
) -> None:
    thresholds = thresholds or []
    ranges = ranges or []
    w, h = 760, 360
    left, right, top, bottom = 58, 24, 36, 52
    plot_w = w - left - right
    plot_h = h - top - bottom
    hist, _ = np.histogram(np.clip(gray, 0, 1), bins=256, range=(0, 1))
    hist = hist.astype(float)
    hist = hist / max(hist.max(), 1)

    canvas = Image.new("RGB", (w, h), (250, 248, 241))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((left, 10), title, fill=(20, 28, 34), font=font)

    for lo, hi, label, color in ranges:
        x0 = left + int(lo * plot_w)
        x1 = left + int(hi * plot_w)
        draw.rectangle((x0, top, x1, top + plot_h), fill=tuple(int(c * 0.18 + 235 * 0.82) for c in color))
        draw.text((x0 + 4, top + 8), label, fill=color, font=font)

    draw.line((left, top, left, top + plot_h, left + plot_w, top + plot_h), fill=(35, 42, 48), width=2)
    for i, value in enumerate(hist):
        x = left + int(i / 255 * plot_w)
        y = top + plot_h - int(value * plot_h)
        draw.line((x, top + plot_h, x, y), fill=(66, 89, 111))

    for t in thresholds:
        x = left + int(np.clip(t, 0, 1) * plot_w)
        draw.line((x, top, x, top + plot_h), fill=(204, 58, 42), width=3)
        draw.text((x + 4, top + plot_h + 8), f"T={t:.2f}", fill=(204, 58, 42), font=font)

    for value, label in [(0, "0"), (0.5, "128"), (1, "255")]:
        x = left + int(value * plot_w)
        draw.line((x, top + plot_h, x, top + plot_h + 5), fill=(35, 42, 48))
        draw.text((x - 10, top + plot_h + 24), label, fill=(35, 42, 48), font=font)

    canvas.save(OUT_DIR / name)


def draw_curve(name: str, mapping: np.ndarray, title: str) -> None:
    w, h = 560, 420
    left, top, plot = 58, 38, 320
    canvas = Image.new("RGB", (w, h), (250, 248, 241))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((left, 12), title, fill=(24, 32, 38), font=font)
    draw.rectangle((left, top, left + plot, top + plot), outline=(30, 38, 45), width=2)
    draw.line((left, top + plot, left + plot, top), fill=(180, 185, 188), width=1)

    points = []
    for i, v in enumerate(mapping):
        x = left + int(i / 255 * plot)
        y = top + plot - int(np.clip(v, 0, 1) * plot)
        points.append((x, y))
    draw.line(points, fill=(196, 56, 42), width=3)
    draw.text((left + plot + 24, top + 6), "v = T(u)", fill=(196, 56, 42), font=font)
    draw.text((left + plot // 2 - 18, top + plot + 28), "input u", fill=(24, 32, 38), font=font)
    draw.text((12, top + plot // 2), "output v", fill=(24, 32, 38), font=font)
    canvas.save(OUT_DIR / name)


def generated_enhancement_image(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    rng = np.random.default_rng(606)
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    base = 0.20 + 0.33 * u + 0.16 * (1 - v)
    lighting = 0.28 * np.exp(-(((u - 0.25) / 0.30) ** 2 + ((v - 0.18) / 0.22) ** 2))
    texture = 0.035 * np.sin(35 * u) + 0.025 * np.cos(28 * v)
    gray = base + lighting + texture

    objects = [
        ((0.18, 0.53), (0.13, 0.20), -0.18),
        ((0.47, 0.44), (0.16, 0.12), 0.16),
        ((0.72, 0.60), (0.15, 0.18), -0.11),
    ]
    for (cx, cy), (rx, ry), delta in objects:
        mask = ((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2 < 1
        gray[mask] += delta
    gray += rng.normal(0, 0.025, gray.shape)
    return np.clip(gray, 0, 1)


def generated_color_image(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    rng = np.random.default_rng(607)
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    image = np.dstack(
        [
            0.26 + 0.34 * u + 0.08 * np.sin(12 * v),
            0.22 + 0.30 * (1 - v) + 0.05 * np.cos(15 * u),
            0.30 + 0.24 * v,
        ]
    )
    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((70, 230, 570, 398), fill=(72, 43, 24, 245))
    draw.ellipse((96, 118, 246, 294), fill=(214, 78, 52, 255))
    draw.rectangle((300, 92, 430, 308), fill=(54, 139, 194, 255))
    draw.ellipse((287, 70, 443, 136), fill=(75, 170, 218, 255))
    draw.ellipse((460, 176, 585, 315), fill=(226, 195, 56, 255))
    image = from_uint8(np.asarray(canvas))
    shade = 0.72 + 0.35 * np.exp(-(((u - 0.16) / 0.44) ** 2 + ((v - 0.10) / 0.34) ** 2))
    image = image * shade[..., None] + rng.normal(0, 0.018, image.shape)
    return np.clip(image, 0, 1)


def generated_segmentation_image(width: int = WIDTH, height: int = HEIGHT, modes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(700 + modes)
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    gray = 0.18 + 0.05 * u + 0.04 * np.sin(11 * v)
    if modes == 2:
        gray += 0.58 * (((u - 0.53) / 0.34) ** 2 + ((v - 0.50) / 0.28) ** 2 < 1)
    else:
        gray += 0.31 * (((u - 0.35) / 0.24) ** 2 + ((v - 0.49) / 0.23) ** 2 < 1)
        gray += 0.60 * (((u - 0.69) / 0.16) ** 2 + ((v - 0.45) / 0.20) ** 2 < 1)
    gray += rng.normal(0, 0.026, gray.shape)
    return np.clip(gray, 0, 1)


def generated_color_segmentation(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    rng = np.random.default_rng(760)
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    image = np.dstack([0.24 + 0.11 * u, 0.25 + 0.09 * v, 0.30 + 0.08 * (1 - u)])
    masks = [
        (((u - 0.25) / 0.14) ** 2 + ((v - 0.48) / 0.23) ** 2 < 1, np.array([0.82, 0.18, 0.12])),
        (((u - 0.52) / 0.15) ** 2 + ((v - 0.40) / 0.19) ** 2 < 1, np.array([0.20, 0.66, 0.28])),
        (((u - 0.76) / 0.12) ** 2 + ((v - 0.58) / 0.20) ** 2 < 1, np.array([0.90, 0.72, 0.15])),
    ]
    for mask, color in masks:
        image[mask] = color
    illumination = 0.68 + 0.34 * np.exp(-(((u - 0.10) / 0.55) ** 2 + ((v - 0.05) / 0.42) ** 2))
    image = image * illumination[..., None] + rng.normal(0, 0.018, image.shape)
    return np.clip(image, 0, 1)


def contrast_stretch(gray: np.ndarray, low_pct: float = 2, high_pct: float = 98) -> np.ndarray:
    lo, hi = np.percentile(gray, [low_pct, high_pct])
    return np.clip((gray - lo) / max(hi - lo, EPS), 0, 1)


def block_quality(gray: np.ndarray, block: int = 32, alpha: float = 0.65, c: float = 0.02) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    filtered = np.zeros_like(gray)
    scores = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = gray[y : y + block, x : x + block]
            imax = float(tile.max())
            imin = float(tile.min())
            ratio = ((imax + c) / (imin + c)) ** alpha
            inv_ratio = (imax + c) / (imin + c)
            scores.append(alpha * ratio * math.log(max(inv_ratio, 1 + EPS)))
            local = (tile - imin) / max(imax - imin, EPS)
            filtered[y : y + block, x : x + block] = np.clip(local ** alpha, 0, 1)
    return filtered, float(np.mean(scores))


def histogram_equalization(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = to_uint8(gray)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    equalized = cdf[values]
    return equalized, cdf


def gamma_correction(gray: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    return np.clip(gray, 0, 1) ** gamma


def color_enhancement(image: np.ndarray) -> np.ndarray:
    y = rgb_to_gray(image)
    y_eq, _ = histogram_equalization(contrast_stretch(y))
    ratio = y_eq / np.maximum(y, 0.04)
    enhanced = image * ratio[..., None]
    saturation_boost = 1.18
    gray = rgb_to_gray(enhanced)
    enhanced = gray[..., None] + saturation_boost * (enhanced - gray[..., None])
    return np.clip(enhanced, 0, 1)


def mean_histogram_stretch(gray: np.ndarray, strength: float = 1.75) -> np.ndarray:
    mean = float(gray.mean())
    std = float(gray.std())
    stretched = (gray - mean) * strength + 0.5
    clipped = np.clip(stretched, 0, 1)
    if std < 0.08:
        return contrast_stretch(clipped)
    return clipped


def iterative_threshold(gray: np.ndarray, initial: float, tol: float = 0.001, max_iter: int = 60) -> tuple[float, list[float]]:
    t = initial
    history = [t]
    for _ in range(max_iter):
        low = gray[gray <= t]
        high = gray[gray > t]
        if low.size == 0 or high.size == 0:
            break
        new_t = 0.5 * (float(low.mean()) + float(high.mean()))
        history.append(new_t)
        if abs(new_t - t) < tol:
            t = new_t
            break
        t = new_t
    return t, history


def multilevel_thresholds(gray: np.ndarray, initials: tuple[float, float], tol: float = 0.001) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    t1, t2 = sorted(initials)
    history = [(t1, t2)]
    for _ in range(60):
        low = gray[gray <= t1]
        mid = gray[(gray > t1) & (gray <= t2)]
        high = gray[gray > t2]
        if low.size == 0 or mid.size == 0 or high.size == 0:
            break
        m1, m2, m3 = float(low.mean()), float(mid.mean()), float(high.mean())
        nt1, nt2 = 0.5 * (m1 + m2), 0.5 * (m2 + m3)
        history.append((nt1, nt2))
        if max(abs(nt1 - t1), abs(nt2 - t2)) < tol:
            t1, t2 = nt1, nt2
            break
        t1, t2 = nt1, nt2
    return (t1, t2), history


def classify_multilevel(gray: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    t1, t2 = thresholds
    labels = np.zeros_like(gray)
    labels[(gray > t1) & (gray <= t2)] = 0.55
    labels[gray > t2] = 1.0
    return labels


def local_stats(gray: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    padded = np.pad(gray, radius, mode="reflect")
    mean = np.zeros_like(gray)
    std = np.zeros_like(gray)
    minv = np.zeros_like(gray)
    maxv = np.zeros_like(gray)
    h, w = gray.shape
    area = (2 * radius + 1) ** 2
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            window = padded[dy : dy + h, dx : dx + w]
            mean += window / area
            minv = window if (dy == 0 and dx == 0) else np.minimum(minv, window)
            maxv = window if (dy == 0 and dx == 0) else np.maximum(maxv, window)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            window = padded[dy : dy + h, dx : dx + w]
            std += (window - mean) ** 2 / area
    return mean, np.sqrt(std), minv, maxv


def adaptive_threshold(gray: np.ndarray, radius: int = 15, offset: float = 0.025) -> np.ndarray:
    mean, _std, _minv, _maxv = local_stats(gray, radius)
    return (gray > mean - offset).astype(float)


def niblack(gray: np.ndarray, radius: int = 13, k: float = -0.20) -> np.ndarray:
    mean, std, _minv, _maxv = local_stats(gray, radius)
    return (gray > mean + k * std).astype(float)


def bernsen(gray: np.ndarray, radius: int = 13, contrast_limit: float = 0.13) -> np.ndarray:
    _mean, _std, minv, maxv = local_stats(gray, radius)
    threshold = 0.5 * (minv + maxv)
    contrast = maxv - minv
    fallback = gray.mean()
    threshold = np.where(contrast < contrast_limit, fallback, threshold)
    return (gray > threshold).astype(float)


def sauvola(gray: np.ndarray, radius: int = 13, k: float = 0.34, r: float = 0.5) -> np.ndarray:
    mean, std, _minv, _maxv = local_stats(gray, radius)
    threshold = mean * (1 + k * ((std / r) - 1))
    return (gray > threshold).astype(float)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    if image.ndim == 2:
        image = gray_to_rgb(image)
    color_arr = np.array(color)
    return np.where(mask[..., None] > 0, image * 0.45 + color_arr * 0.55, image)


def histogram_peak_ranges(gray: np.ndarray) -> tuple[list[tuple[float, float, str, tuple[int, int, int]]], np.ndarray]:
    ranges = [
        (0.00, 0.33, "background", (72, 92, 118)),
        (0.33, 0.58, "object A", (45, 146, 87)),
        (0.58, 1.00, "object B", (200, 72, 48)),
    ]
    labels = np.zeros_like(gray)
    labels[(gray >= 0.33) & (gray < 0.58)] = 0.55
    labels[gray >= 0.58] = 1
    return ranges, labels


def draw_object_boxes(name: str, image: np.ndarray, labels: np.ndarray) -> None:
    rgb = gray_to_rgb(image) * 0.62
    palette = [np.array([0.16, 0.72, 0.42]), np.array([0.94, 0.35, 0.22])]
    canvas = Image.fromarray(to_uint8(rgb), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, value in enumerate([0.55, 1.0]):
        mask = np.isclose(labels, value)
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        color = tuple((palette[index] * 255).astype(int))
        overlay = np.asarray(canvas).astype(float) / 255
        overlay[mask] = overlay[mask] * 0.45 + palette[index] * 0.55
        canvas = Image.fromarray(to_uint8(overlay), mode="RGB")
        draw = ImageDraw.Draw(canvas)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        draw.text((x0 + 8, max(4, y0 - 18)), f"object {index + 1}", fill=color, font=font)
    canvas.save(OUT_DIR / name)


def enhancement_outputs() -> None:
    gray = generated_enhancement_image()
    color = generated_color_image()
    stretched = contrast_stretch(gray)
    quality_filtered, q_before = block_quality(gray)
    _q_after_img, q_after = block_quality(stretched)
    equalized, mapping = histogram_equalization(gray)
    gamma = gamma_correction(gray)
    color_plus = color_enhancement(color)
    spatial = mean_histogram_stretch(gamma)

    save_rgb("01_enhancement_source_gray.png", gray_to_rgb(gray))
    save_rgb("02_enhancement_source_color.png", color)
    save_grid(
        "03_contrast_stretching.png",
        [
            ("Original low contrast image f", gray),
            ("Contrast stretched result", stretched),
            ("Difference amplified", gray_to_rgb(np.abs(stretched - gray) * 2.4)),
            (f"Quality change {q_before:.3f} -> {q_after:.3f}", stretched),
        ],
    )
    save_grid(
        "04_quality_measure_filter.png",
        [
            ("Original image", gray),
            ("Local EME/EEME filter", quality_filtered),
            ("Contrast stretched reference", stretched),
            ("Filtered minus original", gray_to_rgb(np.abs(quality_filtered - gray) * 2.0)),
        ],
    )
    save_grid(
        "05_histogram_equalization.png",
        [
            ("Original image f", gray),
            ("Histogram equalized image", equalized),
            ("Original histogram spread", gray_to_rgb(np.clip(gray, 0, 1))),
            ("Equalized contrast result", equalized),
        ],
    )
    draw_curve("06_equalization_transformation_curve.png", mapping, "Histogram equalization transformation: u vs. v")
    save_grid(
        "07_gamma_correction.png",
        [
            ("Input image f", gray),
            ("Gamma corrected image g, gamma=0.55", gamma),
            ("Mean histogram stretch bonus", spatial),
            ("Spatial enhancement difference", gray_to_rgb(np.abs(spatial - gray) * 1.8)),
        ],
    )
    save_grid(
        "08_color_enhancement.png",
        [
            ("Original color image", color),
            ("Luminance equalized color image", color_plus),
            ("Original luminance", rgb_to_gray(color)),
            ("Enhanced luminance", rgb_to_gray(color_plus)),
        ],
    )


def segmentation_outputs() -> None:
    two_mode = generated_segmentation_image(modes=2)
    three_mode = generated_segmentation_image(modes=3)
    color = generated_color_segmentation()
    color_gray = rgb_to_gray(color)

    ranges, range_labels = histogram_peak_ranges(three_mode)
    draw_histogram(
        "09_histogram_segmentation_plot.png",
        three_mode,
        "Histogram-based segmentation: peaks, ranges, and object classes",
        thresholds=[0.33, 0.58],
        ranges=ranges,
    )
    draw_object_boxes("10_histogram_identified_objects.png", three_mode, range_labels)
    save_grid(
        "11_histogram_segmentation_masks.png",
        [
            ("Input with three significant modes", three_mode),
            ("Background range", (range_labels == 0).astype(float)),
            ("Middle-intensity object range", np.isclose(range_labels, 0.55).astype(float)),
            ("Bright object range", (range_labels == 1).astype(float)),
        ],
    )

    t_two_a, hist_two_a = iterative_threshold(two_mode, 0.25)
    t_two_b, hist_two_b = iterative_threshold(two_mode, 0.72)
    t_three_a, hist_three_a = iterative_threshold(three_mode, 0.30)
    t_three_b, hist_three_b = iterative_threshold(three_mode, 0.75)
    draw_histogram("12_global_threshold_two_mode_hist.png", two_mode, "Two-mode image global thresholding", thresholds=[t_two_a, t_two_b])
    draw_histogram("13_global_threshold_three_mode_hist.png", three_mode, "Three-mode image global thresholding", thresholds=[t_three_a, t_three_b])
    save_grid(
        "14_global_thresholding.png",
        [
            ("Two modes: source", two_mode),
            (f"Initial .25 -> T {t_two_a:.3f}", (two_mode > t_two_a).astype(float)),
            (f"Initial .72 -> T {t_two_b:.3f}", (two_mode > t_two_b).astype(float)),
            ("Three modes: source", three_mode),
            (f"Initial .30 -> T {t_three_a:.3f}", (three_mode > t_three_a).astype(float)),
            (f"Initial .75 -> T {t_three_b:.3f}", (three_mode > t_three_b).astype(float)),
        ],
        columns=3,
    )

    thresholds_a, hist_a = multilevel_thresholds(three_mode, (0.25, 0.70))
    thresholds_b, hist_b = multilevel_thresholds(three_mode, (0.42, 0.84))
    adaptive_a = adaptive_threshold(three_mode, radius=10, offset=0.015)
    adaptive_b = adaptive_threshold(three_mode, radius=22, offset=0.030)
    draw_histogram(
        "15_multilevel_threshold_hist.png",
        three_mode,
        "Multilevel thresholding on three-mode image",
        thresholds=[thresholds_a[0], thresholds_a[1], thresholds_b[0], thresholds_b[1]],
    )
    save_grid(
        "16_multilevel_and_adaptive_thresholding.png",
        [
            ("Three-mode source", three_mode),
            (f"Multilevel init (.25,.70) -> {thresholds_a[0]:.2f}, {thresholds_a[1]:.2f}", classify_multilevel(three_mode, thresholds_a)),
            (f"Multilevel init (.42,.84) -> {thresholds_b[0]:.2f}, {thresholds_b[1]:.2f}", classify_multilevel(three_mode, thresholds_b)),
            ("Adaptive radius 10", adaptive_a),
            ("Adaptive radius 22", adaptive_b),
            ("Adaptive overlay", overlay_mask(three_mode, adaptive_b, (0.92, 0.30, 0.18))),
        ],
        columns=3,
    )

    n_gray = niblack(three_mode)
    b_gray = bernsen(three_mode)
    s_gray = sauvola(three_mode)
    n_color = niblack(color_gray)
    b_color = bernsen(color_gray)
    s_color = sauvola(color_gray)
    save_grid(
        "17_niblack_bernsen_sauvola_gray.png",
        [
            ("Gray source", three_mode),
            ("Niblack thresholding", n_gray),
            ("Bernsen thresholding", b_gray),
            ("Sauvola thresholding", s_gray),
        ],
    )
    save_grid(
        "18_niblack_bernsen_sauvola_color.png",
        [
            ("Color source", color),
            ("Color luminance", color_gray),
            ("Niblack on color luminance", overlay_mask(color, n_color, (0.95, 0.28, 0.18))),
            ("Bernsen on color luminance", overlay_mask(color, b_color, (0.20, 0.78, 0.45))),
            ("Sauvola on color luminance", overlay_mask(color, s_color, (0.98, 0.82, 0.18))),
            ("Sauvola binary mask", s_color),
        ],
        columns=3,
    )

    bonus = mean_histogram_stretch(three_mode)
    t_bonus, _hist_bonus = iterative_threshold(bonus, 0.5)
    save_grid(
        "19_bonus_mean_histogram_stretch_segmentation.png",
        [
            ("Before mean histogram stretch", three_mode),
            ("After mean histogram stretch", bonus),
            (f"Segmentation after stretch, T={t_bonus:.3f}", (bonus > t_bonus).astype(float)),
            ("Overlay after stretch", overlay_mask(bonus, (bonus > t_bonus).astype(float), (0.18, 0.72, 0.92))),
        ],
    )

    summary = [
        "HW6 numeric summary",
        f"Quality measure before contrast stretch: {block_quality(generated_enhancement_image())[1]:.4f}",
        f"Quality measure after contrast stretch: {block_quality(contrast_stretch(generated_enhancement_image()))[1]:.4f}",
        f"Two-mode global threshold histories: {np.round(hist_two_a, 3).tolist()} and {np.round(hist_two_b, 3).tolist()}",
        f"Three-mode global threshold histories: {np.round(hist_three_a, 3).tolist()} and {np.round(hist_three_b, 3).tolist()}",
        f"Multilevel threshold histories: {[(round(a, 3), round(b, 3)) for a, b in hist_a]}",
        f"Alternate multilevel threshold histories: {[(round(a, 3), round(b, 3)) for a, b in hist_b]}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    enhancement_outputs()
    segmentation_outputs()
    outputs = sorted(path.name for path in OUT_DIR.glob("*.png"))
    print(f"Generated {len(outputs)} HW6 images in {OUT_DIR}")
    for name in outputs:
        print(" -", name)


if __name__ == "__main__":
    main()
