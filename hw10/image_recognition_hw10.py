from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "generated_images"
OUT_DIR = ROOT / "output"
SIZE = 192
CLASSES = ("coin", "leaf", "car")


@dataclass
class Sample:
    label: str
    index: int
    image: np.ndarray
    mask: np.ndarray
    filename: str


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722])


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image), mode="RGB").save(path)


def background(size: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    u = x / (size - 1)
    v = y / (size - 1)
    base = np.dstack(
        [
            0.70 + 0.09 * u,
            0.72 + 0.06 * v,
            0.68 + 0.05 * (1 - u),
        ]
    )
    paper = 0.015 * np.sin(2 * math.pi * (u * 7 + v * 4))
    noise = rng.normal(0, 0.012, (size, size, 3))
    return np.clip(base + paper[..., None] + noise, 0, 1)


def add_mask_texture(image: np.ndarray, mask: np.ndarray, color: np.ndarray, texture: np.ndarray) -> np.ndarray:
    result = image.copy()
    textured = np.clip(color + texture[..., None], 0, 1)
    result[mask] = textured[mask]
    return result


def ellipse_mask(size: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


def polygon_mask(size: int, points: list[tuple[float, float]]) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(points, fill=255)
    return np.asarray(image) > 0


def rectangle_mask(size: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    return mask


def make_coin(index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = background(SIZE, rng)
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    cx = SIZE * (0.50 + rng.normal(0, 0.025))
    cy = SIZE * (0.51 + rng.normal(0, 0.02))
    radius = SIZE * (0.30 + rng.normal(0, 0.01))
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    ring = np.sin(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * 0.45 + index)
    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    texture = 0.035 * ring + 0.055 * (1 - radial)
    image = add_mask_texture(image, mask, np.array([0.82, 0.63, 0.22]), texture)

    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(box, outline=(92, 64, 18, 220), width=4)
    draw.ellipse((cx - radius * 0.72, cy - radius * 0.72, cx + radius * 0.72, cy + radius * 0.72), outline=(248, 219, 98, 155), width=3)
    draw.text((cx - 10, cy - 10), "$", fill=(86, 59, 16, 180), font=ImageFont.load_default())
    return np.asarray(canvas).astype(float) / 255, mask


def make_leaf(index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = background(SIZE, rng)
    cx = SIZE * (0.50 + rng.normal(0, 0.025))
    cy = SIZE * (0.50 + rng.normal(0, 0.025))
    points = [
        (cx, cy - 67),
        (cx + 48, cy - 25),
        (cx + 39, cy + 24),
        (cx + 4, cy + 68),
        (cx - 39, cy + 26),
        (cx - 50, cy - 22),
    ]
    mask = polygon_mask(SIZE, points)
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    veins = np.sin((x - cx) * 0.22 + (y - cy) * 0.14 + index)
    gradient = 0.055 * (1 - np.abs(x - cx) / 75)
    texture = 0.028 * veins + gradient
    image = add_mask_texture(image, mask, np.array([0.18, 0.58, 0.25]), texture)

    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.line((cx, cy - 61, cx, cy + 61), fill=(222, 244, 156, 170), width=2)
    for offset in (-40, -22, -5, 13, 31):
        draw.line((cx, cy + offset, cx + 34, cy + offset - 20), fill=(216, 239, 151, 110), width=1)
        draw.line((cx, cy + offset, cx - 34, cy + offset - 20), fill=(216, 239, 151, 110), width=1)
    return np.asarray(canvas).astype(float) / 255, mask


def make_car(index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = background(SIZE, rng)
    body = rectangle_mask(SIZE, 40 + rng.integers(-4, 4), 87 + rng.integers(-3, 3), 153 + rng.integers(-4, 4), 132 + rng.integers(-2, 3))
    cabin = polygon_mask(SIZE, [(65, 87), (86, 55), (123, 55), (143, 87)])
    wheels = ellipse_mask(SIZE, 70, 132, 14, 14) | ellipse_mask(SIZE, 128, 132, 14, 14)
    mask = body | cabin | wheels
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    stripes = np.sin(x * 0.35 + index) * 0.025
    shine = 0.05 * np.exp(-((y - 83) ** 2) / 300)
    image = add_mask_texture(image, body | cabin, np.array([0.66, 0.13, 0.15]), stripes + shine)
    image[wheels] = np.array([0.06, 0.06, 0.07])

    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.polygon([(68, 84), (88, 59), (121, 59), (139, 84)], fill=(145, 207, 226, 210), outline=(70, 42, 44, 180))
    draw.rectangle((45, 91, 150, 125), outline=(74, 32, 31, 210), width=3)
    draw.ellipse((56, 118, 84, 146), fill=(26, 27, 30, 255))
    draw.ellipse((114, 118, 142, 146), fill=(26, 27, 30, 255))
    draw.ellipse((64, 126, 76, 138), fill=(158, 160, 166, 255))
    draw.ellipse((122, 126, 134, 138), fill=(158, 160, 166, 255))
    return np.asarray(canvas).astype(float) / 255, mask


def local_binary_pattern(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    padded = np.zeros(gray.shape, dtype=np.uint8)
    padded[1:-1, 1:-1] = code
    return padded


def glcm_features(levels_image: np.ndarray, mask: np.ndarray, levels: int = 16) -> dict[str, float]:
    glcm = np.zeros((levels, levels), dtype=float)
    offsets = ((0, 1), (1, 0), (1, 1), (-1, 1))
    h, w = levels_image.shape
    for dy, dx in offsets:
        y0 = max(0, -dy)
        y1 = min(h, h - dy)
        x0 = max(0, -dx)
        x1 = min(w, w - dx)
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


def lbp_histogram(lbp: np.ndarray, mask: np.ndarray, bins: int = 16) -> np.ndarray:
    values = lbp[mask]
    hist, _ = np.histogram(values, bins=bins, range=(0, 256))
    hist = hist.astype(float)
    return hist / max(hist.sum(), 1.0)


def boundary_points(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    return np.argwhere(core & ~interior)


def chain_code_histogram(mask: np.ndarray) -> np.ndarray:
    points = boundary_points(mask)
    if len(points) < 2:
        return np.zeros(8)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 0] - center[0], points[:, 1] - center[1])
    ordered = points[np.argsort(angles)]
    deltas = np.diff(np.vstack([ordered, ordered[0]]), axis=0)
    dirs = {
        (0, 1): 0,
        (-1, 1): 1,
        (-1, 0): 2,
        (-1, -1): 3,
        (0, -1): 4,
        (1, -1): 5,
        (1, 0): 6,
        (1, 1): 7,
    }
    hist = np.zeros(8, dtype=float)
    for dy, dx in deltas:
        sy = int(np.sign(dy))
        sx = int(np.sign(dx))
        hist[dirs.get((sy, sx), 0)] += 1
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


def extract_features(sample: Sample) -> tuple[dict[str, float], np.ndarray]:
    image = sample.image
    mask = sample.mask
    gray = rgb_to_gray(image)
    lbp_gray = local_binary_pattern(gray)
    lbp_channels = [local_binary_pattern(image[..., channel]) for channel in range(3)]

    features: dict[str, float] = {
        "area_ratio": float(mask.mean()),
        "box_counting_dimension": box_counting_dimension(mask),
    }

    chain_hist = chain_code_histogram(mask)
    for i, value in enumerate(chain_hist):
        features[f"chain_{i}"] = float(value)

    hist = lbp_histogram(lbp_gray, mask)
    for i, value in enumerate(hist):
        features[f"lbp_bin_{i:02d}"] = float(value)

    names = ("red", "green", "blue")
    for channel_name, lbp in zip(names, lbp_channels):
        levels = np.clip(lbp // 16, 0, 15)
        haralick = glcm_features(levels, mask)
        for key, value in haralick.items():
            features[f"haralick_{channel_name}_{key}"] = value

    return features, lbp_gray


def generate_samples() -> list[Sample]:
    IMAGE_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    generators = {"coin": make_coin, "leaf": make_leaf, "car": make_car}
    samples = []
    for label in CLASSES:
        for index in range(6):
            rng = np.random.default_rng(1000 + CLASSES.index(label) * 100 + index)
            image, mask = generators[label](index, rng)
            filename = f"{label}_{index + 1:02d}.png"
            save_rgb(IMAGE_DIR / filename, image)
            samples.append(Sample(label=label, index=index, image=image, mask=mask, filename=filename))
    return samples


def nearest_centroid_classifier(rows: list[dict[str, object]], feature_names: list[str]) -> list[str]:
    matrix = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    labels = np.array([str(row["label"]) for row in rows])
    predictions = []
    for i, row in enumerate(matrix):
        train = np.delete(matrix, i, axis=0)
        train_labels = np.delete(labels, i)
        mean = train.mean(axis=0)
        std = train.std(axis=0) + 1e-9
        normalized_train = (train - mean) / std
        normalized_row = (row - mean) / std
        centroids = {
            label: normalized_train[train_labels == label].mean(axis=0)
            for label in CLASSES
            if np.any(train_labels == label)
        }
        prediction = min(centroids, key=lambda label: float(np.linalg.norm(normalized_row - centroids[label])))
        predictions.append(prediction)
    return predictions


def contact_sheet(path: Path, panels: list[tuple[str, np.ndarray]], columns: int = 6) -> None:
    tile_w, tile_h = 178, 214
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), (246, 246, 241))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (label, image) in enumerate(panels):
        row, col = divmod(idx, columns)
        x = col * tile_w
        y = row * tile_h
        panel = Image.fromarray(to_uint8(image), mode="RGB").resize((160, 160), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x + 9, y + 30))
        draw.rectangle((x, y, x + tile_w, y + 26), fill=(32, 38, 44))
        draw.text((x + 8, y + 7), label, fill=(246, 244, 236), font=font)
    canvas.save(path)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=float)
    image[..., 0] = mask * 0.95
    image[..., 1] = mask * 0.88
    image[..., 2] = mask * 0.22
    return image


def lbp_to_rgb(lbp: np.ndarray) -> np.ndarray:
    normalized = lbp.astype(float) / 255
    return np.dstack([normalized, np.sqrt(normalized), 1 - normalized * 0.45])


def write_report(rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(rows)
    confusion = {(actual, predicted): 0 for actual in CLASSES for predicted in CLASSES}
    for actual, predicted in zip(labels, predictions):
        confusion[(actual, predicted)] += 1

    lines = [
        "HW10 object recognition report",
        "================================",
        "",
        "Feature set:",
        "- Haralick contrast, energy, homogeneity, and correlation extracted from RGB-channel LBP images.",
        "- 16-bin grayscale LBP histogram.",
        "- Normalized object area.",
        "- Box-counting dimension of the segmented mask.",
        "- 8-direction chain-code histogram from the object boundary.",
        "",
        "Classifier: leave-one-out nearest centroid with z-score normalization.",
        f"Accuracy: {accuracy:.3f} ({sum(a == b for a, b in zip(labels, predictions))}/{len(rows)})",
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        "actual/predicted," + ",".join(CLASSES),
    ]
    for actual in CLASSES:
        lines.append(actual + "," + ",".join(str(confusion[(actual, predicted)]) for predicted in CLASSES))
    lines.append("")
    lines.append("Per-image predictions:")
    for row in rows:
        lines.append(f"{row['filename']}: actual={row['label']}, predicted={row['prediction']}")
    (OUT_DIR / "classification_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def feature_summary_image(rows: list[dict[str, object]]) -> None:
    metrics = [
        ("Area", "area_ratio"),
        ("Box dim.", "box_counting_dimension"),
        ("LBP high", "lbp_bin_15"),
        ("R contrast", "haralick_red_contrast"),
        ("G homog.", "haralick_green_homogeneity"),
        ("B energy", "haralick_blue_energy"),
    ]
    class_values = {
        label: [np.mean([float(row[key]) for row in rows if row["label"] == label]) for _, key in metrics]
        for label in CLASSES
    }
    maxima = [
        max(class_values[label][index] for label in CLASSES) or 1.0
        for index in range(len(metrics))
    ]

    width, height = 980, 420
    canvas = Image.new("RGB", (width, height), (247, 247, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, 38), fill=(31, 37, 43))
    draw.text((18, 13), "Class-level feature averages", fill=(247, 244, 235), font=font)

    colors = {"coin": (210, 158, 47), "leaf": (62, 151, 73), "car": (185, 50, 52)}
    left = 90
    chart_top = 80
    group_w = 142
    bar_w = 26
    baseline = 330
    scale_h = 220
    for metric_index, (metric_label, _key) in enumerate(metrics):
        group_x = left + metric_index * group_w
        draw.text((group_x, baseline + 22), metric_label, fill=(31, 37, 43), font=font)
        for class_index, label in enumerate(CLASSES):
            value = class_values[label][metric_index]
            normalized = value / maxima[metric_index]
            x0 = group_x + class_index * (bar_w + 8)
            y0 = baseline - int(normalized * scale_h)
            draw.rectangle((x0, y0, x0 + bar_w, baseline), fill=colors[label])
            draw.text((x0 - 2, y0 - 14), f"{value:.2f}", fill=(31, 37, 43), font=font)

    legend_x = 780
    for index, label in enumerate(CLASSES):
        y = chart_top + index * 28
        draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=colors[label])
        draw.text((legend_x + 28, y + 4), label, fill=(31, 37, 43), font=font)
    canvas.save(OUT_DIR / "05_feature_summary.png")


def main() -> None:
    samples = generate_samples()
    rows: list[dict[str, object]] = []
    lbp_images: list[tuple[str, np.ndarray]] = []
    mask_images: list[tuple[str, np.ndarray]] = []

    for sample in samples:
        features, lbp = extract_features(sample)
        row: dict[str, object] = {"filename": sample.filename, "label": sample.label}
        row.update(features)
        rows.append(row)
        lbp_images.append((sample.filename, lbp_to_rgb(lbp)))
        mask_images.append((sample.filename, mask_to_rgb(sample.mask)))

    feature_names = [key for key in rows[0].keys() if key not in {"filename", "label", "prediction"}]
    predictions = nearest_centroid_classifier(rows, feature_names)
    for row, prediction in zip(rows, predictions):
        row["prediction"] = prediction

    fieldnames = ["filename", "label", "prediction"] + feature_names
    with (OUT_DIR / "features.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_report(rows)
    feature_summary_image(rows)

    contact_sheet(OUT_DIR / "01_dataset_contact_sheet.png", [(s.filename, s.image) for s in samples])
    contact_sheet(OUT_DIR / "02_lbp_contact_sheet.png", lbp_images)
    contact_sheet(OUT_DIR / "03_masks_contact_sheet.png", mask_images)

    montage_panels = []
    for sample, row in zip(samples, rows):
        image = Image.fromarray(to_uint8(sample.image), mode="RGB")
        draw = ImageDraw.Draw(image)
        predicted = str(row["prediction"])
        color = (24, 116, 54) if predicted == sample.label else (180, 35, 28)
        draw.rectangle((0, 0, SIZE, 20), fill=color)
        draw.text((6, 6), f"{sample.label} -> {predicted}", fill=(255, 255, 255), font=ImageFont.load_default())
        montage_panels.append((sample.filename, np.asarray(image).astype(float) / 255))
    contact_sheet(OUT_DIR / "04_classification_montage.png", montage_panels)

    correct = sum(str(row["label"]) == str(row["prediction"]) for row in rows)
    print(f"Generated {len(samples)} images in {IMAGE_DIR}")
    print(f"Extracted {len(feature_names)} features per image")
    print(f"Leave-one-out nearest-centroid accuracy: {correct}/{len(rows)}")
    print(f"Wrote report to {OUT_DIR / 'classification_report.txt'}")


if __name__ == "__main__":
    main()
