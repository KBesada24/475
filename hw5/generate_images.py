from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
WIDTH = 960
HEIGHT = 640


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def save_rgb(name: str, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image), mode="RGB").save(OUT_DIR / name)


def save_grid(name: str, panels: list[tuple[str, np.ndarray]], columns: int = 2) -> None:
    tile_w, tile_h = 480, 320
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), (248, 246, 236))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, image) in enumerate(panels):
        row, col = divmod(index, columns)
        panel = Image.fromarray(to_uint8(image), mode="RGB").resize((tile_w, tile_h - 26), Image.Resampling.LANCZOS)
        x = col * tile_w
        y = row * tile_h
        canvas.paste(panel, (x, y + 26))
        draw.rectangle((x, y, x + tile_w, y + 26), fill=(30, 37, 44))
        draw.text((x + 12, y + 7), label, fill=(244, 238, 222), font=font)

    canvas.save(OUT_DIR / name)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722])


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


def convolve(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(channel, dtype=float)
    for y in range(kh):
        for x in range(kw):
            output += kernel[y, x] * padded[y : y + channel.shape[0], x : x + channel.shape[1]]
    return output


def convolve_rgb(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.dstack([convolve(image[..., channel], kernel) for channel in range(3)])


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def generated_still_life(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    rng = np.random.default_rng(4755)

    wall = np.dstack(
        [
            0.70 + 0.12 * (1 - v),
            0.66 + 0.10 * (1 - v),
            0.58 + 0.08 * (1 - v),
        ]
    )
    table = np.dstack(
        [
            0.36 + 0.10 * u,
            0.25 + 0.07 * u,
            0.16 + 0.05 * u,
        ]
    )
    image = np.where((v > 0.56)[..., None], table, wall)

    plank = 0.025 * np.sin(2 * math.pi * (v * 34 + 0.2 * np.sin(u * 9)))
    grain = 0.018 * np.sin(2 * math.pi * (u * 42 + v * 4))
    image[v > 0.56] += (plank[v > 0.56] + grain[v > 0.56])[..., None]

    sun = np.exp(-(((u - 0.18) / 0.36) ** 2 + ((v - 0.12) / 0.24) ** 2))
    image += sun[..., None] * np.array([0.24, 0.17, 0.08])

    canvas = Image.fromarray(to_uint8(image), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    objects = [
        ((232, 348, 410, 526), (218, 92, 54, 255), "orange bowl"),
        ((492, 274, 680, 526), (44, 126, 178, 255), "blue vase"),
        ((674, 372, 820, 520), (236, 194, 58, 255), "yellow cup"),
        ((360, 410, 548, 560), (62, 152, 94, 255), "green apple"),
    ]

    for box, color, _name in objects:
        sx0, sy0, sx1, sy1 = box
        draw.ellipse((sx0 + 30, sy1 - 12, sx1 + 75, sy1 + 30), fill=(30, 24, 18, 80))
        draw.ellipse(box, fill=color)
        draw.ellipse((sx0 + 24, sy0 + 20, sx1 - 72, sy0 + 82), fill=(255, 255, 255, 48))
        draw.arc((sx0 + 16, sy0 + 16, sx1 - 16, sy1 - 16), 200, 340, fill=(35, 31, 26, 130), width=4)

    draw.rectangle((502, 306, 668, 512), fill=(42, 121, 173, 255))
    draw.ellipse((492, 250, 680, 352), fill=(56, 150, 198, 255))
    draw.ellipse((512, 270, 650, 324), fill=(28, 76, 116, 255))
    draw.rectangle((552, 185, 620, 284), fill=(50, 139, 188, 255))
    draw.ellipse((552, 168, 620, 206), fill=(74, 169, 216, 255))

    image = np.asarray(canvas).astype(float) / 255
    speckle = rng.normal(0, 0.014, image.shape)
    vignette = 1 - 0.33 * ((u - 0.52) ** 2 + (v - 0.48) ** 2)
    return np.clip((image + speckle) * vignette[..., None], 0, 1)


def convolution_filter_grid(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    blur = convolve_rgb(source, gaussian_kernel(9, 1.8))
    sharpen = np.clip(source + 1.55 * (source - blur), 0, 1)
    gray = rgb_to_gray(source)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    mag = np.sqrt(convolve(gray, sobel_x) ** 2 + convolve(gray, sobel_y) ** 2)
    mag = np.clip(mag / np.percentile(mag, 98), 0, 1)
    emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float)
    emboss = np.clip(convolve_rgb(source, emboss_kernel) * 0.45 + 0.5, 0, 1)
    return [
        ("Original generated image", source),
        ("Gaussian blur, 9x9 kernel", blur),
        ("Unsharp mask sharpening", sharpen),
        ("Horizontal Sobel response", gray_to_rgb(np.clip(np.abs(convolve(gray, sobel_x)) / 2.2, 0, 1))),
        ("Sobel gradient magnitude", gray_to_rgb(mag)),
        ("Emboss convolution", emboss),
    ]


def fft_shift_visual(values: np.ndarray) -> np.ndarray:
    values = np.log1p(np.abs(values))
    values = values / values.max()
    return gray_to_rgb(values)


def frequency_filtering(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = rgb_to_gray(source)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    h, w = gray.shape
    y, x = np.mgrid[0:h, 0:w]
    radius = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    low_mask = radius < 46
    high_mask = radius > 32

    low = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * low_mask)))
    low = np.clip(low, 0, 1)
    high = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * high_mask)))
    high_vis = np.clip((high - high.min()) / (high.max() - high.min()), 0, 1)
    hybrid = np.clip(0.72 * low + 0.58 * high, 0, 1)

    return [
        ("Original grayscale", gray_to_rgb(gray)),
        ("Log FFT magnitude", fft_shift_visual(spectrum)),
        ("Low-pass circular mask", gray_to_rgb(low_mask.astype(float))),
        ("Circular low-pass reconstruction", gray_to_rgb(low)),
        ("High-pass detail", gray_to_rgb(high_vis)),
        ("Hybrid low + high result", gray_to_rgb(hybrid)),
    ]


def edge_pipeline(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = rgb_to_gray(source)
    smooth = convolve(gray, gaussian_kernel(7, 1.35))
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    gx = convolve(smooth, sobel_x)
    gy = convolve(smooth, sobel_y)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_norm = np.clip(mag / np.percentile(mag, 98.5), 0, 1)
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180

    nms = np.zeros_like(mag_norm)
    h, w = mag_norm.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            direction = angle[y, x]
            if direction < 22.5 or direction >= 157.5:
                neighbors = (mag_norm[y, x - 1], mag_norm[y, x + 1])
            elif direction < 67.5:
                neighbors = (mag_norm[y - 1, x + 1], mag_norm[y + 1, x - 1])
            elif direction < 112.5:
                neighbors = (mag_norm[y - 1, x], mag_norm[y + 1, x])
            else:
                neighbors = (mag_norm[y - 1, x - 1], mag_norm[y + 1, x + 1])
            if mag_norm[y, x] >= max(neighbors):
                nms[y, x] = mag_norm[y, x]

    edges = nms > 0.22
    edge_rgb = np.dstack([edges * 0.95, edges * 0.82, edges * 0.30]).astype(float)
    overlay = np.clip(source * 0.35 + edge_rgb, 0, 1)
    return [
        ("Grayscale input", gray_to_rgb(gray)),
        ("Smoothed image", gray_to_rgb(smooth)),
        ("Gradient magnitude", gray_to_rgb(mag_norm)),
        ("Non-maximum suppression", gray_to_rgb(np.clip(nms / max(nms.max(), 1e-6), 0, 1))),
        ("Binary edge mask", gray_to_rgb(edges.astype(float))),
        ("Thresholded edge overlay", overlay),
    ]


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


def connected_components(mask: np.ndarray, min_size: int = 450) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    components: list[tuple[int, int, int, int, int]] = []
    next_label = 1
    h, w = mask.shape
    for start_y in range(h):
        for start_x in range(w):
            if not mask[start_y, start_x] or labels[start_y, start_x] != 0:
                continue
            q: deque[tuple[int, int]] = deque([(start_y, start_x)])
            labels[start_y, start_x] = next_label
            pixels = []
            while q:
                y, x = q.popleft()
                pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = next_label
                        q.append((ny, nx))
            if len(pixels) < min_size:
                for y, x in pixels:
                    labels[y, x] = 0
                continue
            ys, xs = zip(*pixels)
            components.append((min(xs), min(ys), max(xs), max(ys), len(pixels)))
            next_label += 1
    return labels, components


def segmentation_morphology(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    red, green, blue = source[..., 0], source[..., 1], source[..., 2]
    warm_mask = (red > 0.48) & (red > green * 1.10) & (red > blue * 1.25)
    raw = warm_mask
    opened = dilate(erode(raw, 2), 2)
    closed = erode(dilate(opened, 4), 4)
    labels, components = connected_components(closed)

    label_vis = np.zeros_like(source)
    palette = np.array(
        [
            [0.92, 0.25, 0.18],
            [0.18, 0.66, 0.90],
            [0.20, 0.76, 0.42],
            [0.95, 0.78, 0.18],
        ]
    )
    for label in range(1, labels.max() + 1):
        label_vis[labels == label] = palette[(label - 1) % len(palette)]

    overlay_img = Image.fromarray(to_uint8(source * 0.58 + label_vis * 0.42), mode="RGB")
    draw = ImageDraw.Draw(overlay_img)
    for x0, y0, x1, y1, size in components:
        draw.rectangle((x0, y0, x1, y1), outline=(255, 238, 90), width=4)
        draw.text((x0 + 6, max(0, y0 - 16)), f"{size}px", fill=(255, 238, 90), font=ImageFont.load_default())

    return [
        ("Raw warm-color mask", gray_to_rgb(raw.astype(float))),
        ("After opening + closing", gray_to_rgb(closed.astype(float))),
        ("Connected components", label_vis),
        ("Labeled overlay", np.asarray(overlay_img).astype(float) / 255),
    ]


def sunset_scene(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    sky = np.dstack([0.12 + 0.64 * (1 - v), 0.18 + 0.34 * (1 - v), 0.38 + 0.24 * (1 - v)])
    glow = np.exp(-(((u - 0.72) / 0.18) ** 2 + ((v - 0.36) / 0.16) ** 2))
    sky += glow[..., None] * np.array([0.55, 0.28, 0.05])
    hills = v > 0.70 + 0.06 * np.sin(2 * math.pi * (u * 2.3 + 0.15))
    sky[hills] = np.array([0.07, 0.16, 0.15])
    water = v > 0.78
    sky[water] = np.array([0.05, 0.14, 0.24]) + glow[water, None] * np.array([0.45, 0.20, 0.04])
    return np.clip(sky, 0, 1)


def moon_scene(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)
    space = np.dstack([0.01 + 0.05 * (1 - v), 0.015 + 0.04 * (1 - v), 0.05 + 0.12 * (1 - v)])
    rng = np.random.default_rng(975)
    stars = rng.random((height, width)) > 0.9975
    space[stars] = 1
    dist = np.sqrt((u - 0.48) ** 2 + (v - 0.43) ** 2)
    moon = dist < 0.24
    texture = 0.78 + 0.10 * np.sin(u * 100) + 0.06 * np.cos(v * 90)
    craters = np.zeros_like(dist)
    for cx, cy, r in [(0.39, 0.37, 0.035), (0.55, 0.48, 0.055), (0.47, 0.55, 0.025), (0.58, 0.35, 0.022)]:
        craters += np.exp(-(((u - cx) ** 2 + (v - cy) ** 2) / (2 * r * r)))
    moon_color = np.clip(texture - 0.18 * craters, 0, 1)
    space[moon] = moon_color[moon, None] * np.array([0.95, 0.92, 0.84])
    return np.clip(space, 0, 1)


def downsample(image: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(to_uint8(image), mode="RGB").resize((image.shape[1] // 2, image.shape[0] // 2), Image.Resampling.BICUBIC)).astype(float) / 255


def upsample(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(to_uint8(image), mode="RGB").resize((shape[1], shape[0]), Image.Resampling.BICUBIC)).astype(float) / 255


def laplacian_blend() -> list[tuple[str, np.ndarray]]:
    left = sunset_scene()
    right = moon_scene()
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    mask = 1 / (1 + np.exp((x - WIDTH * 0.55 - 42 * np.sin(y / 70)) / 18))
    mask = np.repeat(mask[..., None], 3, axis=2)

    levels = 5
    gp_left, gp_right, gp_mask = [left], [right], [mask]
    for _ in range(levels - 1):
        gp_left.append(downsample(gp_left[-1]))
        gp_right.append(downsample(gp_right[-1]))
        gp_mask.append(downsample(gp_mask[-1]))

    lp_left, lp_right = [], []
    for level in range(levels - 1):
        lp_left.append(gp_left[level] - upsample(gp_left[level + 1], gp_left[level].shape))
        lp_right.append(gp_right[level] - upsample(gp_right[level + 1], gp_right[level].shape))
    lp_left.append(gp_left[-1])
    lp_right.append(gp_right[-1])

    blended = lp_left[-1] * gp_mask[-1] + lp_right[-1] * (1 - gp_mask[-1])
    for level in range(levels - 2, -1, -1):
        blended = upsample(blended, lp_left[level].shape)
        blended += lp_left[level] * gp_mask[level] + lp_right[level] * (1 - gp_mask[level])

    hard_cut = left * mask + right * (1 - mask)
    difference = gray_to_rgb(np.clip(np.mean(np.abs(np.clip(blended, 0, 1) - hard_cut), axis=2) * 8, 0, 1))
    return [
        ("Generated sunset source", left),
        ("Generated moon source", right),
        ("Feather mask", mask),
        ("Direct alpha blend", hard_cut),
        ("Pyramid vs alpha difference", difference),
        ("Laplacian pyramid blend", np.clip(blended, 0, 1)),
    ]


def main() -> None:
    source = generated_still_life()
    save_rgb("01_generated_still_life.png", source)
    save_grid("02_convolution_filter_grid.png", convolution_filter_grid(source), columns=2)
    save_grid("03_frequency_filtering.png", frequency_filtering(source), columns=2)
    save_grid("04_edge_pipeline.png", edge_pipeline(source), columns=2)
    save_grid("05_segmentation_morphology.png", segmentation_morphology(source), columns=2)
    save_grid("06_laplacian_pyramid_blend.png", laplacian_blend(), columns=2)
    print("Generated 6 HW5 images in", OUT_DIR)


if __name__ == "__main__":
    main()
