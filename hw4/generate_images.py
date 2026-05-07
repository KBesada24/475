from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


OUT_DIR = Path(__file__).resolve().parent
WIDTH = 960
HEIGHT = 640


def save_rgb(name: str, pixels: np.ndarray) -> None:
    pixels = np.clip(pixels, 0, 1)
    image = Image.fromarray((pixels * 255).astype(np.uint8), mode="RGB")
    image.save(OUT_DIR / name)


def procedural_scene(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    u = x / (width - 1)
    v = y / (height - 1)

    sky = np.stack(
        [
            0.08 + 0.44 * (1 - v),
            0.18 + 0.48 * (1 - v),
            0.38 + 0.55 * (1 - v),
        ],
        axis=-1,
    )

    sun_center = np.array([0.76, 0.22])
    sun_dist = np.sqrt((u - sun_center[0]) ** 2 + (v - sun_center[1]) ** 2)
    sun = np.exp(-(sun_dist * 10.5) ** 2)[..., None] * np.array([1.0, 0.68, 0.24])
    image = np.clip(sky + sun, 0, 1)

    mountain_1 = 0.54 + 0.08 * np.sin(2 * math.pi * (u * 1.4 + 0.08))
    mountain_1 += 0.035 * np.sin(2 * math.pi * (u * 5.1 + 0.31))
    mountain_2 = 0.67 + 0.07 * np.sin(2 * math.pi * (u * 2.0 + 0.62))
    mountain_2 += 0.02 * np.sin(2 * math.pi * (u * 7.0 + 0.16))

    far = v > mountain_1
    near = v > mountain_2
    image[far] = np.array([0.18, 0.26, 0.34])
    image[near] = np.array([0.08, 0.21, 0.16])

    water = v > 0.76
    wave = 0.035 * np.sin(2 * math.pi * (u * 18 + v * 4))
    reflection = np.exp(-((u - sun_center[0]) * 8) ** 2) * np.exp(-((v - 0.84) * 10) ** 2)
    water_color = np.stack(
        [
            0.04 + 0.10 * (1 - v) + 0.52 * reflection,
            0.15 + 0.14 * (1 - v) + 0.32 * reflection,
            0.24 + 0.26 * (1 - v) + 0.08 * reflection,
        ],
        axis=-1,
    )
    image[water] = np.clip(water_color[water] + wave[water, None], 0, 1)

    rng = np.random.default_rng(475)
    noise = rng.normal(0, 0.012, image.shape)
    vignette = 1 - 0.28 * ((u - 0.5) ** 2 + (v - 0.52) ** 2)
    return np.clip((image + noise) * vignette[..., None], 0, 1)


def convolve_gray(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(gray)
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            out += kernel[row, col] * padded[row : row + gray.shape[0], col : col + gray.shape[1]]
    return out


def sobel_edges(image: np.ndarray) -> np.ndarray:
    gray = image @ np.array([0.2126, 0.7152, 0.0722])
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    gx = convolve_gray(gray, kx)
    gy = convolve_gray(gray, ky)
    magnitude = np.sqrt(gx * gx + gy * gy)
    magnitude = np.clip(magnitude / np.percentile(magnitude, 98), 0, 1)
    colored = np.stack([magnitude * 0.95, magnitude * 0.82, magnitude * 0.34], axis=-1)
    return np.clip(0.16 * image + colored, 0, 1)


def floyd_steinberg_dither(image: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0.05, 0.08, 0.14],
            [0.10, 0.27, 0.22],
            [0.18, 0.36, 0.55],
            [0.80, 0.36, 0.20],
            [0.95, 0.70, 0.32],
            [0.92, 0.90, 0.72],
        ]
    )
    work = image.copy()
    out = np.zeros_like(work)
    h, w, _ = work.shape
    for y in range(h):
        for x in range(w):
            old = work[y, x].copy()
            idx = np.argmin(np.sum((palette - old) ** 2, axis=1))
            new = palette[idx]
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                work[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    work[y + 1, x - 1] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    work[y + 1, x + 1] += err * 1 / 16
    return np.clip(out, 0, 1)


def halftone(image: np.ndarray, cell: int = 10) -> np.ndarray:
    base = Image.new("RGB", (image.shape[1], image.shape[0]), (247, 242, 220))
    draw = ImageDraw.Draw(base, "RGBA")
    gray = image @ np.array([0.2126, 0.7152, 0.0722])
    colors = [(18, 66, 82, 185), (205, 74, 45, 120), (235, 176, 52, 110)]
    offsets = [(0, 0), (cell // 3, cell // 5), (-cell // 4, cell // 3)]
    for layer, color in enumerate(colors):
        shifted = np.roll(gray, shift=layer * 9, axis=1)
        for y in range(cell // 2, image.shape[0], cell):
            for x in range(cell // 2, image.shape[1], cell):
                patch = shifted[max(0, y - cell // 2) : y + cell // 2, max(0, x - cell // 2) : x + cell // 2]
                darkness = 1 - float(np.mean(patch))
                radius = (darkness ** 1.35) * cell * 0.58
                ox, oy = offsets[layer]
                draw.ellipse((x + ox - radius, y + oy - radius, x + ox + radius, y + oy + radius), fill=color)
    return np.asarray(base).astype(float) / 255


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def raytrace(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    spheres = [
        (np.array([-0.85, -0.12, 3.0]), 0.72, np.array([0.95, 0.28, 0.18]), 0.22),
        (np.array([0.36, -0.26, 2.35]), 0.52, np.array([0.18, 0.58, 0.92]), 0.34),
        (np.array([1.05, 0.02, 3.45]), 0.62, np.array([0.98, 0.76, 0.28]), 0.16),
    ]
    light_pos = np.array([-3.0, -4.0, -2.0])
    camera = np.array([0.0, 0.0, -2.6])
    y, x = np.mgrid[0:height, 0:width]
    aspect = width / height
    px = (2 * (x + 0.5) / width - 1) * aspect
    py = 1 - 2 * (y + 0.5) / height
    dirs = np.dstack([px, py, np.ones_like(px) * 1.55])
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    image = np.zeros((height, width, 3), dtype=float)

    for row in range(height):
        for col in range(width):
            origin = camera.copy()
            direction = dirs[row, col]
            color = np.zeros(3)
            throughput = np.ones(3)
            for _bounce in range(2):
                hit_t = float("inf")
                hit = None
                for center, radius, albedo, reflectivity in spheres:
                    oc = origin - center
                    b = 2 * np.dot(oc, direction)
                    c = np.dot(oc, oc) - radius * radius
                    disc = b * b - 4 * c
                    if disc >= 0:
                        t = (-b - math.sqrt(disc)) / 2
                        if 0.001 < t < hit_t:
                            hit_t = t
                            hit = (center, radius, albedo, reflectivity)
                plane_t = float("inf")
                if direction[1] > 1e-4:
                    plane_t = (0.72 - origin[1]) / direction[1]
                if plane_t < hit_t:
                    point = origin + plane_t * direction
                    checker = (math.floor(point[0] * 2) + math.floor(point[2] * 2)) % 2
                    floor_color = np.array([0.78, 0.74, 0.64]) if checker else np.array([0.34, 0.38, 0.40])
                    normal = np.array([0.0, -1.0, 0.0])
                    light = normalize(light_pos - point)
                    diffuse = max(0, np.dot(normal, light))
                    color += throughput * floor_color * (0.18 + 0.72 * diffuse)
                    break
                if hit is None:
                    t = 0.5 * (direction[1] + 1.0)
                    color += throughput * ((1 - t) * np.array([0.88, 0.92, 1.0]) + t * np.array([0.16, 0.24, 0.38]))
                    break
                center, radius, albedo, reflectivity = hit
                point = origin + hit_t * direction
                normal = normalize(point - center)
                light = normalize(light_pos - point)
                diffuse = max(0, np.dot(normal, light))
                view = -direction
                halfway = normalize(light + view)
                specular = max(0, np.dot(normal, halfway)) ** 80
                color += throughput * (albedo * (0.13 + 0.78 * diffuse) + specular * np.array([1.0, 0.94, 0.82]))
                throughput *= reflectivity
                origin = point + normal * 0.002
                direction = direction - 2 * np.dot(direction, normal) * normal
            image[row, col] = color
    return np.clip(image ** (1 / 2.2), 0, 1)


def main() -> None:
    base = procedural_scene()
    save_rgb("01_procedural_scene.png", base)
    save_rgb("02_sobel_edge_overlay.png", sobel_edges(base))
    save_rgb("03_floyd_steinberg_dither.png", floyd_steinberg_dither(base))
    save_rgb("04_halftone_render.png", halftone(base))
    save_rgb("05_raytraced_spheres.png", raytrace())
    print("Generated 5 images in", OUT_DIR)


if __name__ == "__main__":
    main()
