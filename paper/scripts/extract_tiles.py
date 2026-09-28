"""Crop the 64x64 sky-image tiles out of the saved figure grids.

Sources (both render each 64x64 frame with nearest-neighbour upscaling, so
resampling the tile back to 64x64 recovers the original pixels):
  * ../showcase_15min_sky_images.png   -- CSR-MMDiff (this work), notebook Cell 17
  * baseline1 Fig. 4 (embedded bitmap)  -- joint DDIM v1 [B1], same three windows
Output: figures/tiles/{src}_{window}_{row}_{col}.png
"""
import os
import sys

import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "figures", "tiles")
os.makedirs(OUT, exist_ok=True)


def find_tiles(rgb):
    """Bounding boxes of the square image panels (non-white blobs), sorted row-major."""
    nonwhite = rgb.astype(int).sum(-1) < 3 * 235
    nonwhite = ndimage.binary_closing(nonwhite, iterations=3)
    lab, n = ndimage.label(nonwhite)
    boxes = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h > 80 and w > 80 and 0.85 < h / w < 1.15:
            boxes.append((sl[0].start, sl[1].start, h, w))
    hs = np.median([b[2] for b in boxes])
    boxes = [b for b in boxes if abs(b[2] - hs) < 0.1 * hs]
    rows = {}
    for b in boxes:
        key = min(rows, key=lambda k: abs(k - b[0]), default=None)
        if key is None or abs(key - b[0]) > hs / 2:
            rows[b[0]] = [b]
        else:
            rows[key].append(b)
    return [sorted(rows[k], key=lambda b: b[1]) for k in sorted(rows)]


def crop(rgb, box, inset=0.02):
    y, x, h, w = box
    d = int(round(inset * h))
    tile = Image.fromarray(rgb[y + d : y + h - d, x + d : x + w - d])
    return tile.resize((64, 64), Image.NEAREST)


def export(rgb, tag):
    grid = find_tiles(rgb)
    print(tag, "rows:", [len(r) for r in grid])
    names = ["clear", "typical", "cloudy"]
    # rows alternate observed (9 tiles incl. t=0) / generated (8 tiles)
    for j, name in enumerate(names):
        obs, gen = grid[2 * j], grid[2 * j + 1]
        assert len(obs) == 9 and len(gen) == 8, (len(obs), len(gen))
        for c, b in enumerate(obs):
            crop(rgb, b).save(os.path.join(OUT, f"{tag}_{name}_obs_{c}.png"))
        for c, b in enumerate(gen):
            crop(rgb, b).save(os.path.join(OUT, f"{tag}_{name}_gen_{c + 1}.png"))


ours = np.asarray(Image.open(os.path.join(ROOT, "showcase_15min_sky_images.png")).convert("RGB"))
export(ours, "ours")

doc = pymupdf.open(os.path.join(ROOT, "paper_baseline", "baseline1.pdf"))
xref = doc[4].get_images(full=True)[0][0]
pix = pymupdf.Pixmap(doc, xref)
b1 = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
export(np.ascontiguousarray(b1), "b1")

# B2 (matched-checkpoint study) Fig. 1(e): four generated frames of one member (t+3..t+15)
doc2 = pymupdf.open(os.path.join(ROOT, "paper_baseline", "baseline2.pdf"))
imgs = doc2[1].get_images(full=True)
for k, idx in enumerate([4, 5, 6, 7]):
    p = pymupdf.Pixmap(doc2, imgs[idx][0])
    a = np.frombuffer(p.samples, np.uint8).reshape(p.height, p.width, p.n)[..., :3]
    Image.fromarray(np.ascontiguousarray(a)).resize((64, 64), Image.LANCZOS).save(
        os.path.join(OUT, f"b2_gen_{k}.png"))
print("done")
