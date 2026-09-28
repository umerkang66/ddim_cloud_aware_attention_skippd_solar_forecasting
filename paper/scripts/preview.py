"""Renders main.pdf to a page montage (scripts/_pages.png) for a quick visual check."""
import os

import pymupdf
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
doc = pymupdf.open(os.path.join(here, "..", "main.pdf"))
print("pages", len(doc))
ims = []
for i, p in enumerate(doc):
    f = os.path.join(here, f"_page{i + 1}.png")
    p.get_pixmap(dpi=100).save(f)
    ims.append(Image.open(f))
w, h = ims[0].size
W = Image.new("RGB", (w * 3, h * ((len(ims) + 2) // 3)), "white")
for i, im in enumerate(ims):
    W.paste(im, ((i % 3) * w, (i // 3) * h))
W.save(os.path.join(here, "_pages.png"))
