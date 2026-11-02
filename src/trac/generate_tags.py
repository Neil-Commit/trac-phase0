"""Generate printable AprilTag (36h11) sheets at an exact physical size.

    python -m trac.generate_tags --ids 0 1 2 --size-mm 100 --out tags

Each ID produces a PDF and a PNG (both carry the physical scale) containing the tag, a
label, and a 50 mm scale bar. PRINT AT 100% / "ACTUAL SIZE" (never "fit to page"), then
measure the black square with a ruler and put that value in config/trac_config.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import APRILTAG_36H11_ID_COUNT

PAGES_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
MODULES = 8  # 6x6 data bits + 1 module black border each side


def render_tag_sheet(tag_id: int, size_mm: float, dpi: int = 300, page: str = "a4"
                     ) -> Tuple[np.ndarray, float]:
    """Return (grayscale page image, actual printed tag size in mm)."""
    if not 0 <= tag_id < APRILTAG_36H11_ID_COUNT:
        raise ValueError(f"Tag ID must be in 0..{APRILTAG_36H11_ID_COUNT - 1}")
    if size_mm <= 0 or dpi <= 0:
        raise ValueError("size_mm and dpi must be positive")
    page_w_mm, page_h_mm = PAGES_MM[page]
    page_w, page_h = (round(v / 25.4 * dpi) for v in (page_w_mm, page_h_mm))

    # Use a whole number of pixels per module so tag edges are perfectly crisp.
    module_px = max(1, round(size_mm / 25.4 * dpi / MODULES))
    side = module_px * MODULES
    actual_mm = side / dpi * 25.4
    margin = 3 * module_px
    if side + 2 * margin > min(page_w, page_h) - round(10 / 25.4 * dpi):
        raise ValueError(f"A {size_mm:g} mm tag does not fit on a {page.upper()} page with margins")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, side, 1)

    sheet = np.full((page_h, page_w), 255, dtype=np.uint8)
    x0, y0 = (page_w - side) // 2, round(page_h * 0.30)
    sheet[y0:y0 + side, x0:x0 + side] = marker
    # Light cut guide, well outside the white quiet zone
    cv2.rectangle(sheet, (x0 - margin, y0 - margin), (x0 + side + margin, y0 + side + margin), 170, 1)

    text_scale = dpi / 300 * 1.1
    thick = max(1, round(dpi / 150))
    lines = [
        f"AprilTag 36h11   ID {tag_id}",
        f"Black square side = {actual_mm:.1f} mm.  Print at 100% / Actual size.  Measure it with a ruler!",
    ]
    y = y0 - margin - round(0.25 * dpi)
    for i, line in enumerate(reversed(lines)):
        cv2.putText(sheet, line, (x0 - margin, y - i * round(0.28 * dpi)), cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale * (0.8 if i == 1 else 1.0), 0, thick, cv2.LINE_AA)

    # 50 mm scale bar to verify print scale
    bar = round(50 / 25.4 * dpi)
    by = y0 + side + margin + round(0.45 * dpi)
    bx = (page_w - bar) // 2
    cv2.line(sheet, (bx, by), (bx + bar, by), 0, max(2, thick))
    for px in (bx, bx + bar):
        cv2.line(sheet, (px, by - 12), (px, by + 12), 0, max(2, thick))
    cv2.putText(sheet, "this bar must measure 50 mm", (bx, by + round(0.35 * dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.7, 0, thick, cv2.LINE_AA)
    return sheet, actual_mm


def write_tag_files(tag_id: int, size_mm: float, out_dir: Path, dpi: int = 300, page: str = "a4"
                    ) -> Tuple[Path, Path, float]:
    sheet, actual_mm = render_tag_sheet(tag_id, size_mm, dpi, page)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tag36h11_id{tag_id:03d}_{actual_mm:.0f}mm"
    img = Image.fromarray(sheet)
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    img.save(png, dpi=(dpi, dpi))
    img.save(pdf, resolution=float(dpi))
    return png, pdf, actual_mm


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate printable AprilTag 36h11 sheets")
    p.add_argument("--ids", type=int, nargs="+", default=[0], help="tag IDs (0-586)")
    p.add_argument("--size-mm", type=float, default=100.0, help="outer black square side in mm")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--page", choices=sorted(PAGES_MM), default="a4")
    p.add_argument("--out", type=Path, default=Path("tags"))
    args = p.parse_args(argv)
    try:
        for tag_id in args.ids:
            png, pdf, actual = write_tag_files(tag_id, args.size_mm, args.out, args.dpi, args.page)
            print(f"ID {tag_id}: {pdf}  (and {png.name})  tag side = {actual:.1f} mm")
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    print("Print at 100% scale, then measure the black square and update tag.size_mm in the config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
