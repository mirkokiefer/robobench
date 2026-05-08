#!/usr/bin/env python3
"""Build "what moved between two frames" visualizations.

Given a pair of images (start, end) — typically the first and last frames
of a servo run — produce three views:

  1. heat       — pixels that changed are highlighted red over a darkened base
  2. anaglyph   — start in cyan, end in red (greyscale anaglyph-style)
  3. strip      — labelled side-by-side: start | end (and optional middle)

Usage:
  python scripts/make_diff_images.py BEFORE.jpg AFTER.jpg --out docs/

Examples used to produce the docs/ images in this repo:
  python scripts/make_diff_images.py \
      /tmp/cams/servo/iter0.jpg /tmp/cams/servo/iter6.jpg --out docs/
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def label(im: np.ndarray, text: str, scale: float = 0.9) -> np.ndarray:
    """Burn-in a top-left label with a black pillarbox for legibility."""
    out = im.copy()
    pad = 10
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(out, (12, 12), (12 + tw + 2 * pad, 12 + th + 2 * pad),
                  (0, 0, 0), -1)
    cv2.putText(out, text, (12 + pad, 12 + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return out


def diff_heat(a: np.ndarray, b: np.ndarray, threshold: int = 25,
              amplify: float = 3.0) -> np.ndarray:
    """Highlight pixels that changed between a and b in red."""
    A = np.asarray(a, dtype=np.int16)
    B = np.asarray(b, dtype=np.int16)
    d = np.abs(A - B).max(axis=2)
    d = np.clip(d.astype(np.float32) * amplify, 0, 255).astype(np.uint8)
    mask = d > threshold
    base = (np.asarray(a) * 0.45).astype(np.uint8)
    heat = np.zeros_like(base)
    heat[mask] = [80, 80, 255]   # BGR — red in OpenCV
    return np.where(mask[..., None], heat, base)


def diff_anaglyph(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Greyscale anaglyph: a in cyan, b in red."""
    A = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    B = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # OpenCV is BGR. Cyan = high G+B, no R. Red = high R, no G+B.
    return np.stack([A, A, B], axis=-1)


def strip(images: list[tuple[np.ndarray, str]], gap: int = 12) -> np.ndarray:
    """Horizontal labelled strip."""
    H = max(im.shape[0] for im, _ in images)
    W = sum(im.shape[1] for im, _ in images) + gap * (len(images) - 1)
    out = np.full((H, W, 3), 18, dtype=np.uint8)
    x = 0
    for im, text in images:
        h, w = im.shape[:2]
        out[:h, x:x + w] = label(im, text)
        x += w + gap
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--middle", default=None,
                    help="optional middle frame for the strip")
    ap.add_argument("--out", default="diffs",
                    help="output directory (created if missing)")
    ap.add_argument("--prefix", default="servo",
                    help="filename prefix for outputs")
    ap.add_argument("--label-before", default="BEFORE")
    ap.add_argument("--label-after",  default="AFTER")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    a = cv2.imread(args.before)
    b = cv2.imread(args.after)
    if a is None or b is None:
        raise SystemExit(f"could not read {args.before} or {args.after}")
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    # 1. heat
    heat = label(diff_heat(a, b),
                 "MOTION  (red = pixels that changed)")
    cv2.imwrite(os.path.join(args.out, f"{args.prefix}_diff_heat.jpg"),
                heat, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # 2. anaglyph
    an = label(diff_anaglyph(a, b), "BEFORE = cyan,  AFTER = red")
    cv2.imwrite(os.path.join(args.out, f"{args.prefix}_diff_anaglyph.jpg"),
                an, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # 3. strip
    panels = [(a, args.label_before)]
    if args.middle:
        m = cv2.imread(args.middle)
        if m is not None:
            if m.shape != a.shape:
                m = cv2.resize(m, (a.shape[1], a.shape[0]))
            panels.append((m, "MIDDLE"))
    panels.append((b, args.label_after))
    cv2.imwrite(os.path.join(args.out, f"{args.prefix}_before_after.jpg"),
                strip(panels), [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    print(f"wrote 3 images to {args.out}/ with prefix {args.prefix}_")


if __name__ == "__main__":
    main()
