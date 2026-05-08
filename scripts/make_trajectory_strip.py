#!/usr/bin/env python3
"""Build a labelled grid of servo iteration frames.

Reads iter0.jpg, iter1.jpg, ... from a servo output directory and tiles them
into a single image with iteration number + a caption per cell.

Usage:
  python scripts/make_trajectory_strip.py /tmp/cams/servo \
      --out docs/servo_trajectory.jpg --cols 4 --tile 480x360

Captions are read from history.json (per-iteration |err|, pose) if present.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("servo_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", default="480x360", help="WxH per tile")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()

    tw, th = (int(x) for x in args.tile.lower().split("x"))

    # Discover iter*.jpg files
    files = []
    pat = re.compile(r"iter(\d+)(?:_ann)?\.jpg")
    for f in os.listdir(args.servo_dir):
        m = pat.fullmatch(f)
        if m:
            files.append((int(m.group(1)), f))
    files.sort()
    # Deduplicate to one per iter (prefer _ann)
    seen = {}
    for it, f in files:
        if it not in seen or "_ann" in f:
            seen[it] = f
    iters = sorted(seen.items())
    if not iters:
        raise SystemExit(f"no iter*.jpg files in {args.servo_dir}")

    # Load history.json for captions if available
    captions = {}
    hist_path = os.path.join(args.servo_dir, "history.json")
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        rows = h.get("history", h) if isinstance(h, dict) else h
        for r in rows:
            err = r.get("err")
            it = r.get("iter")
            if err is not None:
                mag = (err[0] ** 2 + err[1] ** 2) ** 0.5
                captions[it] = f"|err|={mag:.0f} px"
            else:
                captions[it] = "(no detection)"

    cols = args.cols
    rows = (len(iters) + cols - 1) // cols
    W = cols * tw + (cols - 1) * args.gap
    H = rows * th + (rows - 1) * args.gap
    canvas = np.full((H, W, 3), 18, dtype=np.uint8)

    for i, (it, f) in enumerate(iters):
        im = cv2.imread(os.path.join(args.servo_dir, f))
        if im is None:
            continue
        im = cv2.resize(im, (tw, th))
        # Top label
        cv2.rectangle(im, (0, 0), (tw, 36), (0, 0, 0), -1)
        cv2.putText(im, f"iter {it}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        # Bottom caption
        cap = captions.get(it, "")
        if cap:
            cv2.rectangle(im, (0, th - 30), (tw, th), (0, 0, 0), -1)
            cv2.putText(im, cap, (10, th - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        r, c = divmod(i, cols)
        y, x = r * (th + args.gap), c * (tw + args.gap)
        canvas[y:y + th, x:x + tw] = im

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
    print(f"wrote {args.out}  ({canvas.shape[1]}x{canvas.shape[0]}, {len(iters)} cells)")


if __name__ == "__main__":
    main()
