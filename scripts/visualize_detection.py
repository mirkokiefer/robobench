#!/usr/bin/env python3
"""Show every step of the detection pipeline as a 6-panel grid.

For a single input frame, produces:

  +----------+----------+----------+
  | input    | green    | red      |
  | image    | mask     | mask     |
  +----------+----------+----------+
  | wires    | dark-arm | final    |
  | centroid | PCA axis | annotated|
  +----------+----------+----------+

Pi PCB detector  -> green HSV blob, largest contour
Gripper tip      -> red HSV blob (left half), then PCA on dark arm pixels
                    around the wires to extrapolate forward along the axis

Usage:
  python scripts/visualize_detection.py path/to/frame.jpg \
      --out docs/detection/

Drops six numbered .jpg files plus a combined `detection_grid.jpg`.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def label(im, text, scale=0.85):
    out = im.copy()
    pad = 10
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(out, (12, 12), (12 + tw + 2 * pad, 12 + th + 2 * pad),
                  (0, 0, 0), -1)
    cv2.putText(out, text, (12 + pad, 12 + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default="detection")
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---- 1. input
    p1 = label(img, "1. input  (cam1)")
    cv2.imwrite(f"{args.out}/01_input.jpg", p1)

    # ---- 2. green mask  (Pi PCB detector)
    green = cv2.inRange(hsv, (35, 60, 30), (85, 255, 200))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    green_vis = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pi = None
    if cs:
        c = max(cs, key=cv2.contourArea)
        cv2.drawContours(green_vis, [c], -1, (0, 255, 0), 2)
        M = cv2.moments(c)
        if M["m00"] > 0:
            pi = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.drawMarker(green_vis, pi, (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 3)
    p2 = label(green_vis, "2. green HSV mask  ->  Pi PCB centroid")
    cv2.imwrite(f"{args.out}/02_green_mask.jpg", p2)

    # ---- 3. red mask (gripper wires), right half suppressed
    r1 = cv2.inRange(hsv, (0,   120, 80), (10,  255, 255))
    r2 = cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    red = cv2.bitwise_or(r1, r2)
    red[:, W * 55 // 100:] = 0
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    red_vis = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
    rcs, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wires = None
    if rcs:
        c = max(rcs, key=cv2.contourArea)
        cv2.drawContours(red_vis, [c], -1, (0, 0, 255), 2)
        M = cv2.moments(c)
        if M["m00"] > 0:
            wires = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.drawMarker(red_vis, wires, (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 3)
    p3 = label(red_vis, "3. red HSV mask (left half)  ->  wire centroid")
    cv2.imwrite(f"{args.out}/03_red_mask.jpg", p3)

    # ---- 4. wires centroid on the original image
    p4_img = img.copy()
    if wires:
        cv2.drawMarker(p4_img, wires, (0, 0, 255), cv2.MARKER_CROSS, 36, 3)
        cv2.putText(p4_img, "wires", (wires[0] + 14, wires[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    p4 = label(p4_img, "4. wire centroid on input")
    cv2.imwrite(f"{args.out}/04_wires.jpg", p4)

    # ---- 5. dark-arm PCA axis around the wires
    p5_img = img.copy()
    tip = None
    if wires:
        cx, cy = wires
        y0, y1 = max(0, cy - 160), min(H, cy + 160)
        x0, x1 = max(0, cx - 280), min(W, cx + 280)
        band = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, 70)

        # overlay the dark mask (in cyan) onto the input copy
        overlay = p5_img.copy()
        overlay[y0:y1, x0:x1][dark > 0] = (255, 255, 0)
        p5_img = cv2.addWeighted(p5_img, 0.55, overlay, 0.45, 0)

        pts = np.column_stack(np.where(dark > 0))
        if len(pts) >= 200:
            pts_xy = np.flip(pts, axis=1).astype(np.float32)
            mean, evecs = cv2.PCACompute(pts_xy, mean=None)
            v = evecs[0]
            if v[0] < 0:
                v = -v
            cv2.line(p5_img, (cx, cy),
                     (int(cx + v[0] * 220), int(cy + v[1] * 220)),
                     (0, 255, 255), 3)
            tip = (int(cx + v[0] * 200), int(cy + v[1] * 200))
    p5 = label(p5_img, "5. dark-arm PCA  ->  principal axis")
    cv2.imwrite(f"{args.out}/05_pca_axis.jpg", p5)

    # ---- 6. final: TIP and TARGET with error vector
    p6_img = img.copy()
    if pi:
        cv2.drawMarker(p6_img, pi, (0, 255, 0), cv2.MARKER_CROSS, 32, 3)
        cv2.putText(p6_img, "TARGET", (pi[0] + 14, pi[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if tip:
        cv2.drawMarker(p6_img, tip, (0, 165, 255),
                       cv2.MARKER_TILTED_CROSS, 32, 3)
        cv2.putText(p6_img, "TIP", (tip[0] + 14, tip[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    if pi and tip:
        cv2.arrowedLine(p6_img, tip, pi, (255, 255, 0), 2, tipLength=0.05)
        e = (pi[0] - tip[0], pi[1] - tip[1])
        mag = (e[0] ** 2 + e[1] ** 2) ** 0.5
        cv2.putText(p6_img,
                    f"err=({e[0]:+d},{e[1]:+d})  |{mag:.0f} px|",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 0), 2)
    p6 = label(p6_img, "6. result  TIP + TARGET + error vector")
    cv2.imwrite(f"{args.out}/06_final.jpg", p6)

    # ---- combined 3x2 grid
    panels = [p1, p2, p3, p4, p5, p6]
    panels = [cv2.resize(p, (W // 2, H // 2)) for p in panels]
    cols, rows, gap = 3, 2, 6
    pw, ph = panels[0].shape[1], panels[0].shape[0]
    grid = np.full((rows * ph + (rows - 1) * gap,
                    cols * pw + (cols - 1) * gap, 3), 18, dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = r * (ph + gap), c * (pw + gap)
        grid[y:y + ph, x:x + pw] = panel
    cv2.imwrite(f"{args.out}/detection_grid.jpg", grid,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
    print(f"wrote 7 images to {args.out}/  (target={pi}, tip={tip})")


if __name__ == "__main__":
    main()
