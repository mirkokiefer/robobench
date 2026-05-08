"""Detectors for visual servoing.

Tuned for the demo scene (Raspberry Pi 5 PCB on a white table, SO-101 with
red-and-white wires running through the gripper). Replace `detect_target`
with your own object detector for new scenes — the rest of the pipeline
is detector-agnostic.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

PixelXY = Tuple[int, int]


# ---------- target: green Pi PCB

def detect_target(img: np.ndarray) -> Optional[PixelXY]:
    """Return centroid of the largest green blob (the Pi PCB) or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (35, 60, 30), (85, 255, 200))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    M = cv2.moments(c)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# ---------- gripper tip: red wires + PCA forward extrapolation

def _detect_red_wires(img: np.ndarray) -> Optional[PixelXY]:
    """Centroid of the red wire bundle on the gripper.
    Suppresses the right half of the frame so a red Pi *box* doesn't
    confuse it with the wires."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, (0,   120, 80), (10,  255, 255))
    r2 = cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    m = cv2.bitwise_or(r1, r2)
    m[:, W * 55 // 100:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None
    M = cv2.moments(c)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def detect_grip_tip(img: np.ndarray, extrap_px: int = 200) -> Optional[PixelXY]:
    """Estimate gripper-tip pixel by extrapolating along the arm's principal
    axis from the red wire centroid.
    """
    H, W = img.shape[:2]
    wires = _detect_red_wires(img)
    if wires is None:
        return None
    cx, cy = wires
    # PCA over the dark-arm pixels in a band around the wires.
    band = img[max(0, cy - 160):min(H, cy + 160),
               max(0, cx - 280):min(W, cx + 280)]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 70)
    pts = np.column_stack(np.where(dark > 0))
    if len(pts) < 200:
        return (cx + extrap_px, cy)
    pts_xy = np.flip(pts, axis=1).astype(np.float32)
    _, evecs = cv2.PCACompute(pts_xy, mean=None)
    v = evecs[0]
    if v[0] < 0:
        v = -v
    return (int(cx + v[0] * extrap_px), int(cy + v[1] * extrap_px))


# ---------- annotate

def annotate(img: np.ndarray,
             tip: Optional[PixelXY],
             target: Optional[PixelXY],
             label: str = "") -> np.ndarray:
    """Draw tip / target / error vector on a copy of img."""
    out = img.copy()
    H, W = out.shape[:2]
    if target is not None:
        cv2.drawMarker(out, target, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(out, "TARGET", (target[0] + 12, target[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if tip is not None:
        cv2.drawMarker(out, tip, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
        cv2.putText(out, "TIP", (tip[0] + 12, tip[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    if tip is not None and target is not None:
        cv2.arrowedLine(out, tip, target, (255, 255, 0), 2, tipLength=0.05)
        e = (target[0] - tip[0], target[1] - tip[1])
        mag = (e[0] ** 2 + e[1] ** 2) ** 0.5
        cv2.putText(out,
                    f"{label}  err=({e[0]:+d},{e[1]:+d})  |{mag:.0f}px|",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
    elif label:
        cv2.putText(out, label, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (255, 255, 255), 2)
    return out


# ---------- CLI

def _main():
    import argparse, json, os
    ap = argparse.ArgumentParser(description="Run detectors on a single frame.")
    ap.add_argument("image")
    ap.add_argument("--out", default=None,
                    help="path for annotated output (default: <image>.annotated.jpg)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    target = detect_target(img)
    tip = detect_grip_tip(img)
    out_path = args.out or os.path.splitext(args.image)[0] + ".annotated.jpg"
    cv2.imwrite(out_path, annotate(img, tip, target, "detect"))
    print(json.dumps({"target": target, "tip": tip, "out": out_path}))


if __name__ == "__main__":
    _main()
