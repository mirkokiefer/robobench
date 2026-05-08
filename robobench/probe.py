"""Probe-learn the 2x3 image Jacobian.

For each joint q_i in JOINTS:
    read state s0
    command q_i := s0[i] + delta;  settle; snap;  detect tip -> p_plus
    command q_i := s0[i] - delta;  settle; snap;  detect tip -> p_minus
    command q_i := s0[i]          (return to start)
    J[:, i] = (p_plus - p_minus) / (2*delta)

Saves the Jacobian + provenance to a JSON file you can pass to `servo`.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from .client import NodeClient
from .config import DEFAULT_NODE, JOINTS, LIMITS
from .detect import annotate, detect_grip_tip, detect_target


def probe(node: str = DEFAULT_NODE,
          delta: float = 1.5,
          settle: float = 1.2,
          out_dir: str = "/tmp/robobench_probe") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cli = NodeClient(node)

    s0 = cli.state()
    pos0 = {j: s0["positions"][j] for j in JOINTS}
    print("[probe] start pose:")
    for j in JOINTS:
        print(f"  {j} = {pos0[j]:.2f}")

    # baseline frame
    base = os.path.join(out_dir, "baseline.jpg")
    cli.snap_jpeg(base, duration=2.0)
    img0 = cv2.imread(base)
    tip0 = detect_grip_tip(img0)
    target0 = detect_target(img0)
    print(f"[probe] baseline  tip={tip0}  target={target0}")
    cv2.imwrite(base, annotate(img0, tip0, target0, "baseline"))
    if tip0 is None:
        raise RuntimeError("could not detect gripper tip at baseline")

    J = np.zeros((2, len(JOINTS)))
    probes = []
    for i, joint in enumerate(JOINTS):
        q = pos0[joint]
        lo, hi = LIMITS[joint]
        if not (lo <= q + delta <= hi and lo <= q - delta <= hi):
            raise RuntimeError(
                f"probe of {joint} ±{delta} from {q:.2f} would exit "
                f"limits {lo, hi}"
            )

        print(f"\n[probe {i+1}/{len(JOINTS)}] {joint}: +{delta}")
        cli.command(joint, q + delta); time.sleep(settle)
        p_plus = os.path.join(out_dir, f"{joint.replace('.', '_')}_plus.jpg")
        cli.snap_jpeg(p_plus, duration=1.5)
        tip_p = detect_grip_tip(cv2.imread(p_plus))

        print(f"[probe {i+1}/{len(JOINTS)}] {joint}: -{delta}")
        cli.command(joint, q - delta); time.sleep(settle)
        p_minus = os.path.join(out_dir, f"{joint.replace('.', '_')}_minus.jpg")
        cli.snap_jpeg(p_minus, duration=1.5)
        tip_m = detect_grip_tip(cv2.imread(p_minus))

        # return
        cli.command(joint, q); time.sleep(settle)

        if tip_p is None or tip_m is None:
            print(f"  WARN missing detection plus={tip_p} minus={tip_m}")
            J[:, i] = [0.0, 0.0]
        else:
            du = (tip_p[0] - tip_m[0]) / (2 * delta)
            dv = (tip_p[1] - tip_m[1]) / (2 * delta)
            J[:, i] = [du, dv]
            print(f"  tip+={tip_p}  tip-={tip_m}  J[:,{i}] = "
                  f"({du:+.2f}, {dv:+.2f}) px/deg")
        probes.append({"joint": joint, "q0": q,
                       "tip_plus": tip_p, "tip_minus": tip_m})

    print("\n[probe] J (px/deg):")
    print(np.array2string(J, precision=2))

    out = {
        "node": node,
        "joints": JOINTS,
        "delta_deg": delta,
        "start_pose": pos0,
        "tip0": tip0,
        "target0": target0,
        "J": J.tolist(),
        "probes": probes,
    }
    json.dump(out, open(os.path.join(out_dir, "jacobian.json"), "w"), indent=2)
    return out


def _main():
    ap = argparse.ArgumentParser(description="Probe-learn the image Jacobian.")
    ap.add_argument("--node", default=DEFAULT_NODE)
    ap.add_argument("--delta", type=float, default=1.5,
                    help="probe magnitude per joint, deg")
    ap.add_argument("--settle", type=float, default=1.2,
                    help="seconds to wait between command and snap")
    ap.add_argument("--out", default="/tmp/robobench_probe",
                    help="output directory")
    args = ap.parse_args()

    out = probe(args.node, args.delta, args.settle, args.out)
    print(f"\nsaved {os.path.join(args.out, 'jacobian.json')}")


if __name__ == "__main__":
    _main()
