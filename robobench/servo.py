"""Closed-loop image-Jacobian visual servo.

    err = target_px - tip_px            ∈ ℝ²
    Δq  = α · J⁺ · err                  ∈ ℝ³  (deg)
    q_new = clip(q + Δq, joint limits)
    POST joint commands, settle, re-snap, repeat.

Stops on convergence (|err| < tol_px), max iterations, joint clamp violation,
or detector failure. Detector failure leaves the arm in place.
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


def servo(node: str,
          jacobian_path: str,
          out_dir: str = "/tmp/robobench_servo",
          tol_px: float = 15.0,
          gain: float = 0.45,
          max_step: float = 2.5,
          max_travel: float = 15.0,
          max_iters: int = 8,
          settle: float = 1.0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cli = NodeClient(node)

    jac = json.load(open(jacobian_path))
    J = np.array(jac["J"])
    Jpinv = np.linalg.pinv(J, rcond=1e-3)
    print("[servo] J =\n", J)
    print("[servo] J+ =\n", Jpinv)

    s0 = cli.state()
    q0 = np.array([s0["positions"][j] for j in JOINTS])
    pos = q0.copy()
    print("[servo] start pose:", dict(zip(JOINTS, pos.round(2).tolist())))

    # baseline frame
    p0 = os.path.join(out_dir, "iter0.jpg")
    cli.snap_jpeg(p0)
    img = cv2.imread(p0)
    tip = detect_grip_tip(img)
    target = detect_target(img)
    print(f"[servo] iter 0  tip={tip}  target={target}")
    if tip is None or target is None:
        raise RuntimeError(f"missing detection at start: tip={tip} target={target}")
    err = (target[0] - tip[0], target[1] - tip[1])
    cv2.imwrite(p0, annotate(img, tip, target, "iter 0"))

    history = [{"iter": 0, "tip": list(tip), "target": list(target),
                "err": list(err), "pose": pos.tolist()}]

    converged = False
    abort_reason = None
    for it in range(1, max_iters + 1):
        e = np.array([target[0] - tip[0], target[1] - tip[1]], dtype=float)
        mag = float(np.linalg.norm(e))
        print(f"\n[iter {it}] err=({int(e[0]):+d},{int(e[1]):+d})  |{mag:.0f}px|")
        if mag < tol_px:
            converged = True
            print(f"  CONVERGED  (|err|={mag:.1f} < {tol_px})")
            break

        dq = gain * (Jpinv @ e)
        dq = np.clip(dq, -max_step, max_step)

        target_pose = pos + dq
        # cumulative-travel safety
        travel = target_pose - q0
        for i, j in enumerate(JOINTS):
            if abs(travel[i]) > max_travel:
                target_pose[i] = q0[i] + np.sign(travel[i]) * max_travel
                print(f"  CLAMP travel {j} -> ±{max_travel} from start")
            lo, hi = LIMITS[j]
            if not (lo <= target_pose[i] <= hi):
                target_pose[i] = float(np.clip(target_pose[i], lo, hi))
                print(f"  CLAMP limit {j} -> [{lo}, {hi}]")

        print("  dq:", dict(zip(JOINTS, (target_pose - pos).round(2).tolist())))
        for i, j in enumerate(JOINTS):
            if abs(target_pose[i] - pos[i]) > 0.01:
                cli.command(j, float(target_pose[i]))
        pos = target_pose
        time.sleep(settle)

        # snap, detect
        p_iter = os.path.join(out_dir, f"iter{it}.jpg")
        cli.snap_jpeg(p_iter)
        img = cv2.imread(p_iter)
        tip = detect_grip_tip(img)
        target = detect_target(img)
        if tip is None or target is None:
            print(f"  WARN missing detection tip={tip} target={target}; aborting")
            abort_reason = "detection_failed"
            cv2.imwrite(p_iter, annotate(img, tip, target, f"iter {it} FAIL"))
            history.append({"iter": it, "tip": tip, "target": target,
                            "err": None, "pose": pos.tolist()})
            break
        err = (target[0] - tip[0], target[1] - tip[1])
        cv2.imwrite(p_iter, annotate(img, tip, target, f"iter {it}"))
        print(f"  -> tip={tip} target={target} err={err} "
              f"|{(err[0]**2+err[1]**2)**0.5:.0f}px|")
        history.append({"iter": it, "tip": list(tip), "target": list(target),
                        "err": list(err), "pose": pos.tolist()})
    else:
        abort_reason = "max_iters"

    sN = cli.state()
    final = {j: sN["positions"][j] for j in JOINTS}
    print("\n[servo] final pose:",
          {j: round(final[j], 2) for j in JOINTS},
          "Δ:", {j: round(final[j] - q0[i], 2) for i, j in enumerate(JOINTS)})

    summary = {
        "node": node, "jacobian_path": jacobian_path,
        "converged": converged, "abort_reason": abort_reason,
        "iters": len(history) - 1,
        "tol_px": tol_px, "gain": gain, "max_step": max_step,
        "max_travel": max_travel, "max_iters": max_iters,
        "start_pose": dict(zip(JOINTS, q0.tolist())),
        "final_pose": final,
        "history": history,
    }
    json.dump(summary, open(os.path.join(out_dir, "history.json"), "w"),
              indent=2, default=list)
    return summary


def _main():
    ap = argparse.ArgumentParser(description="Run closed-loop visual servo.")
    ap.add_argument("--node", default=DEFAULT_NODE)
    ap.add_argument("--jacobian", required=True,
                    help="path to jacobian.json from `probe`")
    ap.add_argument("--out", default="/tmp/robobench_servo")
    ap.add_argument("--tol", type=float, default=15.0,
                    help="convergence threshold in pixels")
    ap.add_argument("--gain", type=float, default=0.45,
                    help="step gain α (0..1)")
    ap.add_argument("--max-step", type=float, default=2.5,
                    help="per-iteration joint cap, deg")
    ap.add_argument("--max-travel", type=float, default=15.0,
                    help="cumulative joint travel cap from start, deg")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--settle", type=float, default=1.0)
    args = ap.parse_args()

    s = servo(args.node, args.jacobian, args.out,
              tol_px=args.tol, gain=args.gain,
              max_step=args.max_step, max_travel=args.max_travel,
              max_iters=args.max_iters, settle=args.settle)
    print("\nresult:", "CONVERGED" if s["converged"] else f"abort: {s['abort_reason']}")


if __name__ == "__main__":
    _main()
