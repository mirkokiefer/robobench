"""End-to-end: probe -> servo. Convenience wrapper."""
from __future__ import annotations

import argparse
import os

from .config import DEFAULT_NODE
from .probe import probe
from .servo import servo


def _main():
    ap = argparse.ArgumentParser(description="Probe + servo end-to-end.")
    ap.add_argument("--node", default=DEFAULT_NODE)
    ap.add_argument("--out", default="/tmp/robobench_run",
                    help="output dir; probe/ and servo/ subdirs are created")
    ap.add_argument("--delta", type=float, default=1.5)
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--gain", type=float, default=0.45)
    ap.add_argument("--max-step", type=float, default=2.5)
    ap.add_argument("--max-travel", type=float, default=15.0)
    ap.add_argument("--max-iters", type=int, default=8)
    args = ap.parse_args()

    probe_dir = os.path.join(args.out, "probe")
    servo_dir = os.path.join(args.out, "servo")
    probe(args.node, args.delta, settle=1.2, out_dir=probe_dir)

    jacobian = os.path.join(probe_dir, "jacobian.json")
    servo(args.node, jacobian, out_dir=servo_dir,
          tol_px=args.tol, gain=args.gain,
          max_step=args.max_step, max_travel=args.max_travel,
          max_iters=args.max_iters)


if __name__ == "__main__":
    _main()
