"""Default endpoints, joint set, and safety limits."""

DEFAULT_NODE = "https://pi5b-node.daslab.dev"
DEFAULT_CAM  = "/dev/video0"

# Browser-y UA — Cloudflare 403s python-urllib's default UA.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0"
)

# Joints used by visual servo (a 3-DoF subset is plenty to position the tip
# in the image plane). Ordering matters — it defines columns of the Jacobian.
JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
]

# Safety clamps (deg). Tuned conservatively for a tabletop SO-101.
LIMITS = {
    "shoulder_pan.pos":  ( 60.0, 130.0),
    "shoulder_lift.pos": (-60.0, -10.0),  # don't slam into table
    "elbow_flex.pos":    ( 10.0,  80.0),
    "wrist_flex.pos":    (-90.0,  90.0),
    "wrist_roll.pos":    (-90.0,  90.0),
    "gripper.pos":       (  0.0, 100.0),
}
