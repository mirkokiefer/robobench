"""HTTP client for a daslab Pi node hosting an SO-101.

Wraps three endpoints:
  GET  /api/robot/state                  -> joint positions
  POST /api/robot/command                -> set a single joint target
  GET  /api/stream?device=...            -> MJPEG (snap_jpeg pulls one frame)
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Optional

from .config import DEFAULT_CAM, DEFAULT_NODE, USER_AGENT


class NodeClient:
    def __init__(self, node: str = DEFAULT_NODE, timeout: float = 5.0):
        self.node = node.rstrip("/")
        self.timeout = timeout

    # ---- low level
    def _req(self, path: str, data: Optional[bytes] = None,
             method: str = "GET", ctype: Optional[str] = None) -> urllib.request.Request:
        headers = {"user-agent": USER_AGENT}
        if ctype:
            headers["content-type"] = ctype
        return urllib.request.Request(self.node + path, data=data,
                                      headers=headers, method=method)

    # ---- robot
    def state(self) -> dict:
        with urllib.request.urlopen(self._req("/api/robot/state"),
                                    timeout=self.timeout) as r:
            return json.loads(r.read())

    def command(self, joint: str, value: float) -> int:
        body = json.dumps({"joint": joint, "value": float(value)}).encode()
        with urllib.request.urlopen(
            self._req("/api/robot/command", body, "POST", "application/json"),
            timeout=self.timeout
        ) as r:
            return r.status

    # ---- video
    def snap_jpeg(self, path: str, device: str = DEFAULT_CAM,
                  width: int = 960, height: int = 720, fps: int = 15,
                  duration: float = 1.5) -> bool:
        """Pull one JPEG frame from the MJPEG stream and write it to `path`.

        Returns True on success, False if no JPEG was found in the buffer.
        """
        url = (f"/api/stream?device={device}&width={width}"
               f"&height={height}&fps={fps}")
        deadline = time.time() + duration
        buf = b""
        with urllib.request.urlopen(self._req(url), timeout=self.timeout) as r:
            while time.time() < deadline:
                chunk = r.read(65536)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 4_000_000:
                    buf = buf[-2_000_000:]
        end = buf.rfind(b"\xff\xd9")
        start = buf.rfind(b"\xff\xd8", 0, end)
        if start < 0 or end < 0:
            return False
        with open(path, "wb") as f:
            f.write(buf[start:end + 2])
        return True
