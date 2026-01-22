import json
import socket
import threading
import time

import numpy as np
import yaml


class Gripper:
    """
    This implementation sends commands to a remote host over UDP using JSON payloads.

    Expected remote protocol (UDP/JSON):
      - {"cmd":"ping"} -> {"ok": true}
      - {"cmd":"set_normalized_target","q":[right,left]}
      - {"cmd":"get_state"} -> {"ok": true, "state":[right,left]}
      - {"cmd":"homing"} -> {"ok": true}
    """

    # Set to True to keep incoming normalized targets as-is (no inversion).
    GRIPPER_DIRECTION = False

    def __init__(self):
        # Resolve remote target (priority: config.yaml -> env -> defaults)
        host = None
        port = None
        timeout = None
        try:
            with open("rby1-data-collection/config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                host = cfg.get("remote_gripper_host", None)
                port = cfg.get("remote_gripper_port", None)
                timeout = cfg.get("remote_gripper_timeout", None)
        except Exception:
            print("Failed to load remote gripper config from config.yaml")
            pass

        self.host = host
        self.port = port
        self.timeout = timeout

        # Keep the same attributes other code may touch.
        # In remote mode we store normalized targets directly and return cached values.
        self.target_q = np.array([0.0, 0.0], dtype=float)  # normalized target cache
        self._running = False
        self._thread = None

    def _udp_request(self, payload: dict, expect_reply: bool) -> dict | None:
        data = json.dumps(payload).encode("utf-8")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout)
                sock.sendto(data, (self.host, self.port))
                if not expect_reply:
                    return None
                resp, _ = sock.recvfrom(65535)
            return json.loads(resp.decode("utf-8"))
        except Exception:
            return None

    def initialize(self, verbose=False):
        resp = self._udp_request({"cmd": "ping", "ts": time.time()}, expect_reply=True)
        ok = bool(resp and resp.get("ok", False))
        if verbose:
            print(f"[Gripper] Remote gripper ping ({self.host}:{self.port}) -> {ok}")
        return ok

    def set_operating_mode(self, mode):
        # Not applicable in remote client mode (handled by remote server).
        _ = mode
        return

    def homing(self):
        resp = self._udp_request({"cmd": "homing", "ts": time.time()}, expect_reply=True)
        ok = bool(resp and resp.get("ok", False))
        self.min_q = np.asarray(resp.get("min_q", None), dtype=float).reshape(-1)
        self.max_q = np.asarray(resp.get("max_q", None), dtype=float).reshape(-1)
        if not ok:
            print("[Gripper] Remote homing failed or no response")
        return ok

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def loop(self):
        # Periodically re-send the last target to the remote server for robustness.
        while self._running:
            if self.target_q is not None:
                q = np.asarray(self.target_q, dtype=float).reshape(-1)
                if q.size == 2:
                    q = np.clip(q, 0.0, 1.0)
                    self._udp_request(
                        {"cmd": "set_normalized_target", "q": q.tolist(), "ts": time.time()},
                        expect_reply=False,
                    )
            time.sleep(0.1)

    def get_target(self):
        return self.target_q
    
    def get_normalized_target(self):
        """
        Return the *remote* gripper normalized target.

        No local-cache fallback: if the remote server doesn't respond or returns
        invalid data, raise so callers notice immediately.
        """
        resp = self._udp_request({"cmd": "get_normalized_target", "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to fetch remote normalized target (no response or ok=false)")
        target = resp.get("target", None)
        
        if target is None:
            raise RuntimeError("[Gripper] Remote normalized target missing in response")
        arr = np.asarray(target, dtype=float).reshape(-1)
        if arr.size != 2:
            raise RuntimeError(f"[Gripper] Remote normalized target invalid shape: {arr.shape}")
        arr = np.clip(arr, 0.0, 1.0)
        self.target_q = arr.copy()
        
        return arr

    def set_normalized_target(self, normalized_q):
        # self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            print("[Gripper] Cannot set target. min_q or max_q is not valid.")
            return
        
        if Gripper.GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q


    def get_state(self):
        """
        Read current gripper state from remote server.
        Returns:
            np.ndarray: normalized positions for each servo [right, left] or None
        """
        resp = self._udp_request({"cmd": "get_state", "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            return None
        state = resp.get("state", None)
        if state is None:
            return None
        try:
            arr = np.asarray(state, dtype=float).reshape(-1)
            if arr.size != 2:
                return None
            return arr
        except Exception:
            return None
