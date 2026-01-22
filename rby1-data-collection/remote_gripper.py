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
    """

    # Set to True to keep incoming normalized targets as-is (no inversion).
    GRIPPER_DIRECTION = True

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
        self.min_q = np.array([0.0, 0.0], dtype=float)
        self.max_q = np.array([1.0, 1.0], dtype=float)
        self.target_q = np.array([1.0, 1.0], dtype=float)  # normalized target cache
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
        # Homing is handled on the remote server (gripper PC). Keep API compatibility.
        return True

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
        # In remote mode target_q is already normalized [0..1].
        return np.asarray(self.target_q, dtype=float).copy()

    def set_normalized_target(self, normalized_q):
        q = np.asarray(normalized_q, dtype=float).reshape(-1)
        if q.size != 2:
            raise ValueError(f"[Gripper] normalized_q must be size 2, got shape {q.shape}")

        q = np.clip(q, 0.0, 1.0)
        if not Gripper.GRIPPER_DIRECTION:
            q = np.array([1.0 - q[0], 1.0 - q[1]], dtype=float)

        self.target_q = q.copy()
        self._udp_request(
            {"cmd": "set_normalized_target", "q": q.tolist(), "ts": time.time()},
            expect_reply=False,
        )


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
