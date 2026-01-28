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
      - {"cmd":"set_normalized_target","target":[right,left]}
      - {"cmd":"homing"} -> {"ok": true}
      - {"cmd":"get_normalized_target"} -> {"ok": true, "target":[right,left]}
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

        # In remote mode we store normalized targets directly and return cached values.
        self.target_q: np.typing.NDArray = None  # normalized target cache

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
        self._udp_request({"cmd": "initialize", "ts": time.time()}, expect_reply=False)
        resp = self._udp_request({"cmd": "ping", "ts": time.time()}, expect_reply=True)
        ok = bool(resp and resp.get("ok", False))
        if verbose:
            print(f"[Gripper] Remote gripper ping ({self.host}:{self.port}) -> {ok}")
        return ok

    def set_operating_mode(self, mode):
        resp = self._udp_request({"cmd": "set_operating_mode", "mode": mode, "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to set remote operating mode (no response or ok=false)")

    def homing(self):
        resp = self._udp_request({"cmd": "homing", "ts": time.time()}, expect_reply=True)
        ok = bool(resp and resp.get("ok", False))
        self.min_q = np.asarray(resp.get("min_q", None), dtype=float).reshape(-1)
        self.max_q = np.asarray(resp.get("max_q", None), dtype=float).reshape(-1)
        if not ok:
            print("[Gripper] Remote homing failed or no response")
        return ok

    def start(self):
        resp = self._udp_request({"cmd": "start", "ts": time.time()},expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to start remote gripper loop (no response or ok=false)")

    def stop(self):
        resp = self._udp_request({"cmd": "stop", "ts": time.time()},expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to stop remote gripper loop (no response or ok=false)")

    def loop(self):
        pass

    def get_target(self):
        resp = self._udp_request({"cmd": "get_target", "ts": time.time()}, expect_reply=True)
        if not resp or resp.get("target", None) is None:
            raise RuntimeError("[Gripper] Failed to get remote target (no response or target is None)")
        self.target_q = np.asarray(resp.get("target", None), dtype=float).reshape(-1)
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
        normalized_target = np.asarray(target, dtype=float).reshape(-1)
        return normalized_target

    def set_normalized_target(self, normalized_q):
        resp = self._udp_request({"cmd": "set_normalized_target", "normalized_q": normalized_q.tolist(), "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to set remote normalized target (no response or ok=false)")
        self.target_q = np.asarray(resp.get("target", None), dtype=float).reshape(-1)
        

    def get_state(self):
        resp = self._udp_request({"cmd": "get_state", "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to get remote state (no response or ok=false)")
        state = resp.get("state", None)
        if state is None:
            raise RuntimeError("[Gripper] Remote state missing in response")
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size != 2:
            raise RuntimeError(f"[Gripper] Remote state invalid shape: {state.shape}")
        return state