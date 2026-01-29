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
      - {"cmd":"ping"} -> {"ok": true, "cmd":"ping"}
      - {"cmd":"set_normalized_target","target":[right,left]}
      - {"cmd":"homing"} -> {"ok": true, "cmd":"homing"}
      - {"cmd":"get_normalized_target"} -> {"ok": true, "cmd":"get_normalized_target", "target":[right,left]}
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
        # NOTE: initialize to a sane default so callers can mutate it immediately
        # (e.g. in a control loop that does get_target()[i]=... then set_target()).
        self.target_q: np.typing.NDArray = np.zeros(2, dtype=float)  # normalized target cache

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_lock = threading.Lock()
        if self.timeout is not None:
            try:
                self._sock.settimeout(float(self.timeout))
            except Exception:
                pass

    def _udp_request(self, payload: dict, expect_reply: bool) -> dict | None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            if self.host is None or self.port is None:
                return None
            with self._sock_lock:
                self._sock.sendto(data, (self.host, self.port))
                if not expect_reply:
                    return None

                expected_cmd = payload.get("cmd", None)
                # We want the most recent response whose `cmd` matches the request.
                candidate = None

                # Use the currently configured socket timeout as an upper bound.
                try:
                    sock_timeout = self._sock.gettimeout()
                except Exception:
                    sock_timeout = None
                if sock_timeout is None:
                    sock_timeout = 1.0
                deadline = time.monotonic() + float(sock_timeout)

                recv_count = 0
                max_reads = 100
                while time.monotonic() < deadline and recv_count < max_reads:
                    remaining = max(0.0, deadline - time.monotonic())
                    try:
                        self._sock.settimeout(remaining)
                        raw, _ = self._sock.recvfrom(65535)
                        recv_count += 1
                        try:
                            resp = json.loads(raw.decode("utf-8"))
                        except Exception:
                            continue

                        # If remote doesn't include cmd, fall back to first response.
                        resp_cmd = resp.get("cmd", None) if isinstance(resp, dict) else None
                        if expected_cmd is None or resp_cmd is None:
                            return resp

                        if resp_cmd == expected_cmd:
                            candidate = resp
                            # Drain any already-queued packets without blocking to get the latest.
                            prev_timeout = None
                            try:
                                prev_timeout = self._sock.gettimeout()
                            except Exception:
                                prev_timeout = None
                            try:
                                self._sock.settimeout(0.0)
                                for _ in range(max_reads - recv_count):
                                    try:
                                        raw2, _ = self._sock.recvfrom(65535)
                                        recv_count += 1
                                        try:
                                            resp2 = json.loads(raw2.decode("utf-8"))
                                        except Exception:
                                            continue
                                        if isinstance(resp2, dict) and resp2.get("cmd", None) == expected_cmd:
                                            candidate = resp2
                                    except (socket.timeout, BlockingIOError):
                                        break
                                    except Exception:
                                        break
                            finally:
                                try:
                                    self._sock.settimeout(prev_timeout)
                                except Exception:
                                    pass
                            return candidate
                        # else: ignore non-matching responses and keep reading
                    except socket.timeout:
                        break
                return candidate
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
        if not ok:
            print("[Gripper] Remote homing failed or no response")
            return False
        self.min_q = np.asarray(resp.get("min_q", None), dtype=float).reshape(-1)
        self.max_q = np.asarray(resp.get("max_q", None), dtype=float).reshape(-1)
        print(f"[Gripper] Remote homing success. min_q: {self.min_q}, max_q: {self.max_q}")
        return True

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
        """
        Return the locally cached (last commanded) normalized target.

        Control loops call this frequently; making a network round-trip here adds
        significant latency. If you need the remote-side value, use
        `get_target_remote()`.
        """
        return self.target_q

    def get_target_remote(self):
        """
        Fetch the target from the remote server (slow; network round-trip).
        """
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

    def set_normalized_target(self, normalized_q, wait_for_reply: bool = False):
        """
        Send a normalized target to the remote server.

        For high-rate control, set `wait_for_reply=False` (default) to avoid a
        blocking network round-trip on every command. The local cache is updated
        immediately. If you need reliability/ack, set `wait_for_reply=True`.
        """
        normalized_q = np.asarray(normalized_q, dtype=float).reshape(-1)
        # Update local cache immediately (keeps control loops fast).
        self.target_q = normalized_q

        resp = self._udp_request(
            {"cmd": "set_normalized_target", "normalized_q": normalized_q.tolist(), "ts": time.time()},
            expect_reply=bool(wait_for_reply),
        )
        if wait_for_reply:
            if not resp or not resp.get("ok", False):
                raise RuntimeError("[Gripper] Failed to set remote normalized target (no response or ok=false)")
            # If server provides authoritative target, sync it.
            if resp.get("target", None) is not None:
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