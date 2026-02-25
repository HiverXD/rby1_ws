"""
remote_master_arm.py — Remote PC (노트북)에서 import

master_arm_server.py(Robot PC)와 UDP/JSON으로 통신하는 클라이언트.
remote_gripper.py와 동일한 인터페이스 패턴을 따릅니다.

[사용법]
    from remote_master_arm import RemoteMasterArm

    ma = RemoteMasterArm(host="192.168.0.56", state_port=5010, cmd_port=5011)
    ma.connect()          # ping 확인 + 상태 수신 스레드 시작
    ma.start_gravity()    # 중력 보상 모드 시작

    q, grav, btn_r, btn_l = ma.get_state()

    # Homing (비동기)
    ma.homing(target_right_deg=[...7...], target_left_deg=[...7...])
    ma.wait_homing(timeout=15.0)

    ma.stop()             # 제어 루프 정지
    ma.close()
"""

import json
import socket
import threading
import time
from typing import Optional, Tuple

import numpy as np
import yaml
import os
from pathlib import Path


class RemoteMasterArm:
    """
    master_arm_server.py와 통신하는 UDP 클라이언트.

    state_port: 서버 → 클라이언트 (상태 수신 Listen)
    cmd_port:   클라이언트 → 서버 (명령 전송)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        state_port: Optional[int] = None,
        cmd_port: Optional[int] = None,
        timeout: float = 5.0,
        config_path: Optional[str] = None,
    ):
        # config.yaml에서 설정 로드 (인자가 없으면)
        if config_path is None:
            config_path = os.getenv(
                "RBY1_CONFIG_PATH",
                str(Path(__file__).resolve().parent / "config.yaml"),
            )
        cfg = {}
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

        self.host       = host       or cfg.get("remote_master_arm_host", "192.168.0.56")
        self.state_port = state_port or int(cfg.get("remote_master_arm_state_port", 5010))
        self.cmd_port   = cmd_port   or int(cfg.get("remote_master_arm_cmd_port",   5011))
        self.timeout    = timeout

        # 캐시
        self._lock      = threading.Lock()
        self._q         = np.zeros(14, dtype=float)
        self._gravity   = np.zeros(14, dtype=float)
        self._btn_right = 0
        self._btn_left  = 0
        self._ts        = 0.0
        self._mode      = "idle"
        self._connected = False

        # UDP 소켓
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.settimeout(self.timeout)

        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._state_sock.bind(("0.0.0.0", self.state_port))
        self._state_sock.settimeout(2.0)

        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    # ── 내부: 명령 전송 ─────────────────────────────────────
    def _send_cmd(self, payload: dict, expect_reply: bool = True) -> Optional[dict]:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self._cmd_sock.sendto(data, (self.host, self.cmd_port))
            if not expect_reply:
                return None
            raw, _ = self._cmd_sock.recvfrom(65535)
            return json.loads(raw.decode("utf-8"))
        except socket.timeout:
            return None
        except Exception:
            return None

    # ── 상태 수신 루프 ─────────────────────────────────────
    def _recv_loop(self):
        while self._running:
            try:
                raw, _ = self._state_sock.recvfrom(65535)
                msg = json.loads(raw.decode("utf-8"))
                if msg.get("type") != "state":
                    continue
                with self._lock:
                    self._q         = np.array(msg["q"],       dtype=float)
                    self._gravity   = np.array(msg["gravity"], dtype=float)
                    self._btn_right = int(msg.get("btn_right", 0))
                    self._btn_left  = int(msg.get("btn_left",  0))
                    self._ts        = float(msg.get("ts", time.time()))
                    self._mode      = msg.get("mode", "gravity")
            except socket.timeout:
                continue
            except Exception:
                continue

    # ── 공개 API ─────────────────────────────────────────────
    def connect(self, verbose: bool = True) -> bool:
        """서버에 ping을 보내 연결을 확인하고, 상태 수신 스레드를 시작합니다."""
        resp = self._send_cmd({"cmd": "ping"})
        ok   = bool(resp and resp.get("ok", False))
        if not ok:
            if verbose:
                print(
                    f"[RemoteMasterArm] ❌ 연결 실패 — {self.host}:{self.cmd_port}\n"
                    f"   Robot PC에서 master_arm_server.py가 실행 중인지 확인하세요."
                )
            return False

        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="ma_recv", daemon=True
        )
        self._recv_thread.start()
        self._connected = True

        if verbose:
            print(
                f"[RemoteMasterArm] ✅ 연결 완료\n"
                f"   서버  : {self.host}\n"
                f"   상태  수신 포트 : {self.state_port}\n"
                f"   명령  전송 포트 : {self.cmd_port}"
            )
        return True

    def start_gravity(self) -> bool:
        """서버에 중력 보상 모드 시작을 요청합니다."""
        resp = self._send_cmd({"cmd": "start_gravity"})
        return bool(resp and resp.get("ok", False))

    def stop(self) -> bool:
        """서버에 제어 루프 정지를 요청합니다."""
        resp = self._send_cmd({"cmd": "stop"})
        return bool(resp and resp.get("ok", False))

    def homing(
        self,
        target_right_deg: list,
        target_left_deg: list,
        torque_limit: Optional[list] = None,
        threshold_deg: float = 5.0,
        max_speed_deg_per_sec: Optional[float] = None,
    ) -> bool:
        """
        서버에 homing 명령을 전송합니다 (비동기 — 서버가 백그라운드에서 실행).
        완료를 기다리려면 wait_homing() 을 사용하세요.

        Args:
            max_speed_deg_per_sec: 각 관절의 최대 이동 속도 (deg/sec).
                                   None이면 서버 기본값(30 deg/sec)을 사용합니다.
        """
        payload: dict = {
            "cmd":           "homing",
            "target_right":  list(target_right_deg),
            "target_left":   list(target_left_deg),
            "threshold_deg": threshold_deg,
        }
        if torque_limit is not None:
            payload["torque_limit"] = list(torque_limit)
        if max_speed_deg_per_sec is not None:
            payload["max_speed_deg_per_sec"] = float(max_speed_deg_per_sec)
        resp = self._send_cmd(payload)
        return bool(resp and resp.get("ok", False))

    def wait_homing(self, timeout: float = 15.0) -> bool:
        """homing이 완료될 때까지 블로킹 대기."""
        resp = self._send_cmd({"cmd": "homing_wait", "timeout": timeout},
                              expect_reply=True)
        # homing_wait는 서버측 블로킹 — cmd_sock 타임아웃을 임시로 늘림
        old_timeout = self._cmd_sock.gettimeout()
        self._cmd_sock.settimeout(timeout + 5.0)
        try:
            data = json.dumps(
                {"cmd": "homing_wait", "timeout": timeout}, separators=(",", ":")
            ).encode("utf-8")
            self._cmd_sock.sendto(data, (self.host, self.cmd_port))
            raw, _ = self._cmd_sock.recvfrom(65535)
            msg = json.loads(raw.decode("utf-8"))
            return bool(msg.get("reached", False))
        except socket.timeout:
            return False
        except Exception:
            return False
        finally:
            self._cmd_sock.settimeout(old_timeout)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        최신 캐시된 상태를 반환합니다 (네트워크 접근 없음).
        Returns: (q_joint(14,), gravity_term(14,), btn_right, btn_left)
        """
        with self._lock:
            return (
                self._q.copy(),
                self._gravity.copy(),
                self._btn_right,
                self._btn_left,
            )

    def get_q(self) -> np.ndarray:
        with self._lock:
            return self._q.copy()

    def get_timestamp(self) -> float:
        with self._lock:
            return self._ts

    def is_stale(self, max_age_secs: float = 1.0) -> bool:
        """마지막 상태 수신으로부터 max_age_secs 이상 경과하면 True."""
        with self._lock:
            return (time.time() - self._ts) > max_age_secs

    def close(self):
        self._running = False
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        try:
            self._state_sock.close()
        except Exception:
            pass
        self._connected = False
