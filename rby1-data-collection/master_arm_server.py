"""
master_arm_server.py — Robot PC (UPC)에서 실행

Master Arm이 로봇 PC에 USB로 연결되어 있을 때,
UDP/JSON 소켓을 통해 Remote PC(노트북)에 관절 상태를 스트리밍하고
명령(homing, set_gravity_compensation 등)을 수신합니다.

[프로토콜]
  서버 → 클라이언트 (브로드캐스트, STATE_PORT):
    {"type":"state", "q":[...14...], "gravity":[...14...],
     "btn_right": 0|1, "btn_left": 0|1, "ts": float}

  클라이언트 → 서버 (CMD_PORT):
    {"cmd": "ping"}
    {"cmd": "start_gravity"}            # 중력 보상 모드 시작
    {"cmd": "stop"}                     # 제어 루프 정지
    {"cmd": "homing",                   # 초기 자세 이동
     "target_right": [...7 deg...],
     "target_left":  [...7 deg...],
     "torque_limit": [...14...],        # optional
     "threshold_deg": float}            # optional
    {"cmd": "get_state"}                # 현재 상태 즉시 반환

[실행 방법 — Robot PC에서]
  python master_arm_server.py [--device /dev/rby1_master_arm]
                              [--urdf /path/to/model.urdf]
                              [--state-port 5010]
                              [--cmd-port 5011]
                              [--client-host 192.168.0.27]
"""

import argparse
import json
import logging
import os
import signal
import socket
import threading
import time
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger("master_arm_server")

# ── 기본 설정 ──────────────────────────────────────────────
DEFAULT_DEVICE     = "/dev/rby1_master_arm"
DEFAULT_URDF       = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../rby1-sdk/models/master_arm/model.urdf",
)
DEFAULT_STATE_PORT  = 5010   # 서버→클라이언트 상태 스트리밍 포트
DEFAULT_CMD_PORT    = 5011   # 클라이언트→서버 명령 수신 포트
DEFAULT_CONTROL_DT  = 0.01   # 100 Hz
DEFAULT_GRAVITY_GAIN = 1.3   # 중력 보상 배율 (1.0 = 원본, >1 = 강하게)


# ══════════════════════════════════════════════════════════
# MasterArmServer
# ══════════════════════════════════════════════════════════

class MasterArmServer:
    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        urdf_path: str = DEFAULT_URDF,
        state_port: int = DEFAULT_STATE_PORT,
        cmd_port: int = DEFAULT_CMD_PORT,
        client_host: Optional[str] = None,  # None = broadcast
        control_dt: float = DEFAULT_CONTROL_DT,
        gravity_gain: float = DEFAULT_GRAVITY_GAIN,
    ):
        self.device       = device
        self.urdf_path    = os.path.abspath(urdf_path)
        self.state_port   = state_port
        self.cmd_port     = cmd_port
        self.client_host  = client_host
        self.control_dt   = control_dt
        self._gravity_gain = float(gravity_gain)  # 중력 보상 배율

        # 상태
        self._lock           = threading.Lock()
        self._q              = np.zeros(14, dtype=float)
        self._gravity        = np.zeros(14, dtype=float)
        self._btn_right      = 0   # 오른쪽 그리퍼 트리거 (.trigger)
        self._btn_left       = 0   # 왼쪽  그리퍼 트리거 (.trigger)
        self._unlock_right   = 0   # 오른쪽 잠금 해제 버튼 (.button)
        self._unlock_left    = 0   # 왼쪽  잠금 해제 버튼 (.button)
        self._ts             = 0.0
        self._mode           = "gravity"   # "gravity" | "homing" | "hold" | "idle"

        # homing 목표
        self._homing_target       = np.zeros(14, dtype=float)
        self._homing_interp_target = np.zeros(14, dtype=float)  # 보간 중간 목표
        self._homing_torque       = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
        # 위치 유지 모드 최대 전류 한계(A): 중력 기반 hold 토크 상한값
        # 상위 관절(어깨)은 중력 부하가 크므로 더 높게 설정
        self._hold_torque_max    = np.array([6.0, 6.0, 6.0, 3.5, 3.0, 2.5, 2.0,
                                             6.0, 6.0, 6.0, 3.5, 3.0, 2.5, 2.0])
        # hold 토크 여유분(A): 중력 보상값 + 이 마진으로 외란에도 안정적 유지
        self._hold_torque_margin = 1.5
        self._homing_thresh       = np.deg2rad(5.0)
        self._homing_max_speed    = np.deg2rad(30.0)  # rad/sec (기본 30 deg/sec)
        self._homing_done         = threading.Event()

        # gravity 모드 hold 위치 (SDK 예시 패턴: 버튼 OFF → 위치 유지)
        # 잠금 해제 버튼이 눌리지 않을 때 팔을 고정할 관절 위치 (7 DOF each)
        self._hold_q_right = np.zeros(7, dtype=float)
        self._hold_q_left  = np.zeros(7, dtype=float)

        # SDK 17_teleoperation_with_joint_mapping 패턴: 중력 보상 + 관절 한계 배리어 + 점성 감쇠
        self._ma_q_limit_barrier = 0.5
        self._ma_min_q = np.deg2rad(
            [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
        )
        self._ma_max_q = np.deg2rad(
            [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
        )
        self._ma_torque_limit = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
        self._ma_viscous_gain = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)

        self._master_arm = None
        self._running    = False

        # UDP 소켓
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._cmd_sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.bind(("0.0.0.0", cmd_port))
        self._cmd_sock.settimeout(1.0)
        self._clients    = {}   # addr → last_seen

    # ── Master Arm 초기화 ───────────────────────────────────
    def _init_master_arm(self):
        try:
            import rby1_sdk as rby
            self._rby = rby
        except ImportError:
            raise RuntimeError("rby1_sdk를 import할 수 없습니다. Robot PC에서 실행하세요.")

        logger.info(f"initialize_device({self.device})")
        try:
            rby.upc.initialize_device(self.device)
        except RuntimeError as e:
            if "latency_timer" in str(e):
                logger.warning(f"latency_timer 설정 실패 (무시): {e}")
            else:
                raise

        ma = rby.upc.MasterArm(self.device)
        ma.set_model_path(self.urdf_path)
        ma.set_control_period(self.control_dt)

        active_ids = ma.initialize(verbose=True)
        expected   = rby.upc.MasterArm.DeviceCount
        if len(active_ids) != expected:
            raise RuntimeError(
                f"Master Arm 장치 수 불일치: 감지={len(active_ids)}, 예상={expected}"
            )

        self._master_arm = ma
        logger.info(f"✅ Master Arm 초기화 완료 (장치 수={len(active_ids)})")

    # ── 제어 콜백 ───────────────────────────────────────────
    def _control_cb(self, ma_state):
        rby = self._rby
        q    = np.array(ma_state.q_joint,      dtype=float)
        qvel = np.array(ma_state.qvel_joint,   dtype=float)
        grav = np.array(ma_state.gravity_term,  dtype=float)
        # .trigger = 그리퍼 open/close 트리거  (btn_right/btn_left로 전달)
        # .button  = 잠금 해제 버튼            (unlock_right/unlock_left로 전달)
        r_trigger     = int(ma_state.button_right.trigger)
        l_trigger     = int(ma_state.button_left.trigger)
        r_unlock      = int(ma_state.button_right.button)
        l_unlock      = int(ma_state.button_left.button)

        with self._lock:
            self._q            = q
            self._gravity      = grav
            self._btn_right    = r_trigger
            self._btn_left     = l_trigger
            self._unlock_right = r_unlock
            self._unlock_left  = l_unlock
            self._ts           = time.time()
            mode               = self._mode

        inp = rby.upc.MasterArm.ControlInput()

        if mode == "homing":
            with self._lock:
                target    = self._homing_target.copy()
                torque    = self._homing_torque.copy()
                thresh    = self._homing_thresh
                max_speed = self._homing_max_speed

            max_err = float(np.max(np.abs(q - target)))
            if max_err < thresh:
                with self._lock:
                    self._mode = "hold"  # gravity 아닌 hold 모드로 전환
                self._homing_done.set()
                logger.info(
                    f"[Homing] 완료 → 위치 유지(hold) 모드. 최대 오차: {np.rad2deg(max_err):.2f}°\n"
                    f"  start_gravity() 호출(Step 6) 전까지 목표 자세를 능동 유지합니다."
                )
                # 수렴 직후 한 스텝도 목표 위치를 능동 유지
                inp.target_operating_mode.fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                inp.target_torque[:] = torque
                inp.target_position[:] = target
            else:
                # 보간 목표를 max_speed에 맞게 한 스텝씩 전진
                max_step = max_speed * self.control_dt
                with self._lock:
                    remaining = target - self._homing_interp_target
                    step = np.clip(remaining, -max_step, max_step)
                    self._homing_interp_target += step
                    interp_target = self._homing_interp_target.copy()

                inp.target_operating_mode.fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                inp.target_torque[:]   = torque
                inp.target_position[:] = interp_target

        elif mode == "hold":
            # homing 완료 후 목표 자세를 위치 제어로 능동 유지
            # start_gravity() 호출(Step 6) 시 gravity 모드로 전환됨
            with self._lock:
                hold_pos    = self._homing_target.copy()
                hold_torque = self._homing_torque.copy()
            inp.target_operating_mode.fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            inp.target_torque[:] = hold_torque
            inp.target_position[:] = hold_pos

        elif mode == "idle":
            # 정지 모드: unlock 버튼 상태에 관계없이 모든 관절을
            # 현재 위치에서 균일하게 위치 제어로 고정
            # (일부 관절만 풀리는 불일치 현상 방지)
            inp.target_operating_mode.fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            inp.target_torque[:] = self._homing_torque
            inp.target_position[:] = q   # 현재 관절 위치를 목표로 설정

        else:  # "gravity"
            # SDK 17_teleoperation_with_joint_mapping 패턴:
            #   토크 = gravity_term + q_limit_barrier*(barrier) + viscous_gain*qvel
            #   버튼 누름(unlock=1): CurrentControlMode + 위 토크 → 팔이 자유롭게 뜸
            #   버튼 OFF (unlock=0): CurrentBasedPositionControlMode + 마지막 위치 → 팔 고정
            #
            # SDK 원본 코드 그대로 적용 (gravity_term*gain만 사용하면 점성 감쇠가
            # 없어서 팔이 불안정하고 버튼 해제→누름 전환 시 로봇이 튀는 현상 발생)
            torque = (
                grav
                + self._ma_q_limit_barrier
                * (
                    np.maximum(self._ma_min_q - q, 0)
                    + np.minimum(self._ma_max_q - q, 0)
                )
                + self._ma_viscous_gain * qvel
            )
            torque = np.clip(torque, -self._ma_torque_limit, self._ma_torque_limit)

            # ── 오른팔 (joints 0–6) ─────────────────────────────
            if r_unlock:
                inp.target_operating_mode[:7].fill(rby.DynamixelBus.CurrentControlMode)
                inp.target_torque[:7] = torque[:7]
                with self._lock:
                    self._hold_q_right = q[:7].copy()   # 버튼 누르는 동안 계속 갱신
            else:
                inp.target_operating_mode[:7].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                with self._lock:
                    hold_r = self._hold_q_right.copy()
                inp.target_position[:7] = hold_r
                inp.target_torque[:7]   = self._ma_torque_limit[:7]

            # ── 왼팔 (joints 7–13) ─────────────────────────────
            if l_unlock:
                inp.target_operating_mode[7:].fill(rby.DynamixelBus.CurrentControlMode)
                inp.target_torque[7:] = torque[7:]
                with self._lock:
                    self._hold_q_left = q[7:].copy()    # 버튼 누르는 동안 계속 갱신
            else:
                inp.target_operating_mode[7:].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                with self._lock:
                    hold_l = self._hold_q_left.copy()
                inp.target_position[7:] = hold_l
                inp.target_torque[7:]   = self._ma_torque_limit[7:]

        return inp

    # ── 상태 브로드캐스트 스레드 ───────────────────────────
    def _broadcast_loop(self):
        logger.info(f"상태 브로드캐스트 시작 (포트: {self.state_port})")
        while self._running:
            with self._lock:
                q            = self._q.tolist()
                gravity      = self._gravity.tolist()
                r_btn        = self._btn_right
                l_btn        = self._btn_left
                r_unlock     = self._unlock_right
                l_unlock     = self._unlock_left
                ts           = self._ts
                mode         = self._mode

            payload = json.dumps({
                "type":         "state",
                "q":            q,
                "gravity":      gravity,
                "btn_right":    r_btn,      # 그리퍼 트리거 (.trigger)
                "btn_left":     l_btn,      # 그리퍼 트리거 (.trigger)
                "unlock_right": r_unlock,   # 잠금 해제 버튼 (.button)
                "unlock_left":  l_unlock,   # 잠금 해제 버튼 (.button)
                "ts":           ts,
                "mode":         mode,
            }, separators=(",", ":")).encode("utf-8")

            # 등록된 클라이언트들에게 전송
            with self._lock:
                clients = dict(self._clients)

            if clients:
                for addr in list(clients.keys()):
                    try:
                        self._state_sock.sendto(payload, (addr, self.state_port))
                    except Exception as e:
                        logger.warning(f"브로드캐스트 실패 ({addr}): {e}")
            else:
                # 클라이언트 미등록 시 브로드캐스트
                try:
                    self._state_sock.sendto(payload, ("<broadcast>", self.state_port))
                except Exception:
                    pass

            time.sleep(self.control_dt)

    # ── 명령 수신 스레드 ────────────────────────────────────
    def _cmd_loop(self):
        logger.info(f"명령 수신 대기 (포트: {self.cmd_port})")
        while self._running:
            try:
                data, addr = self._cmd_sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.warning(f"명령 수신 오류: {e}")
                continue

            # 클라이언트 등록
            client_ip = addr[0]
            with self._lock:
                self._clients[client_ip] = time.time()

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            cmd = msg.get("cmd", "")
            reply = self._handle_cmd(cmd, msg, addr)
            if reply:
                try:
                    self._cmd_sock.sendto(
                        json.dumps(reply, separators=(",", ":")).encode("utf-8"),
                        addr,
                    )
                except Exception as e:
                    logger.warning(f"응답 전송 실패: {e}")

    def _handle_cmd(self, cmd: str, msg: dict, addr) -> Optional[dict]:
        if cmd == "ping":
            logger.info(f"ping from {addr[0]}")
            return {"ok": True, "cmd": "ping"}

        elif cmd == "get_state":
            with self._lock:
                q        = self._q.tolist()
                gravity  = self._gravity.tolist()
                r_btn    = self._btn_right
                l_btn    = self._btn_left
                r_unlock = self._unlock_right
                l_unlock = self._unlock_left
                ts       = self._ts
                mode     = self._mode
            return {
                "ok": True, "cmd": "get_state",
                "q": q, "gravity": gravity,
                "btn_right": r_btn, "btn_left": l_btn,
                "unlock_right": r_unlock, "unlock_left": l_unlock,
                "ts": ts, "mode": mode,
            }

        elif cmd == "start_gravity":
            with self._lock:
                # gravity 모드 진입 시 현재 관절 위치를 hold 기준으로 초기화
                # (버튼을 처음 누르기 전까지 이 위치를 유지)
                self._hold_q_right = self._q[:7].copy()
                self._hold_q_left  = self._q[7:].copy()
                self._mode = "gravity"
            logger.info(f"[{addr[0]}] 중력 보상 모드 시작 (per-arm deadman switch 활성화)")
            return {"ok": True, "cmd": "start_gravity"}

        elif cmd == "stop":
            with self._lock:
                self._mode = "idle"
            logger.info(f"[{addr[0]}] 제어 정지 요청")
            return {"ok": True, "cmd": "stop"}

        elif cmd == "homing":
            target_right = np.deg2rad(msg.get("target_right",
                                               np.rad2deg(self._homing_target[:7]).tolist()))
            target_left  = np.deg2rad(msg.get("target_left",
                                               np.rad2deg(self._homing_target[7:]).tolist()))
            torque_limit          = msg.get("torque_limit", None)
            thresh_deg            = float(msg.get("threshold_deg", 5.0))
            max_speed_deg_per_sec = float(msg.get("max_speed_deg_per_sec", 30.0))

            with self._lock:
                self._homing_target        = np.concatenate([target_right, target_left])
                self._homing_interp_target = self._q.copy()  # 현재 위치에서 출발
                if torque_limit is not None:
                    self._homing_torque = np.array(torque_limit, dtype=float)
                self._homing_thresh     = np.deg2rad(thresh_deg)
                self._homing_max_speed  = np.deg2rad(max_speed_deg_per_sec)
                self._mode              = "homing"
            self._homing_done.clear()
            logger.info(
                f"[{addr[0]}] Homing 시작 — 목표 오른팔: "
                f"{np.round(np.rad2deg(target_right), 1)}"
            )
            return {"ok": True, "cmd": "homing", "status": "started"}

        elif cmd == "homing_wait":
            timeout = float(msg.get("timeout", 15.0))
            reached = self._homing_done.wait(timeout=timeout)
            return {"ok": True, "cmd": "homing_wait", "reached": reached}

        elif cmd == "set_gravity_gain":
            gain = float(msg.get("gain", DEFAULT_GRAVITY_GAIN))
            gain = float(np.clip(gain, 0.5, 3.0))  # 안전 범위 제한
            with self._lock:
                self._gravity_gain = gain
            logger.info(f"[{addr[0]}] gravity_gain → {gain:.2f}")
            return {"ok": True, "cmd": "set_gravity_gain", "gain": gain}

        elif cmd == "get_gravity_gain":
            with self._lock:
                gain = self._gravity_gain
            return {"ok": True, "cmd": "get_gravity_gain", "gain": gain}

        elif cmd == "set_hold_torque_margin":
            # hold 모드 추가 여유 전류(A) 설정
            # gravity 기반 hold_torque = |grav|*gain + margin
            margin = float(msg.get("margin", 1.5))
            margin = float(np.clip(margin, 0.0, 5.0))  # 안전 범위 제한
            with self._lock:
                self._hold_torque_margin = margin
            logger.info(f"[{addr[0]}] hold_torque_margin → {margin:.2f} A")
            return {"ok": True, "cmd": "set_hold_torque_margin", "margin": margin}

        elif cmd == "get_hold_torque_margin":
            with self._lock:
                margin = self._hold_torque_margin
            return {"ok": True, "cmd": "get_hold_torque_margin", "margin": margin}

        else:
            logger.warning(f"알 수 없는 명령: {cmd}")
            return {"ok": False, "cmd": cmd, "error": "unknown command"}

    # ── 서버 시작/종료 ──────────────────────────────────────
    def start(self):
        self._init_master_arm()
        self._running = True

        # 중력 보상 제어 루프 시작
        self._master_arm.start_control(self._control_cb)
        with self._lock:
            self._mode = "gravity"
        logger.info("Master Arm 제어 루프 시작 (중력 보상 모드)")

        # 브로드캐스트 스레드
        self._bcast_thread = threading.Thread(
            target=self._broadcast_loop, name="bcast", daemon=True
        )
        self._bcast_thread.start()

        # 명령 수신 스레드
        self._cmd_thread = threading.Thread(
            target=self._cmd_loop, name="cmd", daemon=True
        )
        self._cmd_thread.start()

        logger.info(
            f"✅ MasterArmServer 시작\n"
            f"   상태 스트리밍 포트 : {self.state_port}\n"
            f"   명령 수신 포트     : {self.cmd_port}\n"
            f"   장치               : {self.device}\n"
        )

    def stop(self):
        self._running = False
        if self._master_arm is not None:
            try:
                self._master_arm.stop_control()
            except Exception as e:
                logger.warning(f"제어 루프 정지 오류: {e}")
        try:
            self._state_sock.close()
        except Exception:
            pass
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        logger.info("MasterArmServer 종료")

    def wait(self):
        """Ctrl+C까지 블로킹."""
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Ctrl+C 감지 — 종료 중...")
            self.stop()


# ══════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Master Arm UDP 서버 (Robot PC에서 실행)")
    parser.add_argument("--device",       default=DEFAULT_DEVICE,     help="Master Arm 장치 경로")
    parser.add_argument("--urdf",         default=DEFAULT_URDF,       help="Master Arm URDF 경로")
    parser.add_argument("--state-port",   type=int, default=DEFAULT_STATE_PORT, help="상태 스트리밍 포트")
    parser.add_argument("--cmd-port",     type=int, default=DEFAULT_CMD_PORT,   help="명령 수신 포트")
    parser.add_argument("--client-host",  default=None, help="Remote PC IP (지정하면 unicast)")
    parser.add_argument("--control-dt",   type=float, default=DEFAULT_CONTROL_DT, help="제어 주기 (초)")
    parser.add_argument("--gravity-gain", type=float, default=DEFAULT_GRAVITY_GAIN,
                        help=f"중력 보상 배율 (기본 {DEFAULT_GRAVITY_GAIN}, 범위 0.5~3.0)")
    args = parser.parse_args()

    server = MasterArmServer(
        device       = args.device,
        urdf_path    = args.urdf,
        state_port   = args.state_port,
        cmd_port     = args.cmd_port,
        client_host  = args.client_host,
        control_dt   = args.control_dt,
        gravity_gain = args.gravity_gain,
    )

    def _sig(signum, frame):
        server.stop()
        import sys; sys.exit(0)

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    server.start()
    server.wait()


if __name__ == "__main__":
    main()
