"""
gripper_server.py — Robot PC (UPC)에서 실행

Dynamixel 그리퍼가 Robot PC에 USB로 연결되어 있을 때,
UDP/JSON 소켓을 통해 Remote PC(노트북)의 remote_gripper.py 클라이언트에
그리퍼 제어 인터페이스를 제공합니다.

[프로토콜 — remote_gripper.py 클라이언트와 호환]
  클라이언트 → 서버 (CMD_PORT 수신):
    {"cmd": "ping"}
    {"cmd": "initialize"}
    {"cmd": "homing"}
    {"cmd": "start"}
    {"cmd": "stop"}
    {"cmd": "get_target"}
    {"cmd": "get_normalized_target"}
    {"cmd": "set_normalized_target", "normalized_q": [right, left]}
    {"cmd": "set_operating_mode", "mode": int}
    {"cmd": "get_state"}

  서버 → 클라이언트 (응답):
    {"ok": true,  "cmd": "<echo>", ...}
    {"ok": false, "cmd": "<echo>", "error": "..."}

[실행 방법 — Robot PC에서]
  python gripper_server.py [--port 5009] [--host 0.0.0.0]

[참고]
  - config.yaml: remote_gripper_host, remote_gripper_port
  - Remote PC 클라이언트: remote_gripper.py
"""

import argparse
import json
import logging
import signal
import socket
import sys
import threading
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger("gripper_server")

DEFAULT_PORT = 5009
DEFAULT_HOST = "0.0.0.0"


# ══════════════════════════════════════════════════════════
# GripperServer
# ══════════════════════════════════════════════════════════

class GripperServer:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        self.host = host
        self.port = port

        # gripper.py의 Gripper 클래스 (rby.DynamixelBus 로컬 USB 사용)
        # import는 start() 시점에 수행 (Robot PC에서만 가능)
        self._gripper = None
        self._initialized = False

        self._lock    = threading.Lock()
        self._running = False

        # UDP 소켓
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(1.0)

        logger.info(f"GripperServer UDP 소켓 바인드: {host}:{port}")

    def _init_gripper(self):
        """gripper.py의 Gripper 클래스를 사용해 로컬 Dynamixel 그리퍼를 초기화합니다."""
        try:
            # gripper.py가 같은 디렉터리에 있어야 합니다
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
            from gripper import Gripper
        except ImportError as e:
            raise RuntimeError(f"gripper.py import 실패: {e}")

        logger.info("Gripper 초기화 중 (rby.DynamixelBus)...")
        g = Gripper()
        ok = g.initialize(verbose=True)
        if not ok:
            raise RuntimeError("Dynamixel 그리퍼 초기화 실패 — 장치 연결을 확인하세요.")

        self._gripper      = g
        self._initialized  = True
        logger.info("✅ Gripper 초기화 완료")
        return g

    def _reply(self, sock: socket.socket, addr, payload: dict):
        try:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            sock.sendto(data, addr)
        except Exception as e:
            logger.warning(f"[reply] 응답 전송 실패 ({addr}): {e}")

    def _handle_cmd(self, cmd_str: str, addr):
        """JSON 명령을 파싱하여 처리하고 응답을 반환합니다."""
        try:
            req = json.loads(cmd_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 오류: {e}")
            return {"ok": False, "error": f"JSON parse error: {e}"}

        cmd = req.get("cmd", "")
        logger.debug(f"[cmd] {cmd} from {addr}")

        g = self._gripper  # 현재 그리퍼 핸들 (None 가능)

        # ── ping ──────────────────────────────────────────────
        if cmd == "ping":
            return {"ok": True, "cmd": "ping", "initialized": self._initialized}

        # ── initialize ────────────────────────────────────────
        elif cmd == "initialize":
            if self._initialized and g is not None:
                logger.info("[initialize] 이미 초기화됨 — 재사용")
                return None   # remote_gripper.py: expect_reply=False → 응답 불필요
            try:
                self._init_gripper()
                logger.info("[initialize] 완료")
            except Exception as e:
                logger.error(f"[initialize] 실패: {e}")
            return None  # expect_reply=False

        # ── homing ────────────────────────────────────────────
        elif cmd == "homing":
            if g is None:
                try:
                    self._init_gripper()
                    g = self._gripper
                except Exception as e:
                    return {"ok": False, "cmd": "homing", "error": str(e)}
            try:
                logger.info("[homing] 시작...")
                ok = g.homing()
                if ok:
                    logger.info(f"[homing] 완료 — min_q={g.min_q}, max_q={g.max_q}")
                    return {
                        "ok":    True,
                        "cmd":   "homing",
                        "min_q": g.min_q.tolist(),
                        "max_q": g.max_q.tolist(),
                    }
                else:
                    return {"ok": False, "cmd": "homing", "error": "homing returned False"}
            except Exception as e:
                logger.error(f"[homing] 오류: {e}")
                return {"ok": False, "cmd": "homing", "error": str(e)}

        # ── start ─────────────────────────────────────────────
        elif cmd == "start":
            if g is None:
                return {"ok": False, "cmd": "start", "error": "그리퍼 미초기화"}
            try:
                g.start()
                logger.info("[start] 제어 루프 시작")
                return {"ok": True, "cmd": "start"}
            except Exception as e:
                logger.error(f"[start] 오류: {e}")
                return {"ok": False, "cmd": "start", "error": str(e)}

        # ── stop ──────────────────────────────────────────────
        elif cmd == "stop":
            if g is None:
                return {"ok": False, "cmd": "stop", "error": "그리퍼 미초기화"}
            try:
                g.stop()
                logger.info("[stop] 제어 루프 정지")
                return {"ok": True, "cmd": "stop"}
            except Exception as e:
                logger.error(f"[stop] 오류: {e}")
                return {"ok": False, "cmd": "stop", "error": str(e)}

        # ── get_target ────────────────────────────────────────
        elif cmd == "get_target":
            if g is None or g.target_q is None:
                return {"ok": False, "cmd": "get_target", "error": "target 없음"}
            return {
                "ok":     True,
                "cmd":    "get_target",
                "target": g.target_q.tolist(),
            }

        # ── get_normalized_target ─────────────────────────────
        elif cmd == "get_normalized_target":
            if g is None:
                return {"ok": False, "cmd": "get_normalized_target", "error": "그리퍼 미초기화"}
            try:
                with self._lock:
                    nt = g.get_normalized_target()
                if nt is None:
                    return {"ok": False, "cmd": "get_normalized_target", "error": "target 없음"}
                return {
                    "ok":     True,
                    "cmd":    "get_normalized_target",
                    "target": nt.tolist(),
                }
            except Exception as e:
                return {"ok": False, "cmd": "get_normalized_target", "error": str(e)}

        # ── set_normalized_target ─────────────────────────────
        elif cmd == "set_normalized_target":
            if g is None:
                return None  # 높은 빈도로 호출되므로 로그 생략
            nq = req.get("normalized_q", None)
            if nq is None:
                return None
            try:
                normalized_q = np.asarray(nq, dtype=float).reshape(-1)
                with self._lock:
                    g.set_normalized_target(normalized_q)
                # remote_gripper.py: wait_for_reply=False(기본) → 응답 불필요
                # wait_for_reply=True인 경우에도 아래 응답 반환
                return {
                    "ok":     True,
                    "target": g.target_q.tolist() if g.target_q is not None else None,
                }
            except Exception as e:
                logger.warning(f"[set_normalized_target] 오류: {e}")
                return None

        # ── set_operating_mode ────────────────────────────────
        elif cmd == "set_operating_mode":
            if g is None:
                return {"ok": False, "cmd": "set_operating_mode", "error": "그리퍼 미초기화"}
            mode = req.get("mode", None)
            if mode is None:
                return {"ok": False, "cmd": "set_operating_mode", "error": "mode 누락"}
            try:
                g.set_operating_mode(int(mode))
                return {"ok": True, "cmd": "set_operating_mode"}
            except Exception as e:
                logger.error(f"[set_operating_mode] 오류: {e}")
                return {"ok": False, "cmd": "set_operating_mode", "error": str(e)}

        # ── get_state ─────────────────────────────────────────
        elif cmd == "get_state":
            if g is None:
                return {"ok": False, "cmd": "get_state", "error": "그리퍼 미초기화"}
            try:
                with self._lock:
                    state = g.get_state()
                if state is None:
                    return {"ok": False, "cmd": "get_state", "error": "인코더 읽기 실패"}
                return {
                    "ok":    True,
                    "cmd":   "get_state",
                    "state": state.tolist(),
                }
            except Exception as e:
                logger.error(f"[get_state] 오류: {e}")
                return {"ok": False, "cmd": "get_state", "error": str(e)}

        else:
            logger.warning(f"[unknown cmd] {cmd}")
            return {"ok": False, "cmd": cmd, "error": f"알 수 없는 명령: {cmd}"}

    # ── 메인 서버 루프 ─────────────────────────────────────
    def run(self):
        self._running = True
        logger.info(
            f"\n{'='*55}\n"
            f"  GripperServer 시작\n"
            f"  Listen: {self.host}:{self.port} (UDP)\n"
            f"{'='*55}"
        )
        logger.info("초기화 대기 중... (클라이언트가 'initialize' 명령을 보낼 때 자동 초기화)")

        while self._running:
            try:
                raw, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            cmd_str = raw.decode("utf-8", errors="ignore")

            # 응답 처리 (blocking — 빠른 처리, 별도 스레드 불필요)
            try:
                resp = self._handle_cmd(cmd_str, addr)
            except Exception as e:
                logger.error(f"명령 처리 오류: {e}")
                resp = {"ok": False, "error": str(e)}

            if resp is not None:
                self._reply(self._sock, addr, resp)

        logger.info("GripperServer 종료")

    def stop(self):
        self._running = False
        if self._gripper is not None:
            try:
                self._gripper.stop()
            except Exception:
                pass
        try:
            self._sock.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gripper UDP 서버 — Robot PC(UPC)에서 실행"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"바인드 호스트 (기본: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"UDP 수신 포트 (기본: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="디버그 로그 출력"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    server = GripperServer(host=args.host, port=args.port)

    def _sig_handler(sig, frame):
        logger.info("종료 시그널 수신 — 서버 정지 중...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    server.run()


if __name__ == "__main__":
    main()
