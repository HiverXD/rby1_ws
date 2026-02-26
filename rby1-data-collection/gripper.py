import rby1_sdk as rby
import numpy as np
import time
import threading


class Gripper:
    """
    Class for gripper
    """

    GRIPPER_DIRECTION = False
    GOAL_CURRENT = 5  # Goal Current limit for CurrentBasedPositionControlMode

    def __init__(self):
        self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self.bus.open_port()
        self.bus.set_baud_rate(2_000_000)
        self.bus.set_torque_constant([1, 1])
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q: np.typing.NDArray = None
        self._lock = threading.Lock()  # thread-safe target_q access
        self._running = False
        self._thread = None

    def initialize(self, verbose=False):
        rv = True
        for dev_id in [0, 1]:
            if not self.bus.ping(dev_id):
                if verbose:
                    print(f"[Gripper] Dynamixel ID {dev_id} is not active")
                rv = False
            else:
                if verbose:
                    print(f"[Gripper] Dynamixel ID {dev_id} is active")
        if rv:
            print("[Gripper] Servo on gripper")
            self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
        return rv

    def set_operating_mode(self, mode):
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])

    def homing(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        direction = 0
        q = np.array([0, 0], dtype=np.float64)
        prev_q = np.array([0, 0], dtype=np.float64)
        counter = 0
        while direction < 2:
            self.bus.group_sync_write_send_torque(
                [(dev_id, 0.5 * (1 if direction == 0 else -1)) for dev_id in [0, 1]]
            )
            rv = self.bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            self.min_q = np.minimum(self.min_q, q)
            self.max_q = np.maximum(self.max_q, q)
            if np.array_equal(prev_q, q):
                counter += 1
            prev_q = q
            # A small value (e.g., 5) was too short and failed to detect limits properly, so a reasonably larger value was chosen.
            if counter >= 30:
                direction += 1
                counter = 0
            time.sleep(0.1)
        # Validate that homing found distinct limits
        range_q = self.max_q - self.min_q
        if np.any(np.abs(range_q) < 1e-6):
            print(
                f"[Gripper] WARNING: homing range is near-zero: min_q={self.min_q}, max_q={self.max_q}. "
                "Check encoder reads — gripper may not respond to commands."
            )
        else:
            print(f"[Gripper] Homing done: min_q={self.min_q}, max_q={self.max_q}, range={range_q}")
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
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        # Bug fix: send_torque must be called every iteration.
        # Calling it only once before the loop can lose the Goal Current value
        # if the SDK's send_position internally resets it.
        while self._running:
            with self._lock:  # thread-safe snapshot of target_q
                target = self.target_q.copy() if self.target_q is not None else None
            if target is not None:
                self.bus.group_sync_write_send_torque(
                    [(dev_id, self.GOAL_CURRENT) for dev_id in [0, 1]]
                )
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(target.tolist())]
                )
            time.sleep(0.1)

    def get_target(self):
        return self.target_q
    
    def get_normalized_target(self):
        return (self.target_q - self.min_q)/(self.max_q - self.min_q)

    def set_normalized_target(self, normalized_q):
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            print("[Gripper] Cannot set target. min_q or max_q is not valid.")
            return

        if Gripper.GRIPPER_DIRECTION:
            new_target = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            new_target = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q
        with self._lock:  # atomic write shared with loop() thread
            self.target_q = new_target


    def get_state(self):
        """
        Read current gripper encoder values.
        Returns:
            np.ndarray: current encoder positions for each servo [id0, id1]
        """
        rv = self.bus.group_fast_sync_read_encoder([0, 1])
        if rv is None:
            print("[Gripper] Failed to read encoder values.")
            return None
        q = np.zeros(2)
        for dev_id, enc in rv:
            q[dev_id] = enc

        # normalize
        q_norm = (q - self.min_q) / (self.max_q - self.min_q)

        # print(q_norm)
        return q_norm
