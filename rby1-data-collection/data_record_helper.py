from dataclasses import dataclass
from utils import * 
import logging 
from h5py_writer import H5Writer
import threading 
import time

@dataclass
class RecordingSession:
    writer: H5Writer
    stop_event: threading.Event
    path: str
    started_at: float

def start_recording(gripper, fps=10, start_demo_logger=None) -> RecordingSession:
    """Start a new numbered .h5 and a logger thread."""
    assert start_demo_logger is not None, "set the demo logger"

    path = get_next_h5_path()  # from utils.py (you already import *)
    writer = H5Writer(path=path, flush_every=60, flush_secs=1.0).start()
    stop_event = start_demo_logger(gripper, writer, fps=fps)
    logging.info(f"[record] START  -> {path}")
    return RecordingSession(writer=writer, stop_event=stop_event, path=path, started_at=time.time())

def stop_recording(sess: RecordingSession | None):
    """Stop logger thread and close/flush the H5 file."""
    if not sess:
        return
    try:
        sess.stop_event.set()
    except Exception:
        pass

    # Be conservative: call whatever the writer supports
    try:
        if hasattr(sess.writer, "stop"):
            sess.writer.stop()
        elif hasattr(sess.writer, "close"):
            sess.writer.close()
        elif hasattr(sess.writer, "flush"):
            sess.writer.flush()
    except Exception as e:
        logging.warning(f"[record] writer close error: {e}")

    dur = time.time() - sess.started_at
    logging.info(f"[record] STOP   -> {sess.path}  ({dur:.1f}s)")
