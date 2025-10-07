#recorder.py
import os
import cv2
import logging
from datetime import datetime

class Recorder:
    def __init__(self, output_dir='recordings', fps=20.0,
                 duration=20, snapshot=False, snapshot_interval=5, no_record=False):
        """
        Manages video writing and optional snapshots.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.fps = fps
        self.duration = duration
        self.snapshot = snapshot
        self.snapshot_interval = snapshot_interval
        self.no_record = no_record

        self.writer = None
        self.start_time = None
        self.last_snap = None

    def start(self, frame):
        """
        Begin recording to a timestamped .avi.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"record_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        self.start_time = datetime.now()
        self.last_snap = datetime.now()
        logging.info(f"Recording started: {path}")

    def update(self, frame):
        """
        Write frame, take snapshots if enabled, and stop when duration elapses.
        Returns True if still recording.
        """
        if self.writer is None or self.no_record:
            return False

        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > self.duration:
            self.stop()
            return False

        self.writer.write(frame)

        if self.snapshot:
            since = (datetime.now() - self.last_snap).total_seconds()
            if since > self.snapshot_interval:
                self._snapshot(frame)

        return True

    def _snapshot(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"snap_{ts}.jpg")
        cv2.imwrite(path, frame)
        self.last_snap = datetime.now()
        logging.info(f"Snapshot saved: {path}")

    def stop(self):
        """
        Finish recording and release resources.
        """
        if self.writer:
            self.writer.release()
            logging.info("Recording stopped.")
            self.writer = None