"""Approximate events from ordinary video frames (no event camera available), for the screen demo.

ESIM / v2e-style idealised sensor: every pixel keeps a reference log intensity; when the current log
intensity differs from it by at least the contrast threshold C, floor(|dL| / C) events of that sign are
emitted and the reference moves by that many steps. Only per-pixel counts are produced (sum of
polarities and number of events), which is all GEAR needs (GEAR.from_counts), so the result goes
through exactly the same GEAR encoding as the real DSEC events.

What this does NOT model: sub-frame timing (all events of one frame interval are simultaneous),
sensor noise and refractory period, motion between frames (a fast edge may jump several pixels),
and the different optics / resolution of a real event camera. Calibrate C with
tools/event_sim_check.py so the event density is close to DSEC's.
"""
import cv2
import numpy as np

from hsf.data.gear import GEAR


class EventSimulator:
    def __init__(self, contrast=0.2, height=480, width=640, eps=1e-3):
        self.c, self.eps = contrast, eps
        self.gear = GEAR(height, width)
        self.ref = None

    def reset(self):
        self.ref = None
        self.gear.reset()

    def counts(self, gray):
        """gray: uint8 (H, W). Returns (feat, num) int32 counts since the previous frame."""
        log_i = np.log(gray.astype(np.float32) / 255.0 + self.eps)
        if self.ref is None:
            self.ref = log_i
            z = np.zeros(gray.shape, np.int32)
            return z, z
        d = log_i - self.ref
        n = np.floor(np.abs(d) / self.c).astype(np.int32)
        sign = np.sign(d).astype(np.int32)
        self.ref = self.ref + sign * n * self.c
        return sign * n, n

    def __call__(self, bgr):
        """bgr: uint8 (480, 640, 3) frame -> GEAR image uint8 (480, 640, 3) in the stored [h, feat, num] order
        (what cv2.imread returns for the preprocessed DSEC gear/*.png)."""
        feat, num = self.counts(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        return GEAR.to_uint8(self.gear.from_counts(feat, num))
