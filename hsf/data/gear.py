"""GEAR: Gated Event Accumulation Representation (HSF paper, Sec. III-B, Eq. 1-5).

For each RGB frame t, take the events between the previous and the current image (~50 ms):
    x_feat[x, y] = sum_i p_i * c          (Eq. 1, polarity in {-1, +1}, c = 8 gray levels)
    x_num [x, y] = number of events        (Eq. 2)
    f_t          = exp(-x_num / tau)       (Eq. 3, tau = 10: busy pixels forget their history)
    h_t          = f_t * h_{t-1} + x_feat  (Eq. 4, long-term memory, carried along the sequence)
    y_t          = tanh([h_t, x_feat, x_num])   (Eq. 5, 3 channels)

Our interpretation (the paper does not give the scaling inside tanh): all three quantities are
expressed in normalized gray units, i.e. each event contributes c / 255, then pass through tanh.
Storage as uint8 (decided after looking at previews):
  h, feat (signed):   (y + 1) / 2 * 255  -> no events = mid-gray 128 ("initialize to a mid-value")
  num (>= 0):         y * 255            -> no events = black, like the x_t(Number) map in Fig. 1
"""
import numba
import numpy as np


@numba.njit(cache=True)
def _accumulate(x, y, p, feat, num):
    for i in range(x.shape[0]):
        xi, yi = x[i], y[i]
        num[yi, xi] += 1
        feat[yi, xi] += 1 if p[i] else -1


class GEAR:
    """Stateful GEAR encoder for one event stream (call reset() at the start of each sequence)."""

    def __init__(self, height=480, width=640, c=8.0, tau=10.0):
        self.h, self.w = height, width
        self.scale = c / 255.0
        self.tau = tau
        self.reset()

    def reset(self):
        self.hidden = np.zeros((self.h, self.w), np.float32)

    def __call__(self, x, y, p):
        """x, y: pixel coords; p: polarity (0/1). Returns float32 (3, H, W) in [-1, 1]."""
        feat = np.zeros((self.h, self.w), np.int32)
        num = np.zeros((self.h, self.w), np.int32)
        if len(x):
            _accumulate(x, y, p, feat, num)
        return self.from_counts(feat, num)

    def from_counts(self, feat, num):
        """Same update from per-pixel counts: feat = sum of polarities (+1/-1), num = number of events.
        Used by the event simulator (hsf/data/event_sim.py), which produces counts directly."""
        x_feat = feat.astype(np.float32) * self.scale
        x_num = num.astype(np.float32)
        f = np.exp(-x_num / self.tau)
        self.hidden = f * self.hidden + x_feat
        return np.tanh(np.stack([self.hidden, x_feat, x_num * self.scale]))

    @staticmethod
    def to_uint8(y):
        """(3, H, W) tanh outputs -> (H, W, 3) uint8, channel order [h, feat, num].

        h and feat are signed (mid-gray = 0); num is non-negative (black = 0)."""
        out = np.empty((y.shape[1], y.shape[2], 3), np.uint8)
        out[..., 0] = np.round((y[0] + 1.0) * 127.5)
        out[..., 1] = np.round((y[1] + 1.0) * 127.5)
        out[..., 2] = np.round(y[2] * 255.0)
        return out

    @staticmethod
    def from_uint8(img):
        """Inverse of to_uint8 for the network input: (H, W, 3) uint8 -> float32 (3, H, W) in [-1, 1]."""
        x = img.astype(np.float32).transpose(2, 0, 1)
        x[:2] = x[:2] / 127.5 - 1.0
        x[2] = x[2] / 255.0
        return x


class EventReader:
    """Reads DSEC events.h5 by absolute microsecond time windows (exact to the microsecond)."""

    def __init__(self, h5):
        self.h5 = h5
        self.ev = h5['events']
        self.t = self.ev['t']
        self.t_offset = int(h5['t_offset'][()])
        self.ms_to_idx = np.asarray(h5['ms_to_idx'], dtype=np.int64)
        self.n = self.t.shape[0]

    def index_at(self, t_abs):
        """First event index with timestamp >= t_abs."""
        rel = t_abs - self.t_offset
        if rel <= 0:
            return 0
        ms = rel // 1000
        if ms >= len(self.ms_to_idx) - 1:
            return self.n
        lo, hi = int(self.ms_to_idx[ms]), int(self.ms_to_idx[ms + 1])
        return lo + int(np.searchsorted(self.t[lo:hi], rel, side='left'))

    def window(self, t0_abs, t1_abs):
        a, b = self.index_at(t0_abs), self.index_at(t1_abs)
        return self.ev['x'][a:b], self.ev['y'][a:b], self.ev['p'][a:b]


def first_window_start(ts, i, default_dt=50_000):
    """Start time of frame i's window: previous image time, or 50 ms before the first image."""
    return ts[i - 1] if i > 0 else ts[0] - default_dt


__all__ = ['GEAR', 'EventReader', 'first_window_start']
