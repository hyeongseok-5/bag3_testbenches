from typing import Mapping, Any
import numpy as np
import matplotlib.pyplot as plt

from bag.math.interpolate import LinearInterpolator


class EyeAnalysis:
    def __init__(self, t_per: float, t_delay: float, strobe: int = 20):
        self._t_per: float = t_per
        self._t_delay: float = t_delay
        self._strobe: int = strobe
        self._eye_per: float = 2 * t_per
        self._time_eye = np.linspace(0, self._eye_per, self._strobe, endpoint=False)

    def _plot_eye(self, eye_vals: np.ndarray, ans: Mapping[str, Any]) -> None:
        eye_w = ans['width']
        eye_h = ans['height']
        plt.figure()
        plt.plot(self._time_eye, eye_vals.T, 'b')
        plt.hlines(ans['val_mid'], self._time_eye[ans['pos_w0']], self._time_eye[ans['pos_w1']],
                   label=f'width={eye_w} s', linestyle="--", colors=['r'])
        plt.vlines(self._time_eye[ans['pos_h']], ans['val_mid'] - eye_h / 2, ans['val_mid'] + eye_h / 2,
                   label=f'height={eye_h} V', linestyle=":", colors=['g'])
        plt.xlabel('time (s)')
        plt.ylabel('amplitude (V)')
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()

    def _compute_width_height(self, eye_vals: np.ndarray, val_mid: float) -> Mapping[str, Any]:
        eye_0 = eye_vals - val_mid
        # replace all negative values with max value
        eye_pos = np.where(eye_0 > 0, eye_0, np.max(eye_0))
        # replace all positive values with min value
        eye_neg = np.where(eye_0 < 0, eye_0, np.min(eye_0))

        eye_heights = np.min(eye_pos, 0) - np.max(eye_neg, 0)

        # find min eye height position in first half of the array
        pos0 = np.argmin(eye_heights[self._strobe // 2::-1])    # last occurrence
        pos0 = self._strobe // 2 - pos0
        pos1 = np.argmin(eye_heights[self._strobe // 2:])
        pos1 += self._strobe // 2

        eye_h = np.max(eye_heights[pos0:pos1])
        pos_h = np.argmax(eye_heights[pos0:pos1])
        pos_h += pos0
        eye_w = self._time_eye[pos1] - self._time_eye[pos0]
        return dict(
            width=eye_w,
            height=eye_h,
            pos_w0=pos0,
            pos_w1=pos1,
            pos_h=pos_h,
            val_mid=val_mid,
        )

    def analyze_eye(self, time: np.ndarray, sig: np.ndarray, plot_eye: bool = False) -> Mapping[str, Any]:
        # linearly interpolate the signal
        sig_li = LinearInterpolator([time], sig, [self._t_per / 10])

        # starting from t_delay, find the first transition
        val_mid = (min(sig) + max(sig)) / 2
        _t0 = self._t_delay
        while True:
            _t1 = _t0 + self._t_per
            if (sig_li(np.array([_t0])) - val_mid) * (sig_li(np.array([_t1])) - val_mid) < 0:
                # this is the first transition
                break
            _t0 = _t1

        # Find exact zero crossing between _t0 and _t1.
        # This is the extra delay of the output beyond t_delay of the input.
        t_delay1 = bin_search(sig_li - val_mid, _t0, _t0 + self._t_per, self._t_per / self._strobe) - _t0
        # To center the eye, the first transistion should be quarter period shifted.
        t_delay2 = t_delay1 - self._eye_per / 4

        # fold the signal to form centered eye
        t_sim = time[-1]
        num_traces = int((t_sim - self._t_delay - t_delay2) / self._eye_per)
        t_fin = self._t_delay + t_delay2 + self._eye_per * num_traces
        time_tot = np.linspace(self._t_delay + t_delay2, t_fin, self._strobe * num_traces, endpoint=False)
        sig_vals = sig_li(time_tot)
        eye_vals = np.empty((num_traces, self._strobe), dtype=float)
        for tridx in range(num_traces):
            eye_vals[tridx] = sig_vals[tridx * self._strobe:(tridx + 1) * self._strobe]

        # compute eye width and height
        ans = self._compute_width_height(eye_vals, val_mid)

        if plot_eye:
            self._plot_eye(eye_vals, ans)

        return dict(
            width=ans['width'],
            height=ans['height'],
        )


def bin_search(data: LinearInterpolator, t_i: float, t_f: float, tol: float) -> float:
    t_mid = (t_i + t_f) / 2
    if t_f - t_i <= tol:
        return t_mid
    if data(np.array([t_i])) * data(np.array([t_mid])) < 0:
        return bin_search(data, t_i, t_mid, tol)
    return bin_search(data, t_mid, t_f, tol)
