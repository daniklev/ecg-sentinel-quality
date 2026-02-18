"""
ECG signal filters - Python port of C# filter pipeline from
CommwellSentinelAnalysis + reference from Ecg-Interpretation-Python-Service.

Pipeline: HP filter (baseline) -> Sample rate conversion (125->200 Hz) -> Notch
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos


# ---------- IIR filter (port of C# 'filter' class) ----------

class IIRFilter:
    """
    Generic IIR filter matching CommwellSentinelAnalysis.filter class.
    Supports both FIR (b only) and IIR (b + a coefficients).
    """

    def __init__(self, b: np.ndarray, a: np.ndarray = None, size: int = None):
        self.b = np.array(b, dtype=np.float64)
        if a is not None:
            self.a = np.array(a, dtype=np.float64)
        else:
            self.a = None
        sz = size or len(self.b)
        self.x_hist = np.zeros(sz + 1, dtype=np.float64)
        self.y_hist = np.zeros(sz + 1, dtype=np.float64)
        self.size = sz

    def filter_value(self, new_data: float) -> float:
        # Shift history
        self.x_hist = np.roll(self.x_hist, 1)
        self.x_hist[0] = new_data

        y = 0.0
        for j in range(min(self.size, len(self.b))):
            y += self.b[j] * self.x_hist[j]

        if self.a is not None:
            for j in range(1, min(self.size, len(self.a))):
                y -= self.a[j] * self.y_hist[j - 1]

        self.y_hist = np.roll(self.y_hist, 1)
        self.y_hist[0] = y
        return y


# ---------- C# filter coefficients from CommwellSentinelAnalysis ----------

# HP 0.5 Hz baseline removal filter (bHP2/aHP2 in C#)
_bHP2 = [0.967694808889672, -3.870779235558687, 5.806168853338031,
         -3.870779235558687, 0.967694808889672]
_aHP2 = [1.0, -3.934325820798737, 5.805125421055140,
         -3.807232457228852, 0.936433243152019]


def hp_baseline_filter(signal: np.ndarray) -> np.ndarray:
    """
    Apply the 0.5 Hz high-pass filter for baseline removal.
    Matches C#: HP1_05hz / HP2_05hz using bHP2/aHP2 coefficients.
    First 200 samples used as warmup to stabilize IIR state.
    """
    filt = IIRFilter(_bHP2, _aHP2, 5)
    warmup = min(200, len(signal))
    for i in range(warmup):
        filt.filter_value(signal[i])
    out = np.empty_like(signal)
    for i in range(len(signal)):
        out[i] = filt.filter_value(signal[i])
    return out


# ---------- Sample rate conversion (125 Hz -> 200 Hz) ----------

def resample_125_to_200(signal: np.ndarray) -> np.ndarray:
    """
    Port of C# SampleRateConverter (5:8 ratio).
    Takes 5 input samples, produces 8 output samples.
    Uses max-deviation interpolation matching the C# algorithm.
    """
    in_size = 5
    out_size = 8
    n = len(signal)
    out_len = 1 + n * out_size // in_size
    output = np.zeros(out_len, dtype=np.float64)
    out_idx = 0
    last_avg = 0.0

    for i in range(0, n - in_size + 1, in_size):
        # Expand: each input sample repeated out_size times
        expanded = np.repeat(signal[i:i + in_size], out_size)

        for j in range(out_size):
            # For each output sample, look at in_size values
            chunk = expanded[j * in_size:(j + 1) * in_size]
            max_val = last_avg
            max_delta = 0.0
            s = 0.0
            for val in chunk:
                delta = abs(last_avg - val)
                if delta > max_delta:
                    max_delta = delta
                    max_val = val
                s += val
            last_avg = s / out_size
            if out_idx < out_len:
                output[out_idx] = max_val
                out_idx += 1

    return output[:out_idx]


# ---------- Notch filter ----------

def apply_notch(signal: np.ndarray, freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """Apply a zero-phase notch filter at the given frequency."""
    b, a = iirnotch(freq, Q, fs)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, signal)


# ---------- Complete preprocessing pipeline ----------

def preprocess_dat_signal(
    lead: np.ndarray,
    native_fs: int = 125,
    target_fs: int = 200,
    notch_freqs: list = None,
) -> np.ndarray:
    """
    Full preprocessing pipeline matching C# CommwellSentinelAnalysis.ECG1():
    1. HP 0.5 Hz baseline removal
    2. Sample rate conversion 125 -> 200 Hz
    3. Optional notch filtering (50/60 Hz)

    Args:
        lead: raw ECG samples at native_fs
        native_fs: input sample rate (125 Hz from device)
        target_fs: output sample rate (200 Hz)
        notch_freqs: powerline frequencies to notch (e.g. [50] or [50, 100])

    Returns:
        preprocessed signal at target_fs
    """
    # 1. HP baseline removal
    filtered = hp_baseline_filter(lead)

    # 2. Sample rate conversion
    resampled = resample_125_to_200(filtered)

    # 3. Optional notch filtering
    if notch_freqs:
        for freq in notch_freqs:
            if freq < target_fs / 2:  # Only if below Nyquist
                resampled = apply_notch(resampled, freq, target_fs)

    return resampled
