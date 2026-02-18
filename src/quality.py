"""
ECG Quality Analysis for Sentinel/Holter 2-lead recordings.
Adapted from Ecg-Interpretation-Python-Service quality.py
with HOLTER_THRESHOLDS for 200 Hz 2-lead signals.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

# Holter-specific thresholds for 200 Hz 2-lead recordings
HOLTER_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Muscle_Artifact": (0.045, 0.10),
    "Bad_Electrode_Contact": (10, 800),
    "Powerline_Interference": (0.01, 0.05),
    "Baseline_Drift": (0.03, 0.90),
    "Low_SNR": (22, 12),
}

FLAGS_WEIGHTS: Dict[str, float] = {
    "Muscle_Artifact": 0.2,
    "Bad_Electrode_Contact": 0.25,
    "Powerline_Interference": 0.15,
    "Baseline_Drift": 0.2,
    "Low_SNR": 0.2,
}

HOLTER_WEIGHTS: Dict[str, float] = {
    "Lead1": 0.6,
    "Lead2": 0.4,
}

FLAG_MESSAGES: Dict[str, str] = {
    "Muscle_Artifact": "Excess muscle noise",
    "Bad_Electrode_Contact": "Poor electrode contact",
    "Powerline_Interference": "Power-line interference detected",
    "Baseline_Drift": "Baseline drift present",
    "Low_SNR": "Low signal-to-noise ratio",
}


def analyze_lead_quality(
    signal: np.ndarray,
    sampling_rate: int = 200,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Compute signal-quality metrics for a single ECG segment.
    Adapted from Ecg-Interpretation-Python-Service quality.py.
    """
    t_grades = thresholds if thresholds is not None else HOLTER_THRESHOLDS

    sig = signal - np.mean(signal)
    freqs, psd = welch(sig, fs=sampling_rate)
    total_power = np.sum(psd) + 1e-12

    flags: Dict[str, float] = {}

    # Notch 50/60 Hz before analysis
    b50, a50 = iirnotch(50, Q=50, fs=sampling_rate)
    sos50 = tf2sos(b50, a50)
    b60, a60 = iirnotch(60, Q=50, fs=sampling_rate)
    sos60 = tf2sos(b60, a60)

    sig_notched = sosfiltfilt(sos50, sig)
    sig_notched = sosfiltfilt(sos60, sig_notched)

    nperseg = min(int(2.5 * sampling_rate), len(sig_notched))
    if nperseg < 4:
        nperseg = len(sig_notched)
    freqs2, psd2 = welch(sig_notched, fs=sampling_rate, nperseg=nperseg)
    total_power2 = np.sum(psd2) + 1e-12

    # Powerline interference (48-52 and 58-62 Hz)
    mains_bins = ((freqs2 >= 48) & (freqs2 <= 52)) | ((freqs2 >= 58) & (freqs2 <= 62))
    mains_power = np.sum(psd2[mains_bins])
    pi = mains_power / total_power2
    flags["Powerline_Interference"] = float(
        np.clip(
            (pi - t_grades["Powerline_Interference"][0])
            / (
                t_grades["Powerline_Interference"][1]
                - t_grades["Powerline_Interference"][0]
            ),
            0,
            1,
        )
    )

    # Muscle artifact (35-100 Hz)
    hf_bins = (freqs2 >= 35) & (freqs2 <= 100)
    hf_power = np.sum(psd2[hf_bins])
    ma_ratio = hf_power / total_power2
    flags["Muscle_Artifact"] = float(
        np.clip(
            (ma_ratio - t_grades["Muscle_Artifact"][0])
            / (t_grades["Muscle_Artifact"][1] - t_grades["Muscle_Artifact"][0]),
            0,
            1,
        )
    )

    # Baseline drift (<0.5 Hz)
    lf = np.sum(psd[freqs < 0.5])
    bd = lf / total_power
    flags["Baseline_Drift"] = float(
        np.clip(
            (bd - t_grades["Baseline_Drift"][0])
            / (t_grades["Baseline_Drift"][1] - t_grades["Baseline_Drift"][0]),
            0,
            1,
        )
    )

    # QRS amplitude
    amp = float(np.ptp(sig))

    # Bad electrode contact
    if amp < t_grades["Bad_Electrode_Contact"][0]:
        flags["Bad_Electrode_Contact"] = 1.0
    elif amp > t_grades["Bad_Electrode_Contact"][1]:
        flags["Bad_Electrode_Contact"] = 1.0
    else:
        flags["Bad_Electrode_Contact"] = 0.0

    # SNR
    sos_hp = butter(4, 1, btype="highpass", fs=sampling_rate, output="sos")
    clean_hp = sosfiltfilt(sos_hp, sig)
    amplitude = clean_hp.max() - clean_hp.min()

    if amplitude < 5:
        flags["Low_SNR"] = 1.0
        snr = 0.1
    else:
        try:
            nyq = sampling_rate / 2
            hi = min(40, nyq - 1)
            sos = butter(2, [0.5, hi], btype="bandpass", fs=sampling_rate, output="sos")
            clean = sosfiltfilt(sos, sig)
            noise = sig - clean
            noise_power = np.mean(noise**2) + 1e-12
            snr = 10 * np.log10((amp**2) / noise_power)
        except Exception:
            signal_power = np.mean(sig**2) + 1e-12
            noise_power = np.var(sig) + 1e-12
            snr = 10 * np.log10(signal_power / noise_power)

    flags["Low_SNR"] = float(
        np.clip(
            (snr - t_grades["Low_SNR"][0])
            / (t_grades["Low_SNR"][1] - t_grades["Low_SNR"][0]),
            0,
            1,
        )
    )

    return {
        "flags": flags,
        "values": {
            "m_a": ma_ratio,
            "b_e_c": amp,
            "p_i": pi,
            "b_d": bd,
            "snr": snr,
            "qrs_amp": amp,
        },
    }


def compute_quality_score(flags: Dict[str, float]) -> float:
    """Derive quality score (0.0-1.0) where each flag deducts weighted penalty."""
    score = 1.0
    for flag, value in flags.items():
        if value > 0.0:
            score -= FLAGS_WEIGHTS.get(flag, 0.2) * value
    return max(0.0, min(1.0, score))


def analyze_holter_quality(
    lead1: np.ndarray,
    lead2: np.ndarray,
    sampling_rate: int = 200,
    window_sec: float = 3,
) -> Dict[str, Any]:
    """
    Analyze quality of 2-lead Holter recording.
    Segments into windows, finds best consecutive pair (5 sec).
    """
    wlen = int(window_sec * sampling_rate)
    n = min(len(lead1), len(lead2))
    nwin = n // wlen if wlen > 0 else 0

    if nwin == 0:
        return {
            "lead1_quality": 0.0,
            "lead2_quality": 0.0,
            "overall_quality": 0.0,
            "grade": "Not usable",
            "lead1_flags": {},
            "lead2_flags": {},
            "window_scores": [],
        }

    window_results = []
    for i in range(nwin):
        seg1 = lead1[i * wlen : (i + 1) * wlen]
        seg2 = lead2[i * wlen : (i + 1) * wlen]

        m1 = analyze_lead_quality(seg1, sampling_rate)
        m2 = analyze_lead_quality(seg2, sampling_rate)

        q1 = compute_quality_score(m1["flags"])
        q2 = compute_quality_score(m2["flags"])

        overall = q1 * HOLTER_WEIGHTS["Lead1"] + q2 * HOLTER_WEIGHTS["Lead2"]

        window_results.append(
            {
                "window": i,
                "lead1_score": q1,
                "lead2_score": q2,
                "overall": overall,
                "lead1_flags": m1["flags"],
                "lead2_flags": m2["flags"],
                "lead1_values": m1["values"],
                "lead2_values": m2["values"],
            }
        )

    # Find best consecutive 2 windows (5 sec)
    best_start = 0
    best_quality = -1.0
    windows_needed = max(1, int(5.0 / window_sec))
    for start in range(max(1, nwin - windows_needed + 1)):
        seq = window_results[start : start + windows_needed]
        avg_q = np.mean([w["overall"] for w in seq])
        if avg_q > best_quality:
            best_quality = avg_q
            best_start = start

    best_windows = window_results[best_start : best_start + windows_needed]
    q1_avg = float(np.mean([w["lead1_score"] for w in best_windows]))
    q2_avg = float(np.mean([w["lead2_score"] for w in best_windows]))
    overall = q1_avg * HOLTER_WEIGHTS["Lead1"] + q2_avg * HOLTER_WEIGHTS["Lead2"]

    # Aggregate flags from best windows
    agg_flags1 = {}
    agg_flags2 = {}
    for flag in FLAG_MESSAGES:
        vals1 = [w["lead1_flags"].get(flag, 0) for w in best_windows]
        vals2 = [w["lead2_flags"].get(flag, 0) for w in best_windows]
        agg_flags1[flag] = float(np.mean(vals1))
        agg_flags2[flag] = float(np.mean(vals2))

    if overall > 0.85:
        grade = "Good"
    elif overall > 0.65:
        grade = "Questionable"
    else:
        grade = "Not usable"

    return {
        "lead1_quality": q1_avg,
        "lead2_quality": q2_avg,
        "overall_quality": overall,
        "grade": grade,
        "lead1_flags": agg_flags1,
        "lead2_flags": agg_flags2,
        "best_window_start": best_start,
        "best_window_end": best_start + windows_needed,
        "window_scores": [
            {
                "window": w["window"],
                "lead1": w["lead1_score"],
                "lead2": w["lead2_score"],
                "overall": w["overall"],
            }
            for w in window_results
        ],
    }
