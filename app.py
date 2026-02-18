"""
ECG Sentinel Quality - Streamlit GUI
Minimal testing environment for Sentinel .dat file analysis.

Processes .dat files using ported C# logic from Stream-Analysis-Service
and quality analysis from Ecg-Interpretation-Python-Service.
"""

import glob
import os
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from src.dat_parser import are_files_consecutive, parse_dat_file
from src.filters import preprocess_dat_signal
from src.qrs_detector import compute_heart_rate, detect_qrs
from src.quality import analyze_holter_quality, load_all_presets, save_preset


@st.cache_data
def _cached_load_presets():
    """Cached wrapper around load_all_presets to avoid repeated disk I/O."""
    return load_all_presets()


# Single source of truth for flag names and threshold UI config
_THRESH_CONFIG = {
    "Muscle_Artifact": {"min": 0.0, "max": 1.0, "step": 0.005, "format": "%.3f"},
    "Bad_Electrode_Contact": {"min": 0.0, "max": 2000.0, "step": 5.0, "format": "%.0f"},
    "Powerline_Interference": {"min": 0.0, "max": 1.0, "step": 0.005, "format": "%.3f"},
    "Baseline_Drift": {"min": 0.0, "max": 1.0, "step": 0.01, "format": "%.2f"},
    "Low_SNR": {"min": 0.0, "max": 50.0, "step": 1.0, "format": "%.0f"},
}
_FLAG_NAMES = list(_THRESH_CONFIG.keys())
_PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="ECG Sentinel Quality",
    page_icon="\u2764",
    layout="wide",
)

st.title("ECG Sentinel Quality Analyzer")
st.caption("Minimal testing environment for Sentinel Holter .dat files")

# -- Sidebar: file selection & parameters --------------------------------------

# -- Load presets for config tab -----------------------------------------------

_presets_data = _cached_load_presets()
_preset_names = list(_presets_data.get("presets", {}).keys())
_default_preset = _presets_data.get(
    "default", _preset_names[0] if _preset_names else "hardcoded_default"
)

# Initialize session state for config if not present
if "cfg_preset" not in st.session_state:
    st.session_state["cfg_preset"] = _default_preset


def _populate_config_state(preset_name: str) -> None:
    """Populate all cfg_* session state keys from a preset.
    Uses _FLAG_NAMES to ensure every known flag gets a value even if preset omits it."""
    from src.quality import _FALLBACK_CONFIG

    presets_data = _cached_load_presets()
    preset = presets_data.get("presets", {}).get(preset_name, {})
    if not preset:
        return
    fallback_thresholds = _FALLBACK_CONFIG["thresholds"]
    fallback_weights = _FALLBACK_CONFIG["flags_weights"]
    # Thresholds — fill all known flags with fallback for any missing
    thresholds = preset.get("thresholds", {})
    for flag_name in _FLAG_NAMES:
        bounds = thresholds.get(
            flag_name, fallback_thresholds.get(flag_name, (0.0, 1.0))
        )
        st.session_state[f"cfg_thresh_{flag_name}_low"] = float(bounds[0])
        st.session_state[f"cfg_thresh_{flag_name}_high"] = float(bounds[1])
    # Weights — fill all known flags
    weights = preset.get("flags_weights", {})
    for flag_name in _FLAG_NAMES:
        st.session_state[f"cfg_weight_{flag_name}"] = float(
            weights.get(flag_name, fallback_weights.get(flag_name, 0.2))
        )
    # Grade thresholds
    grades = preset.get("grade_thresholds", {})
    st.session_state["cfg_grade_good"] = float(grades.get("good", 0.85))
    st.session_state["cfg_grade_questionable"] = float(grades.get("questionable", 0.65))
    # NeuroKit
    nk = preset.get("neurokit", {})
    st.session_state["cfg_nk_enabled"] = bool(nk.get("enabled", False))
    st.session_state["cfg_nk_method"] = nk.get("method", "averageQRS")
    st.session_state["cfg_nk_weight"] = float(nk.get("weight", 0.0))


# Handle pending preset switch (set before rerun from save-as-new flow)
if "_pending_preset" in st.session_state:
    _switch_to = st.session_state.pop("_pending_preset")
    st.session_state["cfg_preset"] = _switch_to
    _populate_config_state(_switch_to)
# Populate defaults on first run
elif "cfg_thresh_Muscle_Artifact_low" not in st.session_state:
    _populate_config_state(st.session_state["cfg_preset"])


def _on_preset_change() -> None:
    """Callback when preset selectbox changes."""
    _populate_config_state(st.session_state["cfg_preset"])


# -- Sidebar: tabs for Input & Config -----------------------------------------

with st.sidebar:
    sidebar_input_tab, sidebar_config_tab = st.tabs(["Input", "Config"])

    # ---- Input Tab ----
    with sidebar_input_tab:
        st.header("Input")

        input_mode = st.radio(
            "Source", ["Select folder", "Upload files"], horizontal=True
        )

        dat_files_data = {}

        if input_mode == "Upload files":
            uploaded = st.file_uploader(
                "Upload .dat files",
                type=["dat"],
                accept_multiple_files=True,
            )
            if uploaded:
                for f in uploaded:
                    dat_files_data[f.name] = f.read()
        else:
            base_folder = st.text_input("Base folder", value="data")
            if base_folder and os.path.isdir(base_folder):
                subdirs = sorted(
                    [
                        d
                        for d in os.listdir(base_folder)
                        if os.path.isdir(os.path.join(base_folder, d))
                    ]
                )
                top_dats = sorted(glob.glob(os.path.join(base_folder, "*.dat")))

                options = []
                if top_dats:
                    options.append(f". ({len(top_dats)} files)")
                for sd in subdirs:
                    count = len(glob.glob(os.path.join(base_folder, sd, "*.dat")))
                    if count > 0:
                        options.append(f"{sd} ({count} files)")

                if options:
                    selected = st.selectbox("Subfolder", options)
                    folder_name = selected.split(" (")[0]
                    if folder_name == ".":
                        folder = base_folder
                    else:
                        folder = os.path.join(base_folder, folder_name)

                    found = sorted(glob.glob(os.path.join(folder, "*.dat")))
                    st.write(f"Found {len(found)} .dat files")
                    for fp in found:
                        with open(fp, "rb") as fh:
                            dat_files_data[os.path.basename(fp)] = fh.read()
                else:
                    st.warning("No .dat files found in folder or subfolders")
            elif base_folder:
                st.warning("Folder not found")

        st.divider()
        st.header("Parameters")

        notch_freq = st.multiselect(
            "Notch filter frequencies (Hz)",
            options=[50, 60, 100],
            default=[50, 100],
        )

        process_btn = st.button("Analyze", type="primary", width="stretch")

    # ---- Config Tab ----
    with sidebar_config_tab:
        st.header("Quality Config")

        # Reload presets to pick up any saves during this session
        _live_presets = _cached_load_presets()
        _live_names = list(_live_presets.get("presets", {}).keys())

        st.selectbox(
            "Preset",
            options=_live_names,
            key="cfg_preset",
            on_change=_on_preset_change,
        )

        # -- Thresholds --
        st.subheader("Thresholds")

        for flag_name, cfg in _THRESH_CONFIG.items():
            label = flag_name.replace("_", " ")
            if flag_name == "Low_SNR":
                label_low = f"{label} - Good SNR (high)"
                label_high = f"{label} - Bad SNR (low)"
            else:
                label_low = f"{label} - Low"
                label_high = f"{label} - High"
            st.number_input(
                label_low,
                min_value=cfg["min"],
                max_value=cfg["max"],
                step=cfg["step"],
                format=cfg["format"],
                key=f"cfg_thresh_{flag_name}_low",
            )
            st.number_input(
                label_high,
                min_value=cfg["min"],
                max_value=cfg["max"],
                step=cfg["step"],
                format=cfg["format"],
                key=f"cfg_thresh_{flag_name}_high",
            )

        # -- Flag Weights --
        st.subheader("Flag Weights")

        for flag_name in _FLAG_NAMES:
            st.slider(
                flag_name.replace("_", " "),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key=f"cfg_weight_{flag_name}",
            )
        weight_sum = sum(
            st.session_state.get(f"cfg_weight_{f}", 0.2) for f in _FLAG_NAMES
        )
        st.caption(f"Weight sum: {weight_sum:.2f}")
        if abs(weight_sum - 1.0) > 0.05:
            st.warning(
                "Weights should sum to ~1.0 for properly calibrated grade thresholds."
            )

        # -- Grade Thresholds --
        st.subheader("Grade Thresholds")

        st.slider(
            "Good (above)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_grade_good",
        )
        st.slider(
            "Questionable (above)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="cfg_grade_questionable",
        )

        # -- NeuroKit2 --
        st.subheader("NeuroKit2")

        st.toggle("Enable NeuroKit2", key="cfg_nk_enabled")
        nk_disabled = not st.session_state.get("cfg_nk_enabled", False)
        st.selectbox(
            "Method",
            options=["averageQRS", "zhao2018", "orphanidou2015"],
            key="cfg_nk_method",
            disabled=nk_disabled,
        )
        st.slider(
            "NK Weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="cfg_nk_weight",
            disabled=nk_disabled,
        )

        # -- Save Preset --
        st.divider()
        st.subheader("Save Preset")

        save_mode = st.radio(
            "Save mode",
            ["Overwrite current preset", "Save as new preset"],
            key="cfg_save_mode",
        )
        new_preset_name = ""
        if save_mode == "Save as new preset":
            new_preset_name = st.text_input(
                "New preset name", key="cfg_new_preset_name"
            )

        if st.button("Save Preset", width="stretch"):
            # Validate thresholds before save (F1, F7)
            _validation_errors = []
            for flag_name in _FLAG_NAMES:
                low_val = st.session_state.get(f"cfg_thresh_{flag_name}_low", 0.0)
                high_val = st.session_state.get(f"cfg_thresh_{flag_name}_high", 0.0)
                if flag_name == "Low_SNR":
                    if low_val <= high_val:
                        _validation_errors.append(
                            f"Low SNR: 'Good SNR' ({low_val}) must be > 'Bad SNR' ({high_val})"
                        )
                else:
                    if low_val > high_val:
                        _validation_errors.append(
                            f"{flag_name}: Low ({low_val}) must be <= High ({high_val})"
                        )
            # Validate grade thresholds (F3)
            _good = st.session_state.get("cfg_grade_good", 0.85)
            _quest = st.session_state.get("cfg_grade_questionable", 0.65)
            if _good <= _quest:
                _validation_errors.append(
                    f"Grade 'Good' ({_good}) must be > 'Questionable' ({_quest})"
                )

            if _validation_errors:
                for err in _validation_errors:
                    st.error(err)
            else:
                # Build preset dict from session state
                _save_preset = {
                    "thresholds": {},
                    "flags_weights": {},
                    "neurokit": {
                        "enabled": st.session_state.get("cfg_nk_enabled", False),
                        "method": st.session_state.get("cfg_nk_method", "averageQRS"),
                        "weight": st.session_state.get("cfg_nk_weight", 0.0),
                    },
                    "grade_thresholds": {
                        "good": _good,
                        "questionable": _quest,
                    },
                }
                for flag_name in _FLAG_NAMES:
                    _save_preset["thresholds"][flag_name] = [
                        st.session_state.get(f"cfg_thresh_{flag_name}_low", 0.0),
                        st.session_state.get(f"cfg_thresh_{flag_name}_high", 0.0),
                    ]
                    _save_preset["flags_weights"][flag_name] = st.session_state.get(
                        f"cfg_weight_{flag_name}", 0.2
                    )

                # Preserve _threshold_docs from current preset
                _presets_data_current = _cached_load_presets()
                target_name = st.session_state.get("cfg_preset", "holter_200hz")

                if save_mode == "Save as new preset":
                    if not new_preset_name or not new_preset_name.strip():
                        st.error("Please enter a preset name.")
                    elif not _PRESET_NAME_RE.match(new_preset_name.strip()):
                        st.error(
                            "Preset name must be 1-64 characters: letters, digits, hyphens, underscores only."
                        )
                    else:
                        target_name = new_preset_name.strip()
                        # Carry over _threshold_docs from current preset if available
                        current_docs = (
                            _presets_data_current.get("presets", {})
                            .get(st.session_state.get("cfg_preset", ""), {})
                            .get("_threshold_docs")
                        )
                        if current_docs:
                            _save_preset["_threshold_docs"] = current_docs
                        save_preset(target_name, _save_preset)
                        _cached_load_presets.clear()
                        st.session_state["_pending_preset"] = target_name
                        st.rerun()
                else:
                    # Preserve _threshold_docs from existing preset
                    existing = _presets_data_current.get("presets", {}).get(
                        target_name, {}
                    )
                    if "_threshold_docs" in existing:
                        _save_preset["_threshold_docs"] = existing["_threshold_docs"]
                    save_preset(target_name, _save_preset)
                    _cached_load_presets.clear()
                    st.success(f"Preset '{target_name}' saved.")

# -- Processing ----------------------------------------------------------------

if not dat_files_data:
    st.info("Upload or select .dat files to begin analysis.")
    st.stop()

if process_btn:
    # Validate config before analysis (F1, F3, F7)
    _analysis_errors = []
    for flag in _FLAG_NAMES:
        low_v = st.session_state.get(f"cfg_thresh_{flag}_low", 0.0)
        high_v = st.session_state.get(f"cfg_thresh_{flag}_high", 0.0)
        if flag == "Low_SNR":
            if low_v <= high_v:
                _analysis_errors.append(
                    f"Low SNR: 'Good SNR' ({low_v}) must be > 'Bad SNR' ({high_v})"
                )
        else:
            if low_v > high_v:
                _analysis_errors.append(
                    f"{flag}: Low ({low_v}) must be <= High ({high_v})"
                )
    _g = st.session_state.get("cfg_grade_good", 0.85)
    _q = st.session_state.get("cfg_grade_questionable", 0.65)
    if _g <= _q:
        _analysis_errors.append(f"Grade 'Good' ({_g}) must be > 'Questionable' ({_q})")
    if _analysis_errors:
        for ae in _analysis_errors:
            st.error(ae)
        st.stop()

    # Build quality config dict from session state sliders
    quality_config = {
        "_preset_name": st.session_state.get("cfg_preset", "custom"),
        "thresholds": {
            flag: (
                st.session_state.get(f"cfg_thresh_{flag}_low", 0.0),
                st.session_state.get(f"cfg_thresh_{flag}_high", 0.0),
            )
            for flag in _FLAG_NAMES
        },
        "flags_weights": {
            flag: st.session_state.get(f"cfg_weight_{flag}", 0.2)
            for flag in _FLAG_NAMES
        },
        "neurokit": {
            "enabled": st.session_state.get("cfg_nk_enabled", False),
            "method": st.session_state.get("cfg_nk_method", "averageQRS"),
            "weight": st.session_state.get("cfg_nk_weight", 0.0),
        },
        "grade_thresholds": {
            "good": st.session_state.get("cfg_grade_good", 0.85),
            "questionable": st.session_state.get("cfg_grade_questionable", 0.65),
        },
    }

    # Sort files by name for consistent ordering
    sorted_files = sorted(dat_files_data.keys())
    st.write(f"Processing **{len(sorted_files)}** files: {', '.join(sorted_files)}")

    # Check consecutiveness
    if len(sorted_files) >= 2:
        file_bytes = [dat_files_data[fn] for fn in sorted_files]
        consecutive = are_files_consecutive(file_bytes)
        if consecutive:
            st.success("Files are consecutive (packet numbers follow sequentially)")
        else:
            st.warning("Files are NOT consecutive - gaps detected between packets")

    # Process each file
    all_results = {}
    progress = st.progress(0, text="Parsing files...")

    for idx, fname in enumerate(sorted_files):
        progress.progress((idx + 1) / len(sorted_files), text=f"Processing {fname}...")
        try:
            raw_data = dat_files_data[fname]
            lead1_raw, lead2_raw = parse_dat_file(raw_data)

            # Preprocess (HP filter + resample 125->200 Hz + notch)
            lead1 = preprocess_dat_signal(lead1_raw, notch_freqs=notch_freq)
            lead2 = preprocess_dat_signal(lead2_raw, notch_freqs=notch_freq)

            # QRS detection on both leads
            markers1 = detect_qrs(lead1, fs=200)
            markers2 = detect_qrs(lead2, fs=200)

            # Heart rate stats
            hr1 = compute_heart_rate(markers1, fs=200)
            hr2 = compute_heart_rate(markers2, fs=200)

            # Pick best lead (lower HR std = more reliable, matching C# decide())
            if hr1["hr_std"] <= hr2["hr_std"] and hr1["hr_mean"] > 0:
                best_lead_idx = 1
            else:
                best_lead_idx = 2

            # Quality analysis with config from sidebar
            quality = analyze_holter_quality(
                lead1, lead2, sampling_rate=200, config=quality_config
            )

            all_results[fname] = {
                "lead1_raw": lead1_raw,
                "lead2_raw": lead2_raw,
                "lead1": lead1,
                "lead2": lead2,
                "markers1": markers1,
                "markers2": markers2,
                "hr1": hr1,
                "hr2": hr2,
                "best_lead": best_lead_idx,
                "quality": quality,
            }
        except Exception as e:
            st.error(f"Error processing {fname}: {e}")

    progress.empty()

    if all_results:
        st.session_state["results"] = all_results
        st.session_state["results_config"] = quality_config
    else:
        st.error("No files processed successfully.")
        st.stop()

if "results" not in st.session_state:
    st.info(f"{len(dat_files_data)} files loaded. Click **Analyze** to process.")
    st.stop()

all_results = st.session_state["results"]

# -- Results display -----------------------------------------------------------

tabs = st.tabs(list(all_results.keys()))

for tab, (fname, result) in zip(tabs, all_results.items()):
    with tab:
        q = result["quality"]
        hr1 = result["hr1"]
        hr2 = result["hr2"]

        # -- Summary metrics --
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Quality Grade", q["grade"])
        with col2:
            st.metric("Overall Quality", f"{q['overall_quality']:.2f}")
        with col3:
            best = result["best_lead"]
            best_hr = hr1 if best == 1 else hr2
            st.metric(
                "HR Mean",
                f"{best_hr['hr_mean']:.0f} BPM" if best_hr["hr_mean"] > 0 else "N/A",
            )
        with col4:
            st.metric("Best Lead", f"Lead {best}")

        # -- Quality details --
        with st.expander("Quality Details", expanded=True):
            _rc = st.session_state.get("results_config", {})
            _nk_cfg = _rc.get("neurokit", {})
            nk_enabled = _nk_cfg.get("enabled", False)
            nk_weight = _nk_cfg.get("weight", 0.0)

            qcol1, qcol2 = st.columns(2)
            for col, lead_num in [(qcol1, 1), (qcol2, 2)]:
                with col:
                    blended = q[f"lead{lead_num}_quality"]
                    psd_q = q.get(f"lead{lead_num}_psd_quality", blended)
                    nk_q = q.get(f"lead{lead_num}_nk_quality")
                    vals = q.get(f"lead{lead_num}_values", {})
                    flags = q.get(f"lead{lead_num}_flags", {})
                    is_best = q.get("quality_best_lead") == lead_num

                    header = f"Lead {lead_num}"
                    if is_best:
                        header += " (Best)"
                    st.markdown(f"#### {header}")

                    # Scores
                    st.markdown(f"**Overall Quality: {blended:.3f}**")
                    score_parts = [f"PSD: {psd_q:.3f}"]
                    if nk_enabled and nk_q is not None:
                        score_parts.append(f"NK: {nk_q:.3f}")
                        score_parts.append(f"Weight: {1 - nk_weight:.0%}/{nk_weight:.0%}")
                    elif nk_enabled:
                        score_parts.append("NK: N/A")
                    st.caption(" | ".join(score_parts))

                    # Flags
                    st.markdown("**Flags**")
                    for flag, val in flags.items():
                        label = flag.replace("_", " ")
                        if val > 0.5:
                            st.markdown(f"- :red[{label}: {val:.3f}]")
                        elif val > 0.1:
                            st.markdown(f"- :orange[{label}: {val:.3f}]")
                        elif val > 0.0:
                            st.markdown(f"- {label}: {val:.3f}")
                        else:
                            st.markdown(f"- :green[{label}: {val:.3f}]")

                    # Raw measurement values
                    if vals:
                        st.markdown("**Measurements**")
                        _meas_labels = {
                            "snr": ("SNR", "dB", "%.1f"),
                            "qrs_amp": ("QRS Amplitude", "", "%.1f"),
                            "m_a": ("Muscle Artifact Ratio", "", "%.4f"),
                            "p_i": ("Powerline Interference", "", "%.4f"),
                            "b_d": ("Baseline Drift Ratio", "", "%.4f"),
                        }
                        for key, (label, unit, fmt) in _meas_labels.items():
                            v = vals.get(key)
                            if v is not None:
                                suffix = f" {unit}" if unit else ""
                                st.caption(f"{label}: {fmt % v}{suffix}")
                        if nk_enabled:
                            nk_peaks = vals.get("nk_r_peaks_count", 0)
                            st.caption(f"NK R-peaks detected: {nk_peaks}")

            # Preset & window info
            st.caption(
                f"Preset: {q.get('preset', 'N/A')} | "
                f"Best windows: {q.get('best_window_start', 0)}-{q.get('best_window_end', 0)} | "
                f"Quality best lead: {q.get('quality_best_lead', 'N/A')}"
            )

            # Window scores chart
            if q["window_scores"]:
                win_fig = go.Figure()
                windows = [w["window"] for w in q["window_scores"]]
                win_fig.add_trace(
                    go.Bar(
                        x=windows,
                        y=[w["lead1"] for w in q["window_scores"]],
                        name="Lead 1",
                        marker_color="#1f77b4",
                    )
                )
                win_fig.add_trace(
                    go.Bar(
                        x=windows,
                        y=[w["lead2"] for w in q["window_scores"]],
                        name="Lead 2",
                        marker_color="#ff7f0e",
                    )
                )
                win_fig.add_trace(
                    go.Scatter(
                        x=windows,
                        y=[w["overall"] for w in q["window_scores"]],
                        name="Overall",
                        mode="lines+markers",
                        line=dict(color="white", width=2),
                    )
                )
                # Highlight best windows
                best_s = q.get("best_window_start", 0)
                best_e = q.get("best_window_end", 0)
                win_fig.add_vrect(
                    x0=best_s - 0.5,
                    x1=best_e - 0.5,
                    fillcolor="green",
                    opacity=0.15,
                    annotation_text="Best",
                )
                win_fig.update_layout(
                    title="Quality per Window (5s each)",
                    xaxis_title="Window",
                    yaxis_title="Quality Score",
                    yaxis_range=[0, 1.05],
                    barmode="group",
                    height=300,
                    template="plotly_dark",
                )
                st.plotly_chart(win_fig, width="stretch")

        # -- ECG Signal plots --
        with st.expander("ECG Signals", expanded=True):
            lead1 = result["lead1"]
            lead2 = result["lead2"]
            time_axis = np.arange(len(lead1)) / 200.0
            time_axis2 = np.arange(len(lead2)) / 200.0

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    "Lead 1 (filtered, 200 Hz)",
                    "Lead 2 (filtered, 200 Hz)",
                ),
                vertical_spacing=0.08,
            )

            # Lead 1
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=lead1,
                    mode="lines",
                    line=dict(color="#00ffff", width=1),
                    name="Lead 1",
                    hovertemplate="t=%{x:.3f}s<br>val=%{y:.0f}",
                ),
                row=1,
                col=1,
            )

            # Lead 2
            fig.add_trace(
                go.Scatter(
                    x=time_axis2,
                    y=lead2,
                    mode="lines",
                    line=dict(color="#00ff00", width=1),
                    name="Lead 2",
                    hovertemplate="t=%{x:.3f}s<br>val=%{y:.0f}",
                ),
                row=2,
                col=1,
            )

            # QRS markers for lead 1
            m1 = result["markers1"]
            if len(m1) > 0:
                m1_pos = m1[:, 0]
                valid = m1_pos < len(lead1)
                m1_times = m1_pos[valid] / 200.0
                m1_vals = lead1[m1_pos[valid]]
                fig.add_trace(
                    go.Scatter(
                        x=m1_times,
                        y=m1_vals,
                        mode="markers",
                        marker=dict(color="yellow", size=6, symbol="triangle-down"),
                        name="QRS L1",
                        hovertemplate="Beat @%{x:.2f}s",
                    ),
                    row=1,
                    col=1,
                )

            # QRS markers for lead 2
            m2 = result["markers2"]
            if len(m2) > 0:
                m2_pos = m2[:, 0]
                valid = m2_pos < len(lead2)
                m2_times = m2_pos[valid] / 200.0
                m2_vals = lead2[m2_pos[valid]]
                fig.add_trace(
                    go.Scatter(
                        x=m2_times,
                        y=m2_vals,
                        mode="markers",
                        marker=dict(color="yellow", size=6, symbol="triangle-down"),
                        name="QRS L2",
                        hovertemplate="Beat @%{x:.2f}s",
                    ),
                    row=2,
                    col=1,
                )

            fig.update_layout(
                height=500,
                template="plotly_dark",
                xaxis2_title="Time (seconds)",
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
            )
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)

            st.plotly_chart(fig, width="stretch")

        # -- Raw vs Filtered comparison --
        raw_open_key = f"raw_open_{fname}"
        with st.expander(
            "Raw vs Filtered Comparison",
            expanded=st.session_state.get(raw_open_key, False),
        ):
            lead_choice = st.radio(
                "Lead",
                [1, 2],
                horizontal=True,
                key=f"raw_{fname}",
                on_change=lambda k=raw_open_key: st.session_state.__setitem__(k, True),
            )
            raw = result["lead1_raw"] if lead_choice == 1 else result["lead2_raw"]
            filt = result["lead1"] if lead_choice == 1 else result["lead2"]

            raw_time = np.arange(len(raw)) / 125.0
            filt_time = np.arange(len(filt)) / 200.0

            comp_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Raw (125 Hz)", "Filtered (200 Hz)"),
            )
            comp_fig.add_trace(
                go.Scatter(
                    x=raw_time,
                    y=raw,
                    mode="lines",
                    line=dict(color="gray", width=1),
                    name="Raw",
                ),
                row=1,
                col=1,
            )
            comp_fig.add_trace(
                go.Scatter(
                    x=filt_time,
                    y=filt,
                    mode="lines",
                    line=dict(color="#00ffff", width=1),
                    name="Filtered",
                ),
                row=2,
                col=1,
            )
            comp_fig.update_layout(
                height=400, template="plotly_dark", xaxis2_title="Time (seconds)"
            )
            st.plotly_chart(comp_fig, width="stretch")

        # -- Heart Rate --
        with st.expander("Heart Rate Analysis"):
            hrcol1, hrcol2 = st.columns(2)
            with hrcol1:
                st.write("**Lead 1**")
                h1 = result["hr1"]
                if h1["hr_mean"] > 0:
                    st.write(f"Mean HR: {h1['hr_mean']:.1f} BPM")
                    st.write(f"HR Std: {h1['hr_std']:.1f}")
                    st.write(f"HR Range: {h1['hr_min']:.0f} - {h1['hr_max']:.0f} BPM")
                    st.write(f"Beats detected: {len(result['markers1'])}")
                else:
                    st.write("No beats detected")
            with hrcol2:
                st.write("**Lead 2**")
                h2 = result["hr2"]
                if h2["hr_mean"] > 0:
                    st.write(f"Mean HR: {h2['hr_mean']:.1f} BPM")
                    st.write(f"HR Std: {h2['hr_std']:.1f}")
                    st.write(f"HR Range: {h2['hr_min']:.0f} - {h2['hr_max']:.0f} BPM")
                    st.write(f"Beats detected: {len(result['markers2'])}")
                else:
                    st.write("No beats detected")

            # RR interval plot
            best_hr = result["hr1"] if result["best_lead"] == 1 else result["hr2"]
            rr = best_hr["rr_intervals"]
            if rr:
                rr_fig = go.Figure()
                rr_fig.add_trace(
                    go.Scatter(
                        y=np.array(rr) / 200.0 * 1000,
                        mode="lines+markers",
                        line=dict(color="#00ffff"),
                        name="RR interval",
                    )
                )
                rr_fig.update_layout(
                    title="RR Intervals (Best Lead)",
                    xaxis_title="Beat #",
                    yaxis_title="RR (ms)",
                    height=250,
                    template="plotly_dark",
                )
                st.plotly_chart(rr_fig, width="stretch")
