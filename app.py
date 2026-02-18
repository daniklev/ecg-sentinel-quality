"""
ECG Sentinel Quality - Streamlit GUI
Minimal testing environment for Sentinel .dat file analysis.

Processes .dat files using ported C# logic from Stream-Analysis-Service
and quality analysis from Ecg-Interpretation-Python-Service.
"""

import glob
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from src.dat_parser import are_files_consecutive, parse_dat_file
from src.filters import preprocess_dat_signal
from src.qrs_detector import compute_heart_rate, detect_qrs
from src.quality import analyze_holter_quality

# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="ECG Sentinel Quality",
    page_icon="\u2764",
    layout="wide",
)

st.title("ECG Sentinel Quality Analyzer")
st.caption("Minimal testing environment for Sentinel Holter .dat files")

# -- Sidebar: file selection & parameters --------------------------------------

with st.sidebar:
    st.header("Input")

    input_mode = st.radio("Source", ["Upload files", "Select folder"], horizontal=True)

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
            # List subdirectories for selection
            subdirs = sorted(
                [
                    d
                    for d in os.listdir(base_folder)
                    if os.path.isdir(os.path.join(base_folder, d))
                ]
            )
            # Also check for .dat files directly in base folder
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
        default=[50],
    )

    process_btn = st.button("Analyze", type="primary", use_container_width=True)

# -- Processing ----------------------------------------------------------------

if not dat_files_data:
    st.info("Upload or select .dat files to begin analysis.")
    st.stop()

if process_btn:
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

            # Quality analysis
            quality = analyze_holter_quality(lead1, lead2, sampling_rate=200)

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
            qcol1, qcol2 = st.columns(2)
            with qcol1:
                st.write(f"**Lead 1 Quality:** {q['lead1_quality']:.3f}")
                for flag, val in q["lead1_flags"].items():
                    st.write(f"  {flag}: {val:.3f}")
            with qcol2:
                st.write(f"**Lead 2 Quality:** {q['lead2_quality']:.3f}")
                for flag, val in q["lead2_flags"].items():
                    st.write(f"  {flag}: {val:.3f}")

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
                    title="Quality per Window (3s each)",
                    xaxis_title="Window",
                    yaxis_title="Quality Score",
                    yaxis_range=[0, 1.05],
                    barmode="group",
                    height=300,
                    template="plotly_dark",
                )
                st.plotly_chart(win_fig, use_container_width=True)

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

            st.plotly_chart(fig, use_container_width=True)

        # -- Raw vs Filtered comparison --
        raw_open_key = f"raw_open_{fname}"
        with st.expander(
            "Raw vs Filtered Comparison",
            expanded=st.session_state.get(raw_open_key, False),
        ):
            lead_choice = st.radio(
                "Lead", [1, 2], horizontal=True,
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
            st.plotly_chart(comp_fig, use_container_width=True)

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
                st.plotly_chart(rr_fig, use_container_width=True)
