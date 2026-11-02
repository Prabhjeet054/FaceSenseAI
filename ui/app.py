"""Streamlit dashboard for the FaceSense AI FastAPI backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.draw import draw_faces


DEFAULT_API_URL = os.getenv("FACESENSE_API_URL", "http://localhost:8000")
API_KEY = os.getenv("FACESENSE_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = 30


def inject_styles() -> None:
    """Apply the shared FaceSense visual system without changing app behavior."""
    st.markdown(
        """
        <style>
        :root {
            --fs-bg: #0b1020;
            --fs-panel: #121a2d;
            --fs-panel-soft: #17223a;
            --fs-border: rgba(148, 163, 184, .18);
            --fs-text: #e8eefc;
            --fs-muted: #9aa8c2;
            --fs-accent: #65d8c6;
            --fs-accent-strong: #38bdf8;
        }
        .stApp {
            background: var(--fs-bg);
            color: var(--fs-text);
        }
        [data-testid="stHeader"] { background: transparent; }
        ::selection { background: rgba(101, 216, 198, .35); color: #fff; }
        :focus-visible { outline: 2px solid var(--fs-accent); outline-offset: 3px; }
        .block-container { max-width:  min(92vw, 1440px); padding-top: 2.25rem; }
        main h2:first-of-type { display: none; }
        [data-testid="stSidebar"] {
            background: #0d1528;
            border-right: 1px solid var(--fs-border);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
        .fs-brand {
            display: flex; align-items: center; gap: .75rem;
            padding: .25rem .25rem 1.35rem;
        }
        .fs-brand-mark {
            display: grid; place-items: center; width: 2.45rem; height: 2.45rem;
            border-radius: .8rem; color: #07131d; font-size: 1.25rem;
            background: var(--fs-accent);
        }
        .fs-brand-title { font-weight: 750; letter-spacing: -.02em; font-size: 1.05rem; }
        .fs-brand-subtitle { color: var(--fs-muted); font-size: .72rem; margin-top: .12rem; }
        .fs-hero {
            padding: .25rem 0 1.4rem; margin: .2rem 0 1.5rem;
            border-bottom: 1px solid var(--fs-border);
        }
        .fs-hero h2 { margin: 0; font-size: clamp(1.55rem, 3vw, 2.15rem); letter-spacing: -.03em; }
        .fs-hero p { color: var(--fs-muted); max-width: 70ch; margin: .55rem 0 0; line-height: 1.55; }
        [data-testid="stMetric"] {
            background: var(--fs-panel); border: 1px solid var(--fs-border);
            padding: .8rem 1rem; border-radius: .75rem;
        }
        [data-testid="stFileUploader"], [data-testid="stCameraInput"] {
            border: 1px dashed rgba(101, 216, 198, .42); border-radius: .9rem;
            background: var(--fs-panel); padding: .2rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: .55rem; font-weight: 650; min-height: 2.55rem;
            transition: border-color .15s ease, background-color .15s ease, transform .15s ease;
        }
        .stButton > button[kind="primary"] {
            background: var(--fs-accent); color: #06121d; border: 0;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(18, 26, 45, .5); border-color: var(--fs-border);
        }
        div[data-testid="stDataFrame"] { border: 1px solid var(--fs-border); border-radius: .8rem; overflow: hidden; }
        .fs-note {
            color: var(--fs-muted); font-size: .78rem; padding: .7rem .8rem;
            border: 1px solid var(--fs-border); background: rgba(101, 216, 198, .06);
            border-radius: .55rem; line-height: 1.45;
        }
        @media (max-width: 700px) {
            .block-container { padding: 1.25rem 1rem 2rem; }
            .fs-hero { margin-bottom: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent page header while keeping the existing page title."""
    st.markdown(
        f"""
        <section class="fs-hero">
            <h2>{title}</h2>
            <p>{description}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def api_url(path: str) -> str:
    return f"{st.session_state.api_url.rstrip('/')}/{path.lstrip('/')}"


def request_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return f"{response.status_code}: {detail or 'Backend request failed'}"


def backend_request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    files: Any = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    try:
        response = requests.request(
            method,
            api_url(path),
            params=params,
            files=files,
            data=data,
            headers={"X-API-Key": API_KEY},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(f"Could not connect to the API: {error}") from error
    if not response.ok:
        raise RuntimeError(request_error(response))
    if response.status_code == 204:
        return {}
    return response.json()


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The selected file is not a valid image.")
    return image


def analyze_image(data: bytes, filename: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image = decode_image(data)
    payload = {"image": (filename, data, "image/jpeg")}
    result = backend_request("POST", "/analyze", files=payload)
    return image, result["faces"]


def date_filter_params(date_from: Any, date_to: Any, person_id: str = "") -> dict[str, str]:
    return {
        key: str(value)
        for key, value in {
            "date_from": date_from,
            "date_to": date_to,
            "person_id": person_id.strip(),
        }.items()
        if value
    }


def build_gender_chart(log_table: pd.DataFrame) -> alt.Chart:
    """Build a gender distribution pie chart from recognition logs."""
    counts = (
        log_table["gender"]
        .value_counts()
        .rename_axis("gender")
        .reset_index(name="count")
    )
    return (
        alt.Chart(counts)
        .mark_arc()
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("gender:N"),
            tooltip=["gender:N", "count:Q"],
        )
    )


def build_age_group_chart(log_table: pd.DataFrame) -> alt.Chart:
    """Build an age-group count bar chart from recognition logs."""
    counts = (
        log_table["age_group"]
        .value_counts()
        .rename_axis("age_group")
        .reset_index(name="count")
    )
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("age_group:N", sort="-y"),
            y=alt.Y("count:Q"),
            tooltip=["age_group:N", "count:Q"],
        )
    )


def build_hourly_chart(log_table: pd.DataFrame) -> alt.Chart:
    """Build a 24-point detections-per-hour line chart."""
    timestamps = pd.to_datetime(log_table["time"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("recognition logs contain an invalid time")
    counts = (
        timestamps.dt.hour.value_counts()
        .reindex(range(24), fill_value=0)
        .rename_axis("hour")
        .reset_index(name="count")
    )
    counts["hour"] = counts["hour"].map(lambda hour: f"{hour:02d}:00")
    return (
        alt.Chart(counts)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:N", sort=None),
            y=alt.Y("count:Q"),
            tooltip=["hour:N", "count:Q"],
        )
    )


def show_analysis(image: np.ndarray, results: list[dict[str, Any]]) -> None:
    annotated = cv2.cvtColor(draw_faces(image, results), cv2.COLOR_BGR2RGB)
    st.subheader("Screening results")
    detected = len(results)
    identified = sum(
        str(result.get("name", "")).lower() != "unknown" for result in results
    )
    average_match = (
        float(np.mean([float(result.get("match_score", 0.0)) for result in results]))
        if results
        else 0.0
    )
    average_age = (
        float(np.mean([float(result.get("age", 0.0)) for result in results]))
        if results
        else 0.0
    )
    summary = st.columns(4)
    summary[0].metric("Faces detected", detected)
    summary[1].metric("Registered matches", identified)
    summary[2].metric("Avg. match confidence", f"{average_match:.0%}")
    summary[3].metric("Avg. estimated age", f"{average_age:.1f}" if results else "—")

    left, right = st.columns(2)
    with left:
        st.image(annotated, caption="Analyzed image", use_column_width=True)
    with right:
        table = pd.DataFrame(results)
        if table.empty:
            st.info("No faces were detected.")
        else:
            table["screening_status"] = np.where(
                table["name"].str.lower().eq("unknown"),
                "Unregistered / review",
                "Registered match",
            )
            table["match_confidence"] = table["match_score"].map(
                lambda value: f"{float(value):.1%}"
            )
            table["gender_confidence"] = table["gender_confidence"].map(
                lambda value: f"{float(value):.1%}"
            )
            if "emotion_confidence" in table:
                table["emotion_confidence"] = table["emotion_confidence"].map(
                    lambda value: f"{float(value):.1%}"
                )
            table["age"] = table["age"].map(lambda value: f"{float(value):.1f}")
            columns = [
                "screening_status",
                "name",
                "gender",
                "gender_confidence",
                "age",
                "age_group",
                "emotion",
                "emotion_confidence",
                "match_confidence",
            ]
            st.dataframe(
                table[[column for column in columns if column in table]],
                hide_index=True,
                use_container_width=True,
            )
            unknown_count = int(
                table["screening_status"].eq("Unregistered / review").sum()
            )
            if unknown_count:
                st.warning(
                    f"{unknown_count} face(s) were not confidently matched to the "
                    "registry. Unknown is not an identity prediction; register the "
                    "person with 3–5 consented photos before expecting a name."
                )
            st.caption(
                f"Input frame: {image.shape[1]}×{image.shape[0]} px · "
                "Age, gender, and emotion are approximate model estimates."
            )


def upload_and_analyze_page() -> None:
    st.header("Upload and Analyze")
    page_intro(
        "Analysis workspace",
        "Understand every detected face",
        "Upload a clear image to identify faces and estimate attributes in one pass.",
    )
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="analysis_upload"
    )
    if uploaded is not None and st.button("Analyze image", type="primary"):
        try:
            with st.spinner("Screening image…"):
                image, results = analyze_image(uploaded.getvalue(), uploaded.name)
            show_analysis(image, results)
        except (RuntimeError, ValueError, KeyError) as error:
            st.error(str(error))


def live_camera_page() -> None:
    st.header("Live Camera")
    page_intro(
        "Real-time capture",
        "Analyze a live camera frame",
        "Center one face in the frame, use even lighting, and capture a sharp image.",
    )
    st.caption("Use your browser camera to capture a frame and analyze it.")
    captured = st.camera_input("Capture a face image")
    if captured is not None and st.button("Analyze captured frame", type="primary"):
        try:
            with st.spinner("Screening captured frame…"):
                image, results = analyze_image(captured.getvalue(), "camera.jpg")
            show_analysis(image, results)
        except (RuntimeError, ValueError, KeyError) as error:
            st.error(str(error))


def register_person_page() -> None:
    st.header("Register Person")
    page_intro(
        "Consent-first registry",
        "Add a person securely",
        "Provide consent and 3–5 varied face photos for reliable recognition.",
    )
    name = st.text_input("Name")
    person_id = st.text_input("Person ID")
    consent = st.checkbox("I consent to biometric data processing and storage.")
    photos = st.file_uploader(
        "Upload 3 to 5 face photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="registration_photos",
    )
    if st.button("Register person", type="primary"):
        if not name.strip() or not person_id.strip():
            st.error("Name and person ID are required.")
            return
        if not consent:
            st.error("Consent is required for registration.")
            return
        if not 3 <= len(photos) <= 5:
            st.error("Registration requires 3 to 5 photos.")
            return
        files = [
            ("photos", (photo.name, photo.getvalue(), photo.type or "image/jpeg"))
            for photo in photos
        ]
        try:
            with st.spinner("Registering face profile…"):
                result = backend_request(
                    "POST",
                    "/register",
                    data={"name": name, "person_id": person_id, "consent": "true"},
                    files=files,
                )
            st.success(f"Registered {result['name']} ({result['id']}).")
        except (RuntimeError, KeyError) as error:
            st.error(str(error))


def logs_and_stats_page() -> None:
    st.header("Logs and Stats")
    page_intro(
        "Audit and insights",
        "Explore recognition activity",
        "Filter events by date or person to review system activity and trends.",
    )
    first, last, person = st.columns(3)
    with first:
        date_from = st.date_input("From date", value=None)
    with last:
        date_to = st.date_input("To date", value=None)
    with person:
        person_id = st.text_input("Person ID filter")
    params = date_filter_params(date_from, date_to, person_id)
    if st.button("Refresh logs", type="primary"):
        try:
            with st.spinner("Loading recognition activity…"):
                logs = backend_request("GET", "/logs", params=params)
                stats = backend_request("GET", "/stats", params=params)
            st.subheader("Recognition logs")
            log_table = pd.DataFrame(logs)
            if log_table.empty:
                st.info("No recognition events match these filters.")
            else:
                st.dataframe(log_table, hide_index=True, use_container_width=True)
            st.subheader("Statistics")
            metric_columns = st.columns(4)
            metric_columns[0].metric("Total", stats["total"])
            metric_columns[1].metric("Gender groups", len(stats["by_gender"]))
            metric_columns[2].metric("Age groups", len(stats["by_age_group"]))
            metric_columns[3].metric("Days", len(stats["by_day"]))
            st.json(stats)
        except (RuntimeError, KeyError) as error:
            st.error(str(error))


def attendance_page() -> None:
    st.header("Daily Attendance")
    page_intro(
        "Presence tracking",
        "Review daily attendance",
        "Export a clean Excel sheet for the selected day whenever you need it.",
    )
    attendance_date = st.date_input("Attendance date", value=None, key="attendance_date")
    if st.button("Refresh attendance", type="primary"):
        params = {"attendance_date": str(attendance_date)} if attendance_date else {}
        try:
            with st.spinner("Loading attendance records…"):
                records = backend_request("GET", "/attendance", params=params)
            table = pd.DataFrame(records)
            if table.empty:
                st.info("No attendance records were found for this date.")
            else:
                st.dataframe(table, hide_index=True, use_container_width=True)
                export_url = api_url("/attendance/export")
                if params:
                    export_url = f"{export_url}?attendance_date={attendance_date}"
                st.link_button("Download Excel attendance sheet", export_url)
        except (RuntimeError, KeyError) as error:
            st.error(str(error))


def dashboard_page() -> None:
    st.header("Dashboard")
    page_intro(
        "Operations overview",
        "See the bigger picture",
        "Use recognition logs to monitor volume, demographics, and peak activity.",
    )
    first, last = st.columns(2)
    with first:
        date_from = st.date_input("From date", value=None, key="dashboard_date_from")
    with last:
        date_to = st.date_input("To date", value=None, key="dashboard_date_to")
    params = date_filter_params(date_from, date_to)

    if st.button("Refresh dashboard", type="primary"):
        try:
            with st.spinner("Updating dashboard…"):
                logs = backend_request("GET", "/logs", params=params)
                stats = backend_request("GET", "/stats", params=params)
            log_table = pd.DataFrame(logs)

            if log_table.empty:
                st.info("No recognition events found for the selected date range.")
                return

            metric_columns = st.columns(4)
            metric_columns[0].metric("Detections", stats["total"])
            metric_columns[1].metric("People", log_table["person_id"].nunique())
            metric_columns[2].metric("Gender groups", len(stats["by_gender"]))
            metric_columns[3].metric("Active days", len(stats["by_day"]))

            st.subheader("Gender distribution")
            st.altair_chart(
                build_gender_chart(log_table),
                use_container_width=True,
            )

            st.subheader("Age-group distribution")
            st.altair_chart(
                build_age_group_chart(log_table),
                use_container_width=True,
            )

            st.subheader("Detections per hour")
            st.altair_chart(
                build_hourly_chart(log_table),
                use_container_width=True,
            )

            st.subheader("Most recent detections")
            recent = log_table.head(10)
            st.dataframe(recent, hide_index=True, use_container_width=True)
            st.download_button(
                "Download detections CSV",
                data=log_table.to_csv(index=False).encode("utf-8"),
                file_name="facesense_detections.csv",
                mime="text/csv",
            )
            st.caption(f"{stats['total']} detections in the selected range.")
        except (RuntimeError, KeyError, ValueError) as error:
            st.error(str(error))


def main() -> None:
    st.set_page_config(page_title="FaceSense AI", page_icon="🙂", layout="wide")
    inject_styles()
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    with st.sidebar:
        st.markdown(
            """
            <div class="fs-brand">
                <div class="fs-brand-mark">◉</div>
                <div>
                    <div class="fs-brand-title">FaceSense AI</div>
                    <div class="fs-brand-subtitle">Private vision workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.api_url = st.text_input(
            "FastAPI URL", value=st.session_state.api_url
        )
        st.markdown(
            '<div class="fs-note">Consent-based analysis · API-key protected</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            [
                "Dashboard",
                "Daily Attendance",
                "Upload and Analyze",
                "Live Camera",
                "Register Person",
                "Logs and Stats",
            ],
        )
    if page == "Dashboard":
        dashboard_page()
    elif page == "Daily Attendance":
        attendance_page()
    elif page == "Upload and Analyze":
        upload_and_analyze_page()
    elif page == "Live Camera":
        live_camera_page()
    elif page == "Register Person":
        register_person_page()
    else:
        logs_and_stats_page()


if __name__ == "__main__":
    main()
