"""Streamlit dashboard for the FaceSense AI FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from app.utils.draw import draw_faces


DEFAULT_API_URL = os.getenv("FACESENSE_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30


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
    left, right = st.columns(2)
    with left:
        st.image(annotated, caption="Analyzed image", use_column_width=True)
    with right:
        columns = ["name", "gender", "gender_confidence", "age", "age_group", "match_score"]
        table = pd.DataFrame(results)
        if table.empty:
            st.info("No faces were detected.")
        else:
            st.dataframe(
                table[[column for column in columns if column in table]],
                hide_index=True,
                use_container_width=True,
            )


def upload_and_analyze_page() -> None:
    st.header("Upload and Analyze")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="analysis_upload"
    )
    if uploaded is not None and st.button("Analyze image", type="primary"):
        try:
            image, results = analyze_image(uploaded.getvalue(), uploaded.name)
            show_analysis(image, results)
        except (RuntimeError, ValueError, KeyError) as error:
            st.error(str(error))


def live_camera_page() -> None:
    st.header("Live Camera")
    st.caption("Use your browser camera to capture a frame and analyze it.")
    captured = st.camera_input("Capture a face image")
    if captured is not None and st.button("Analyze captured frame", type="primary"):
        try:
            image, results = analyze_image(captured.getvalue(), "camera.jpg")
            show_analysis(image, results)
        except (RuntimeError, ValueError, KeyError) as error:
            st.error(str(error))


def register_person_page() -> None:
    st.header("Register Person")
    name = st.text_input("Name")
    person_id = st.text_input("Person ID")
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
        if not 3 <= len(photos) <= 5:
            st.error("Registration requires 3 to 5 photos.")
            return
        files = [
            ("photos", (photo.name, photo.getvalue(), photo.type or "image/jpeg"))
            for photo in photos
        ]
        try:
            result = backend_request(
                "POST",
                "/register",
                data={"name": name, "person_id": person_id},
                files=files,
            )
            st.success(f"Registered {result['name']} ({result['id']}).")
        except (RuntimeError, KeyError) as error:
            st.error(str(error))


def logs_and_stats_page() -> None:
    st.header("Logs and Stats")
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
            logs = backend_request("GET", "/logs", params=params)
            stats = backend_request("GET", "/stats", params=params)
            st.subheader("Recognition logs")
            st.dataframe(pd.DataFrame(logs), hide_index=True, use_container_width=True)
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
    attendance_date = st.date_input("Attendance date", value=None, key="attendance_date")
    if st.button("Refresh attendance", type="primary"):
        params = {"attendance_date": str(attendance_date)} if attendance_date else {}
        try:
            records = backend_request("GET", "/attendance", params=params)
            table = pd.DataFrame(records)
            st.dataframe(table, hide_index=True, use_container_width=True)
            export_url = api_url("/attendance/export")
            if params:
                export_url = f"{export_url}?attendance_date={attendance_date}"
            st.link_button("Download Excel attendance sheet", export_url)
        except (RuntimeError, KeyError) as error:
            st.error(str(error))


def dashboard_page() -> None:
    st.header("Dashboard")
    first, last = st.columns(2)
    with first:
        date_from = st.date_input("From date", value=None, key="dashboard_date_from")
    with last:
        date_to = st.date_input("To date", value=None, key="dashboard_date_to")
    params = date_filter_params(date_from, date_to)

    if st.button("Refresh dashboard", type="primary"):
        try:
            logs = backend_request("GET", "/logs", params=params)
            stats = backend_request("GET", "/stats", params=params)
            log_table = pd.DataFrame(logs)

            if log_table.empty:
                st.info("No recognition events found for the selected date range.")
                return

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
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    with st.sidebar:
        st.title("FaceSense AI")
        st.session_state.api_url = st.text_input(
            "FastAPI URL", value=st.session_state.api_url
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
