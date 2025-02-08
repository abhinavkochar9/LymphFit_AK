import streamlit as st
import os
import cv2
import mediapipe as mp
import json
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np 

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Set the root directory for the files
ROOT_DIR = "/Users/abhinavkochar/Desktop/Pose correction webapp/CleanedData_final"

# App Title
st.title("LymphFit Data Visualization")

# Sidebar for Patient and Exercise Selection
st.sidebar.header("Select Patient and Exercise")

# Dynamically list patients
if os.path.exists(ROOT_DIR):
    patients = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]
    selected_patient = st.sidebar.selectbox("Select Patient", options=patients)

    if selected_patient:
        exercises = os.listdir(os.path.join(ROOT_DIR, selected_patient))
        selected_exercise = st.sidebar.selectbox("Select Exercise", options=exercises)

        exercise_dir = os.path.join(ROOT_DIR, selected_patient, selected_exercise)

        # Detect video, JSON, and CSV files
        video_file_path = None
        json_file_path = None
        csv_file_path = None
        for file in os.listdir(exercise_dir):
            if file.endswith(".mp4"):
                video_file_path = os.path.join(exercise_dir, file)
            elif file.endswith(".json"):
                json_file_path = os.path.join(exercise_dir, file)
            elif file.endswith(".csv"):
                csv_file_path = os.path.join(exercise_dir, file)

        if not video_file_path:
            st.sidebar.error("No video file found in the selected directory.")
        if not json_file_path:
            st.sidebar.error("No JSON file found in the selected directory.")
        if not csv_file_path:
            st.sidebar.error("No CSV file found in the selected directory.")
else:
    st.sidebar.error("Root directory not found! Please check your setup.")

# Temporal Data Storage
temporal_data = {
    "time": [],
    "left_shoulder": {"x": [], "y": [], "z": []},
    "right_shoulder": {"x": [], "y": [], "z": []},
    "left_elbow": {"x": [], "y": [], "z": []},
    "right_elbow": {"x": [], "y": [], "z": []},
    "left_wrist": {"x": [], "y": [], "z": []},
    "right_wrist": {"x": [], "y": [], "z": []},
}

# Function to Create Combined Graph
def plot_combined_graph(x, data, title, max_points=200):
    if len(x) > max_points:
        skip = len(x) // max_points
        x = x[::skip]
        data = {key: value[::skip] for key, value in data.items()}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data["x"], mode="lines", name="X", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=x, y=data["y"], mode="lines", name="Y", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=x, y=data["z"], mode="lines", name="Z", line=dict(color="green")))
    fig.update_layout(
        title=dict(text=title, font=dict(size=8)),
        xaxis=dict(title="Frame Index", titlefont=dict(size=8), tickfont=dict(size=7)),
        yaxis=dict(title="Value", titlefont=dict(size=8), tickfont=dict(size=7)),
        height=70,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig

# Function to Create EMG Graphs with Configurations
def plot_emg_graph(x, y, title, max_points=200):
    if len(x) > max_points:
        skip = len(x) // max_points
        x = x[::skip]
        y = y[::skip]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue")))
    fig.update_layout(
        title=dict(text=title, font=dict(size=8)),
        xaxis=dict(title="Frame Index", titlefont=dict(size=8), tickfont=dict(size=7)),
        yaxis=dict(title="Value", titlefont=dict(size=8), tickfont=dict(size=7)),
        height=70,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig

# Function to Process Video, JSON, and EMG Data
def process_data(video_file_path, json_file_path, csv_file_path, acc_placeholder, gyro_placeholder, temporal_graph_placeholders, emg_placeholders):
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    accelerometer_data = pd.DataFrame.from_dict(data.get("accelerometerData", {}), orient="index")
    accelerometer_data.index = accelerometer_data.index.astype(int)
    accelerometer_data.sort_index(inplace=True)

    gyroscope_data = pd.DataFrame.from_dict(data.get("gyroscopeData", {}), orient="index")
    gyroscope_data.index = gyroscope_data.index.astype(int)
    gyroscope_data.sort_index(inplace=True)

    # Load EMG data
    emg_data = pd.read_csv(csv_file_path)
    time_column = "Time_Index"
    affected_columns = [col for col in emg_data.columns if "Affected" in col and "NonAffected" not in col]
    non_affected_columns = [col for col in emg_data.columns if "NonAffected" in col]

    cap = cv2.VideoCapture(video_file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_per_frame = 1 / video_fps
    total_duration = total_frames * duration_per_frame

    # Normalize sensor data index to video duration
    sensor_timestamps = accelerometer_data.index
    normalized_sensor_timestamps = (sensor_timestamps / sensor_timestamps.max() * total_duration)

    points_per_frame = max(1, len(emg_data) // total_frames)

    progress_bar = st.progress(0)
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break

            # Process video frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                temporal_data["time"].append(frame_idx)
                for joint, idx in {
                    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
                    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
                    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
                    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
                    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
                }.items():
                    joint_data = landmarks[idx]
                    temporal_data[joint]["x"].append(joint_data.x)
                    temporal_data[joint]["y"].append(joint_data.y)
                    temporal_data[joint]["z"].append(joint_data.z)

            # Create a 1:1 black background square
            square_size = 500  # Define a uniform square size
            background = np.zeros((square_size, square_size, 3), dtype=np.uint8)  # Create a black square

            # Resize the frame while maintaining aspect ratio
            h, w, _ = frame.shape
            scale = min(square_size / h, square_size / w)
            resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Convert the frame's color from BGR to RGB
            resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Center the resized frame on the black square
            x_offset = (square_size - resized_frame_rgb.shape[1]) // 2
            y_offset = (square_size - resized_frame_rgb.shape[0]) // 2
            background[y_offset:y_offset + resized_frame_rgb.shape[0], x_offset:x_offset + resized_frame_rgb.shape[1]] = resized_frame_rgb

            # Display the centered video frame on the black square
            video_placeholder.image(background, use_column_width=True)

            # Update temporal graphs
            for joint, placeholder in temporal_graph_placeholders.items():
                if len(temporal_data["time"]) > 1:
                    placeholder.plotly_chart(
                        plot_combined_graph(
                            temporal_data["time"],
                            temporal_data[joint],
                            f"{joint.replace('_', ' ').capitalize()} (X, Y, Z)"
                        ),
                        use_container_width=True,
                    )

            # Update Accelerometer graph
            current_time = frame_idx * duration_per_frame
            sensor_indices = (normalized_sensor_timestamps <= current_time)[:len(accelerometer_data)]
            acc_current_data = accelerometer_data.loc[sensor_indices]
            acc_fig = go.Figure()
            acc_fig.add_trace(go.Scatter(y=acc_current_data["Acc_X"], mode="lines", name="Acc_X"))
            acc_fig.add_trace(go.Scatter(y=acc_current_data["Acc_Y"], mode="lines", name="Acc_Y"))
            acc_fig.add_trace(go.Scatter(y=acc_current_data["Acc_Z"], mode="lines", name="Acc_Z"))
            acc_fig.update_layout(
            title=dict(
                text="Accerometer Data",
                font=dict(size=10),  # Reduce title font size
                x=0.5,  # Center align the title
                xanchor="center"
            ),
            xaxis=dict(
                title="Time",
                titlefont=dict(size=8),  # Reduce x-axis title font size
                tickfont=dict(size=7),  # Reduce x-axis tick font size
            ),
            yaxis=dict(
                title="Acceleration",
                titlefont=dict(size=8),  # Reduce y-axis title font size
                tickfont=dict(size=7),  # Reduce y-axis tick font size
            ),
            height=100,  # Compact graph height
            margin=dict(l=10, r=10, t=30, b=10)  # Compact margins
        )
            acc_placeholder.plotly_chart(acc_fig, use_container_width=True)

            # Update Gyroscope graph
            sensor_indices = (normalized_sensor_timestamps <= current_time)[:len(gyroscope_data)]
            gyro_current_data = gyroscope_data.loc[sensor_indices]
            gyro_fig = go.Figure()
            gyro_fig.add_trace(go.Scatter(y=gyro_current_data["Gyro_X"], mode="lines", name="Gyro_X"))
            gyro_fig.add_trace(go.Scatter(y=gyro_current_data["Gyro_Y"], mode="lines", name="Gyro_Y"))
            gyro_fig.add_trace(go.Scatter(y=gyro_current_data["Gyro_Z"], mode="lines", name="Gyro_Z"))
            gyro_fig.update_layout(
            title=dict(
                text="Gyroscope Data",
                font=dict(size=10),  # Reduce title font size
                x=0.5,  # Center align the title
                xanchor="center"
            ),
            xaxis=dict(
                title="Time",
                titlefont=dict(size=8),  # Reduce x-axis title font size
                tickfont=dict(size=7),  # Reduce x-axis tick font size
            ),
            yaxis=dict(
                title="Angular Velocity",
                titlefont=dict(size=8),  # Reduce y-axis title font size
                tickfont=dict(size=7),  # Reduce y-axis tick font size
            ),
            height=100,  # Compact graph height
            margin=dict(l=10, r=10, t=30, b=10)  # Compact margins
        )
            gyro_placeholder.plotly_chart(gyro_fig, use_container_width=True)

            # Update EMG visualizations
            end_idx = min((frame_idx + 1) * points_per_frame, len(emg_data))
            for i, (placeholder, channel) in enumerate(zip(emg_placeholders["left"], affected_columns[:3])):
                fig = plot_emg_graph(
                    x=emg_data[time_column].iloc[:end_idx],
                    y=emg_data[channel].iloc[:end_idx],
                    title=f"Affected: {channel}"
                )
                placeholder.plotly_chart(fig, use_container_width=True)

            for i, (placeholder, channel) in enumerate(zip(emg_placeholders["right"], non_affected_columns[:3])):
                fig = plot_emg_graph(
                    x=emg_data[time_column].iloc[:end_idx],
                    y=emg_data[channel].iloc[:end_idx],
                    title=f"Non-Affected: {channel}"
                )
                placeholder.plotly_chart(fig, use_container_width=True)

            # Synchronize with video FPS
            time.sleep(1 / video_fps)
            frame_idx += 1
            progress_bar.progress(frame_idx / total_frames)

    cap.release()

# Main Visualization Section
if video_file_path and json_file_path and csv_file_path:

    # Layout with five columns
    col1, col2, col_video, col3, col4 = st.columns([1 , 1, 2, 1, 1])

    # Video in the center
    with col_video:
        video_placeholder = st.empty()

    # Left column graphs
    with col2:
        temporal_graph_placeholders = {
            "left_shoulder": st.empty(),
            "left_elbow": st.empty(),
            "left_wrist": st.empty(),
        }
        

    # Right column graphs
    with col3:
        temporal_graph_placeholders.update({
            "right_shoulder": st.empty(),
            "right_elbow": st.empty(),
            "right_wrist": st.empty(),
        })

    # EMG data placeholders
    emg_placeholders = {
        "left": [col1.empty() for _ in range(3)],
        "right": [col4.empty() for _ in range(3)],
    }

    acc_placeholder = st.empty()  # Accelerometer graph
    gyro_placeholder = st.empty()  # Gyroscope graph

    start_button = st.button("Start Visualization")
    
    if start_button:
        process_data(video_file_path, json_file_path, csv_file_path, acc_placeholder, gyro_placeholder, temporal_graph_placeholders, emg_placeholders)

else:
    st.info("Please select a patient and exercise to begin.")