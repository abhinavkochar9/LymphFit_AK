import os
import time
import base64
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ----------------------------
# STEP 1: Choose the Expert (Reference) Video
# ----------------------------

# Folder where your expert videos are stored.
VIDEO_DIR = "/Users/abhinavkochar/Desktop/Pose correction webapp/Reference_Videos"

if not os.path.isdir(VIDEO_DIR):
    st.error(f"Directory '{VIDEO_DIR}' does not exist. Please create it and add your expert videos (e.g. MP4 files).")
    st.stop()

# List available MP4 files in the directory.
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")])
if not video_files:
    st.error(f"No MP4 files found in '{VIDEO_DIR}'. Please add at least one expert video.")
    st.stop()

# Let the user select one of the available expert videos.
selected_video = st.sidebar.selectbox("Select Expert (Reference) Video", video_files)
expert_video_path = os.path.join(VIDEO_DIR, selected_video)
st.sidebar.write(f"**Selected Expert Video:** {selected_video}")

# ----------------------------
# STEP 2: Compute the Expert Video Pose Outlines
# ----------------------------
@st.cache_data(show_spinner=True)
def compute_expert_outlines(video_path):
    """
    Processes the expert video frame-by-frame using MediaPipe Pose to detect keypoints
    and compute a convex hull outline (the reference pose) for each frame.
    
    Returns:
        outlines: A list of numpy arrays (one per frame, or None if no pose is detected)
        ref_w: Frame width of the expert video
        ref_h: Frame height of the expert video
        fps: Frames per second of the expert video
        total_frames: Total number of frames in the expert video
    """
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, None, None, None, None
    ref_h, ref_w, _ = frame.shape
    outlines = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                points = []
                for lm in results.pose_landmarks.landmark:
                    x = int(lm.x * ref_w)
                    y = int(lm.y * ref_h)
                    points.append([x, y])
                points = np.array(points)
                hull = cv2.convexHull(points)
                outlines.append(hull)
            else:
                outlines.append(None)
    cap.release()
    return outlines, ref_w, ref_h, fps, total_frames

st.info("Processing expert video to compute pose outlines...")
(expert_outlines, ex_ref_w, ex_ref_h, ex_fps, ex_total_frames) = compute_expert_outlines(expert_video_path)
if expert_outlines is None:
    st.error("Error processing the expert video. Please check the file and its format.")
    st.stop()
else:
    st.success("Expert video processed successfully!")

# ----------------------------
# STEP 3: Helper Function to Convert Video to Base64 (for HTML embedding)
# ----------------------------
def get_video_base64(video_path):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode()

# ----------------------------
# STEP 4: Define the Video Processor for the Webcam Feed
# ----------------------------
class PoseOverlayProcessor(VideoProcessorBase):
    def __init__(self, expert_outlines, ex_ref_w, ex_ref_h, ex_fps, ex_total_frames, start_time):
        self.expert_outlines = expert_outlines
        self.ex_ref_w = ex_ref_w
        self.ex_ref_h = ex_ref_h
        self.ex_fps = ex_fps
        self.ex_total_frames = ex_total_frames
        self.start_time = start_time  # Timestamp when the exercise started

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            frame_index = int(elapsed * self.ex_fps) % self.ex_total_frames
            outline = self.expert_outlines[frame_index]
            if outline is not None:
                frame_h, frame_w, _ = img.shape
                scale_x = frame_w / self.ex_ref_w
                scale_y = frame_h / self.ex_ref_h
                scaled_outline = outline.astype(np.float32).copy()
                scaled_outline[:, 0, 0] *= scale_x
                scaled_outline[:, 0, 1] *= scale_y
                scaled_outline = scaled_outline.astype(np.int32)
                cv2.polylines(img, [scaled_outline], isClosed=True, color=(0, 255, 0), thickness=3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# STEP 5: Build the App Layout
# ----------------------------
st.set_page_config(page_title="Real-Time Pose Correction", layout="wide")
st.title("Real-Time Pose Correction Application")

# Initialize the exercise start time in session state.
if "start_time" not in st.session_state:
    st.session_state["start_time"] = None

# Create two columns: left for the expert video, right for your webcam feed.
col1, col2 = st.columns(2)

# --- Controls ---
with st.container():
    start_pressed = st.button("Start Exercise")
    stop_pressed = st.button("Stop Exercise")
    if start_pressed:
        st.session_state["start_time"] = time.time()
    if stop_pressed:
        st.session_state["start_time"] = None

# --- Left Column: Expert (Reference) Video ---
with col1:
    st.header("Expert Video (Reference)")
    if st.session_state["start_time"] is not None:
        # Autoplay the expert video (muted to satisfy browser policies)
        video_b64 = get_video_base64(expert_video_path)
        video_html = f"""
            <video width="100%" controls autoplay muted>
              <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
              Your browser does not support the video tag.
            </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.info("Press **Start Exercise** to play the expert video.")

# --- Right Column: Webcam Feed with Dynamic Pose Overlay ---
with col2:
    st.header("Webcam Feed (Your View)")
    if st.session_state["start_time"] is not None:
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_streamer(
            key="pose-correction",
            rtc_configuration=rtc_config,
            video_processor_factory=lambda: PoseOverlayProcessor(
                expert_outlines, ex_ref_w, ex_ref_h, ex_fps, ex_total_frames, st.session_state["start_time"]
            ),
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.info("Press **Start Exercise** to activate your webcam feed.")