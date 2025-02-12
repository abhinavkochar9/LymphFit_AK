import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
from PIL import Image

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Constants
EXERCISES = ["DeepBreathing"]  # Start with one exercise for testing
REF_VIDEO_DIR = "/Users/abhinavkochar/Desktop/Pose correction webapp/Reference_Videos"
OVERLAY_COLOR = (0, 255, 0)  # Green color for overlay

# Force portrait layout
st.markdown("""
<style>
    .main .block-container {
        max-width: 95%;
    }
    .stVideo {
        border-radius: 20px;
        padding: 10px;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def get_video_path(exercise):
    """Validate and return video path"""
    video_name = f"{exercise.lower().replace(' ', '_')}.mp4"
    video_path = os.path.join(REF_VIDEO_DIR, video_name)
    if not os.path.exists(video_path):
        st.error(f"Missing video: {video_path}")
        st.stop()
    return video_path

def initialize_pose():
    """Initialize MediaPipe Pose with error handling"""
    try:
        return mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    except Exception as e:
        st.error(f"Failed to initialize pose detection: {str(e)}")
        st.stop()

def main():
    st.title("Lymphatic Coach - Posture Correction System")
    
    # Main layout columns
    col1, col2 = st.columns([1,1], gap="medium")
    
    with col1:  # Expert Video Column
        st.header("Expert Demonstration")
        exercise = st.selectbox("Select Exercise", EXERCISES)
        video_path = get_video_path(exercise)
        
        # Video player with controlled size
        st.video(video_path, format="video/mp4", start_time=0)

    with col2:  # Live Session Column
        st.header("Live Session")
        
        # State management
        if 'session' not in st.session_state:
            st.session_state.session = {
                'active': False,
                'start_time': None,
                'landmarks': None
            }
        
        # Initialize only once
        if not st.session_state.session['landmarks']:
            st.session_state.session['landmarks'] = extract_landmarks(video_path)
        
        # Control buttons
        if not st.session_state.session['active']:
            if st.button("🚀 Start Exercise", type="primary"):
                st.session_state.session['active'] = True
                st.session_state.session['start_time'] = time.time() + 5  # 5s countdown
                st.rerun()
        
        if st.session_state.session['active']:
            handle_exercise_session()

def handle_exercise_session():
    """Manage live exercise session"""
    countdown = st.empty()
    camera_placeholder = st.empty()
    feedback = st.empty()
    
    # 5-second countdown
    now = time.time()
    if now < st.session_state.session['start_time']:
        remaining = int(st.session_state.session['start_time'] - now)
        countdown.markdown(f"<h1 style='text-align: center'>⏳ {remaining}</h1>", unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()
    
    countdown.empty()
    
    # Main exercise loop
    pose = initialize_pose()
    start_time = st.session_state.session['start_time']
    
    while st.session_state.session['active']:
        img_file = st.camera_input("Align with outline", key="livecam")
        
        if not img_file:
            st.session_state.session['active'] = False
            st.error("Camera disconnected!")
            st.stop()
        
        frame = Image.open(img_file)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # Get current reference frame
        elapsed = time.time() - start_time
        current_frame = int(elapsed * 30)  # 30 FPS
        
        if current_frame >= len(st.session_state.session['landmarks']):
            st.success("✅ Exercise Completed!")
            st.session_state.session['active'] = False
            break
        
        # Apply overlay
        ref_landmark = st.session_state.session['landmarks'][current_frame]
        if ref_landmark:
            frame = draw_overlay(frame, ref_landmark)
        
        # Display
        camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                              use_column_width=True)
        
        # Basic feedback
        feedback.markdown("""
        <div style='text-align: center'>
            <h3 style='color:green'>Follow the GREEN outline!</h3>
            <p>Keep your posture aligned</p>
        </div>
        """, unsafe_allow_html=True)

def extract_landmarks(video_path):
    """Extract pose landmarks from reference video"""
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks.append(results.pose_landmarks)
    
    cap.release()
    return landmarks

def draw_overlay(frame, landmarks):
    """Draw visible overlay with error handling"""
    try:
        overlay = frame.copy()
        mp_drawing.draw_landmarks(
            overlay,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=OVERLAY_COLOR, thickness=4, circle_radius=6),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=OVERLAY_COLOR, thickness=4)
        )
        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    except Exception as e:
        st.error(f"Overlay error: {str(e)}")
        return frame

if __name__ == "__main__":
    main()