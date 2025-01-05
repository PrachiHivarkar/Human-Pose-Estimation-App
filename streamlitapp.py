import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize Mediapipe Pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f0f5;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #fafafa;
        }
        .big-header {
            font-size: 40px;
            color: #5e5e5e;
            text-align: center;
            font-weight: bold;
        }
        .description {
            font-size: 18px;
            color: #333333;
            text-align: justify;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            height: 3em;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

# Title with custom styling
st.markdown("<h1 class='big-header'>Human Pose Estimation App</h1>", unsafe_allow_html=True)

# Sidebar for file upload and settings
st.sidebar.header("Upload Options")
st.sidebar.write("Upload a video or image for pose estimation.")

# File uploader for video/image
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["mp4", "mov", "avi", "mkv", "jpg", "png"])

# FPS control for video speed
fps = st.sidebar.slider("Frames per second (FPS)", 1, 30, 5)

# Display a description
st.markdown("<p class='description'>This app detects and visualizes human pose from your uploaded media. Upload a video or image, and the app will detect the key points of the human body in real-time. Adjust the FPS to control the speed of the video playback.</p>", unsafe_allow_html=True)

# Handle video or image upload
if uploaded_file:
    # Check if the uploaded file is an image or video
    if uploaded_file.type.startswith("image"):
        # Read and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Process pose estimation on the image
        results = pose.process(img_array)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                img_array, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
            )
            # Show image with pose landmarks
            st.image(img_array, caption="Pose Estimation on Image", use_column_width=True)
        else:
            st.error("No human detected in the image.")
        
    elif uploaded_file.type.startswith("video"):
        # Save uploaded video to a temporary file
        temp_video = "temp_video.mp4"
        with open(temp_video, "wb") as f:
            f.write(uploaded_file.read())
        
        # Open the video file
        cap = cv2.VideoCapture(temp_video)

        # Check if video opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open video.")
        else:
            st.success("Video uploaded successfully!")

            # Create a placeholder to display video frames
            stframe = st.empty()

            # Read and process the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("End of video.")
                    break

                # Resize the frame for display
                frame = cv2.resize(frame, (600, 400))

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform pose estimation
                results = pose.process(frame_rgb)

                # Draw landmarks on the frame
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                    )

                # Convert the frame to an image and display
                img = Image.fromarray(frame_rgb)
                stframe.image(img, caption="Pose Estimation", use_column_width=True)

                # Wait before showing the next frame based on FPS
                cv2.waitKey(int(1000 / fps))

            cap.release()

else:
    st.info("Please upload a video or image file to start pose estimation.")

# Release resources
pose.close()
