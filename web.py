import streamlit as st
import av
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="ID Guard",
    page_icon="🪪",
    layout="wide"
)

# -------------------------------
# CUSTOM DESIGN
# -------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 0;
    }

    .subtitle {
        color: #64748B;
        font-size: 17px;
        margin-top: 4px;
    }

    .panel {
        background-color: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }

    .status {
        background-color: #FFF7ED;
        color: #9A3412;
        padding: 14px;
        border-radius: 10px;
        font-weight: 600;
    }

    .reminder {
        background-color: #EFF6FF;
        color: #000000;
        padding: 16px;
        border-radius: 10px;
        font-weight: 600;
    }

    .small-text {
        color: #64748B;
        font-size: 13px;
    }
    
    .info {
        color: #5c697d;
    }
    
    /* Make the main text dark */
.stApp,
.stApp p,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp li,
.stApp span,
.stApp div {
    color: #1E293B;
}

/* Make input labels readable */
.stApp label {
    color: #1E293B !important;
}

/* Keep secondary text readable */
.small-text {
    color: #475569 !important;
}

#BUTTON STUFF
/* Style the camera's Take Photo button */
[data-testid="stCameraInput"] button {
    background-color: #172554 !important;
    color: #FFFFFF !important;
    border: 5px solid #172554 !important;
}

/* Keep the button text white */
[data-testid="stCameraInput"] button p,
[data-testid="stCameraInput"] button span {
    color: #FFFFFF !important;
}

/* Slightly change the button on hover */
[data-testid="stCameraInput"] button:hover {
    background-color: #0c1152 !important;
    border-color: #0c1152 !important;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# LOAD THE TRAINED AI MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/keras_model.h5",
        compile=False
    )

model = load_model()

# Read the model's class names
with open("model/labels.txt", "r", encoding="utf-8") as file:
    labels = []

    for line in file:
        parts = line.strip().split(" ", 1)

        if len(parts) == 2:
            labels.append(parts[1])
        elif parts[0]:
            labels.append(parts[0])

st.write("AI model loaded successfully!")

# -------------------------------
# LIVE AI CAMERA PROCESSOR
# -------------------------------

class CameraProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_label = "Waiting for detection"
        self.last_confidence = 0.0

    def recv(self, frame):

        # Convert webcam frame into an image
        img = frame.to_ndarray(format="rgb24")

        self.frame_count += 1

        # Analyse every 5th frame to reduce processing
        if self.frame_count % 5 == 0:

            # Resize image to Teachable Machine size
            resized = cv2.resize(img, (224, 224))

            # Prepare image for TensorFlow
            image_array = np.asarray(
                resized, dtype=np.float32
            )

            # Normalise pixels from 0-255 to -1 to 1
            image_array = (image_array / 127.5) - 1

            # Add batch dimension
            image_array = np.expand_dims(
                image_array, axis=0
            )

            # Ask the AI to make a prediction
            predictions = model.predict(
                image_array, verbose=0
            )[0]

            predicted_index = int(
                np.argmax(predictions)
            )

            self.last_confidence = float(
                predictions[predicted_index]
            )

            self.last_label = labels[predicted_index]

        # Decide what message to display
        if self.last_label.lower() == "id card":
            message = "ID CARD DETECTED"
            colour = (0, 200, 0)

        elif self.last_label.lower() == "no id card":
            message = "PLEASE WEAR YOUR ID CARD"
            colour = (255, 50, 50)

        else:
            message = "UNKNOWN CLASS"
            colour = (255, 200, 0)

        # Display prediction on the video
        cv2.putText(
            img,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2
        )

        cv2.putText(
            img,
            f"Confidence: {self.last_confidence:.1%}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Return the updated live video
        return av.VideoFrame.from_ndarray(
            img, format="rgb24"
        )


# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    '<div class="main-title">🪪 ID GUARD</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart School ID Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown(
    '<div class="info">'
    "Welcome to ID Guard! Capture an image to prepare "
    "for an ID card check. AI detection is coming soon."
    '</div>',
    unsafe_allow_html=True
)

# -------------------------------
# MAIN PAGE LAYOUT
# -------------------------------

camera_column, result_column = st.columns(
    [1.25, 1],
    gap="large"
)

with camera_column:

    st.markdown("### 📸 Camera Scanner")

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.write("Live student ID scanning")

    webrtc_streamer(
        key="id-guard-live",
        video_processor_factory=CameraProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.markdown(
        '<p class="small-text">'
        'ID Guard continuously analyses the camera feed.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "ID GUARD | Computer Vision School Project | "
)