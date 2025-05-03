# save this as app.py

import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import pickle
import tempfile
from tensorflow.keras.layers import Layer, Dense

# ==================== Load Model ====================

# Define the custom AdditiveAttention layer
class AdditiveAttention(Layer):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values):
        hidden_with_time_axis = tf.expand_dims(values[:, -1], 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        return tf.reduce_sum(context_vector, axis=1)

# Load the model
custom_objects = {'AdditiveAttention': AdditiveAttention}
model = tf.keras.models.load_model("best_sign_language_additiveattention_model.h5", custom_objects=custom_objects)

# Load the label binarizer
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

words = [
    "Busy", "But", "Day", "Good morning", "Happy", "Hello", "Hope", "I am",
    "Learn", "New", "Sign", "Thankyou", "Things", "Time", "Today", "We"
]

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_two_hands(results):
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if results.multi_handedness and results.multi_hand_landmarks:
        for i, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label
            landmarks = results.multi_hand_landmarks[i]
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

            if label == "Left":
                left_hand = landmark_array
            else:
                right_hand = landmark_array

    return np.concatenate([left_hand, right_hand]).flatten()

def predict_sign_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened() and len(sequence) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        keypoints = extract_two_hands(results)
        sequence.append(keypoints)

    cap.release()

    while len(sequence) < 30:
        sequence.append(np.zeros(126))

    input_data = np.expand_dims(np.array(sequence), axis=0)

    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)
    predicted_confidence = np.max(prediction)
    predicted_word = words[predicted_index]

    return predicted_word, predicted_confidence

# ==================== Streamlit App ====================

st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

# Sidebar
st.sidebar.title("🎥 Upload Your Video Here")
uploaded_file = st.sidebar.file_uploader("Upload a Video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Show video smaller in sidebar
    st.sidebar.video(tmp_path)

# Main Page
st.title("🤟 Sign Language Recognition App")
st.markdown(
    """
    Welcome to the Sign Language Recognition App!  
    Upload your sign language video and let our AI model predict what you're saying.  
    """
)

if uploaded_file is not None:
    with st.spinner('🔍 Analyzing the video... Please wait...'):
        predicted_word, confidence = predict_sign_from_video(tmp_path)

    st.subheader("Prediction Results")
    st.success(f"🧠 **Predicted Sign:** `{predicted_word}`")
    st.info(f"🎯 **Confidence Score:** `{confidence:.2f}`")

    if confidence < 0.6:
        st.warning("⚠️ Low confidence. Please try with a clearer video.")
    else:
        st.success("✅ High confidence prediction!")

else:
    st.info("📂 Please upload a video from the sidebar to get started!")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, TensorFlow, and Mediapipe")
