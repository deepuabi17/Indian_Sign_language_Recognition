import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Configuration
DATASET_PATH = r'C:\Users\DELL\Downloads\SignDataset'
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21 * 3 * 2  # 21 landmarks * (x, y, z) * 2 hands

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

X, y = [],[]
label_map = {}
labels = []

def extract_two_hands(results):
    # Initialize both hands with zeros
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

    combined = np.concatenate([left_hand, right_hand]).flatten()
    return combined

print("⏳ Starting dataset processing...")

for idx, label in enumerate(sorted(os.listdir(DATASET_PATH))):
    label_map[idx] = label
    labels.append(label)
    class_folder = os.path.join(DATASET_PATH, label)

    for video_name in tqdm(os.listdir(class_folder), desc=f"Processing '{label}'"):
        video_path = os.path.join(class_folder, video_name)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            landmarks = extract_two_hands(results)
            sequence.append(landmarks)

            frame_count += 1
            if frame_count == SEQUENCE_LENGTH:
                break

        cap.release()

        # Pad if sequence too short
        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append(np.zeros(NUM_LANDMARKS))

        X.append(sequence)
        y.append(idx)

hands.close()

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save to files
np.save("X.npy", X)
np.save("y.npy", y)
np.save("label_map.npy", label_map)

print(f"✅ Saved extracted features: X.shape = {X.shape}, y.shape = {y.shape}")