import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ============ Data Augmentation Functions ============

def add_noise(X, noise_level=0.02):
    return X + noise_level * np.random.randn(*X.shape)

def shift_frames(X, max_shift=2):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(X, shift, axis=1)  # shift along the time axis

def augment_dataset(X, y, factor=3):
    X_aug = []
    y_aug = []
    for i in range(len(X)):
        for _ in range(factor):
            x_new = add_noise(X[i])
            x_new = shift_frames(x_new)
            X_aug.append(x_new)
            y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)

# ============ Load Data ============

X = np.load('X.npy')  # shape: (samples, 30, 126)
y = np.load('y.npy')
label_map = np.load('label_map.npy', allow_pickle=True).item()

print(f"📊 Original Data: X={X.shape}, y={y.shape}, Classes={len(label_map)}")

# Encode labels to one-hot
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Data Augmentation
X_aug, y_aug = augment_dataset(X, y_encoded, factor=3)

# Combine original + augmented
X_total = np.concatenate([X, X_aug], axis=0)
y_total = np.concatenate([y_encoded, y_aug], axis=0)

print(f"🧪 Augmented Data: X={X_total.shape}, y={y_total.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_total, y_total, test_size=0.2, random_state=42, shuffle=True)

# ============ Additive Attention Layer ============

from tensorflow.keras.layers import Layer

class AdditiveAttention(Layer):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values):
        hidden_with_time_axis = tf.expand_dims(values[:, -1], 1)  # Use last timestep
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# ============ Model Architecture ============

input_layer = Input(shape=(30, 126))
x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Bidirectional(LSTM(32, return_sequences=True, activation='tanh'))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

attention = AdditiveAttention(units=64)(x)

dense = Dense(64, activation='relu')(attention)
dense = Dropout(0.3)(dense)
output = Dense(len(label_map), activation='softmax')(dense)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("📐 BiLSTM + Additive Attention Model Summary:")
model.summary()

# ============ Callbacks ============

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_sign_language_additiveattention2_model.h5', monitor='val_loss', save_best_only=True)
]

# ============ Train Model ============

print("🚀 Training model with Additive Attention...")
history = model.fit(
    X_train, y_train,
    epochs=70,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Save model + label encoder
model.save("final_sign_language_additiveattention2_model.h5")
with open("label_binarizer.pkl", "wb") as f:
    pickle.dump(lb, f)

print("✅ Model & label encoder saved!")

# ============ Plot Training History ============

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ============ Evaluation ============

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📄 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_map.values()))

cm = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title("🧠 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
