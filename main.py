import numpy as np
import scipy.io as sio
import re
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from umap import UMAP  # Make sure umap-learn is installed

# -------- CONFIG CONSTANTS --------
N_CHANNELS = 62          # Original number of EEG channels
REDUCED_CHANNELS = 2     # New channel dimension after applying manifold learning (UMAP)
SAMPLING_RATE = 200      # Hz (downsampled from 1000Hz)
SAMPLES_PER_EPOCH = 256  # Epoch length in samples
NUM_CLASSES = 4          # SEED-IV has 4 classes (0: Neutral, 1: Sad, 2: Fear, 3: Happy)

# Directory containing .mat files for SEED-IV (session 1)
DATASET_DIR = r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1"

# Regex pattern to match EEG trial keys (e.g., "ha_eeg1", "cz_eeg24")
EEG_KEY_PATTERN = re.compile(r".+_eeg\d+$")
# Session 1 labels from ReadMe.txt (mapping: 0: Neutral, 1: Sad, 2: Fear, 3: Happy)
SESSION1_LABEL = np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3])

# -------- HELPER: FIND EEG KEYS --------
def find_eeg_keys(mat_dict, pattern=EEG_KEY_PATTERN):
    """Recursively search for keys matching the given pattern."""
    top_keys = [k for k in mat_dict.keys() if pattern.match(k)]
    if top_keys:
        return {k: mat_dict[k] for k in top_keys}
    for k in mat_dict.keys():
        val = mat_dict[k]
        if isinstance(val, np.ndarray) and val.dtype.names is not None:
            subkeys = [fld for fld in val.dtype.names if pattern.match(fld)]
            if subkeys:
                result = {}
                for fld in subkeys:
                    result[fld] = val[fld][0,0]
                return result
    return {}

# -------- STEP 1: LOAD DATASET --------
def load_seed_dataset(mat_path):
    """
    Load EEG data and labels from a .mat file for SEED-IV session 1.
    It detects trial keys, segments each trial into epochs, and assigns labels.
    """
    mat_data = sio.loadmat(mat_path)
    eeg_dict = find_eeg_keys(mat_data, EEG_KEY_PATTERN)
    if not eeg_dict:
        raise ValueError(f"No EEG trial keys found in file: {mat_path}")
    sorted_keys = sorted(eeg_dict.keys(), key=lambda x: int(x.split("eeg")[-1]))
    min_length = min(eeg_dict[k].shape[1] for k in sorted_keys)
    trials = [eeg_dict[k][:, :min_length] for k in sorted_keys]
    
    segment_length = SAMPLES_PER_EPOCH
    epochs_list = []
    for trial in trials:
        num_epochs_trial = trial.shape[1] // segment_length
        trial = trial[:, :num_epochs_trial * segment_length]
        trial_epochs = trial.reshape(N_CHANNELS, num_epochs_trial, segment_length)
        trial_epochs = np.transpose(trial_epochs, (1, 0, 2))  # (num_epochs, channels, time)
        epochs_list.append(trial_epochs)
    eeg_data = np.concatenate(epochs_list, axis=0)  # (total_epochs, 62, segment_length)
    
    if len(SESSION1_LABEL) != len(sorted_keys):
        raise ValueError(f"Mismatch: Found {len(sorted_keys)} trials but {len(SESSION1_LABEL)} labels.")
    epochs_per_trial = epochs_list[0].shape[0]
    labels = []
    for label in SESSION1_LABEL:
        labels.extend([label] * epochs_per_trial)
    labels = np.array(labels)
    
    return eeg_data, labels

# -------- STEP 2: PREPROCESS DATA --------
def preprocess_seed_eeg(eeg_data):
    """
    Preprocess EEG data: apply bandpass filter (0.5-45 Hz), notch filter (50 Hz),
    and z-score normalization per epoch per channel.
    Input shape: (epochs, channels, time)
    """
    preprocessed_epochs = []
    for epoch in eeg_data:
        # Skip empty epochs
        if epoch.shape[1] == 0:
            continue
        try:
            filtered = butter_bandpass_filter(epoch, lowcut=0.5, highcut=45, fs=SAMPLING_RATE, order=4)
            filtered = notch_filter(filtered, freq=50, fs=SAMPLING_RATE)
            preprocessed_epochs.append(filtered)
        except Exception as e:
            print(f"Warning: Skipping an epoch due to filtering error. Epoch shape: {epoch.shape}. Error: {e}")
            continue
    if not preprocessed_epochs:
        raise ValueError("No valid epochs were preprocessed.")
    preprocessed = np.array(preprocessed_epochs)
    normalized = (preprocessed - np.mean(preprocessed, axis=2, keepdims=True)) / (np.std(preprocessed, axis=2, keepdims=True) + 1e-8)
    return normalized

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

def notch_filter(data, freq, fs, quality=30):
    nyquist = 0.5 * fs
    freq = freq / nyquist
    b, a = iirnotch(freq, quality)
    return filtfilt(b, a, data, axis=1)

# -------- STEP 2.5: SCALE DATASET --------
def scale_dataset(X):
    """
    Scale the dataset using StandardScaler.
    X is expected to have shape: (total_epochs, time, channels, 1)
    """
    from sklearn.preprocessing import StandardScaler
    original_shape = X.shape
    X_reshaped = X.reshape(original_shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    return X_scaled.reshape(original_shape)

# -------- MANIFOLD LEARNING: APPLY UMAP --------
def apply_umap(X, n_components=REDUCED_CHANNELS, batch_size=5000):
    """
    Apply UMAP to reduce channel dimensions while handling memory efficiently.
    Processes data in batches to prevent memory overflow.
    """
    total_epochs, time, channels, _ = X.shape
    X_reshaped = X.reshape(total_epochs * time, channels)

    reducer = UMAP(n_neighbors=10, n_components=n_components, min_dist=0.1, 
                   metric="euclidean", random_state=42)
    
    X_umap_list = []
    for i in range(0, X_reshaped.shape[0], batch_size):
        batch = X_reshaped[i : i + batch_size]
        X_umap_list.append(reducer.fit_transform(batch))
    
    X_umap = np.vstack(X_umap_list)
    X_umap = X_umap.reshape(total_epochs, time, n_components, 1)
    
    return X_umap

# -------- STEP 3: BUILD CNN MODEL --------
def build_cnn(input_shape=(SAMPLES_PER_EPOCH, REDUCED_CHANNELS), num_classes=NUM_CLASSES):
    """
    Build a simple CNN model for emotion classification.
    Updated input shape is now (256, reduced_channels, 1) after manifold learning.
    """
    inputs = tf.keras.Input(shape=input_shape + (1,))  # expects (256, reduced_channels, 1)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
    # Changed pooling to (2,1) to avoid negative dimensions (width remains >= 1)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# -------- STEP 4: DATA PIPELINE --------
def seed_data_pipeline(mat_paths):
    X_data, y_labels = [], []
    for path in mat_paths:
        eeg_data_raw, labels_raw = load_seed_dataset(path)
        preprocessed_data = preprocess_seed_eeg(eeg_data_raw)  # shape: (epochs, channels, time)
        X_data.extend(preprocessed_data)
        y_labels.extend(labels_raw)
    X_array = np.array(X_data)  # shape: (total_epochs, channels, time)
    # Swap axes: (total_epochs, time, channels)
    X_array = np.transpose(X_array, (0, 2, 1))
    X_array = X_array[..., np.newaxis]  # shape: (total_epochs, time, channels, 1)
    
    # Scale the dataset
    X_array = scale_dataset(X_array)
    
    # Apply UMAP for manifold learning to reduce channels from 62 to REDUCED_CHANNELS
    X_array = apply_umap(X_array, n_components=REDUCED_CHANNELS)
    
    y_array = tf.keras.utils.to_categorical(np.array(y_labels, dtype=int), num_classes=NUM_CLASSES)
    
    # Oversample the training data (we do oversampling before splitting)
    X_array, y_array = oversample_training_data(X_array, y_array)
    
    # Split the dataset into training and testing sets
    targets = y_array.argmax(axis=1)
    stratify_option = targets if len(np.unique(targets)) > 1 else None
    return train_test_split(X_array, y_array, test_size=0.2, stratify=stratify_option)

# -------- OVERSAMPLING FUNCTION --------
def oversample_training_data(X_train, y_train):
    """
    Oversample the training data using RandomOverSampler.
    X_train is expected to have shape (num_samples, time, channels, 1)
    y_train is one-hot encoded.
    """
    from imblearn.over_sampling import RandomOverSampler
    y_train_labels = np.argmax(y_train, axis=1)
    num_samples, time, channels, _ = X_train.shape
    X_train_flat = X_train.reshape(num_samples, -1)
    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train_flat, y_train_labels)
    
    X_resampled = X_resampled.reshape(-1, time, channels, 1)
    y_resampled = tf.keras.utils.to_categorical(y_resampled, num_classes=NUM_CLASSES)
    print("Oversampled Training Data Shape:", X_resampled.shape)
    return X_resampled, y_resampled

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    plt.show()

# -------- MAIN EXECUTION --------
if __name__ == "__main__":
    # List of 15 session 1 files for a subject (update filenames as needed)
    dataset_paths = [
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\1_20160518.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\2_20150915.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\3_20150919.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\4_20151111.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\5_20160406.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\6_20150507.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\7_20150715.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\8_20151103.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\9_20151028.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\10_20151014.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\11_20150916.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\12_20150725.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\13_20151115.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\14_20151205.mat",
        r"C:\sandhyaa\ai\archive\seed_iv\eeg_raw_data\1\15_20150508.mat",
    ]
    
    X_train, X_test, y_train, y_test = seed_data_pipeline(dataset_paths)
    print("Training Data Shape:", X_train.shape)                                                                                                                     
    print("Testing Data Shape:", X_test.shape)
    
    model = build_cnn(input_shape=(SAMPLES_PER_EPOCH, REDUCED_CHANNELS), num_classes=NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    print("Training the model for 30 epochs...")
    history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
                        epochs=30, batch_size=32, callbacks=[lr_reduce])
    
    plot_history(history)
    
    # Save the trained model as an HDF5 file
    model.save("my_model.h5")
    print("Model saved as 'my_model.h5'.")
