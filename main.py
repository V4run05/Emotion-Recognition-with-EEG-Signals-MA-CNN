import os
import numpy as np
import scipy.io as sio
import re
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import warnings
from umap.umap_ import UMAP  # Ensure umap-learn is installed
warnings.simplefilter(action='ignore', category=FutureWarning)
# Set the global policy
mixed_precision.set_global_policy('mixed_float16')

# -------- CONFIG CONSTANTS --------
N_CHANNELS = 62          # Original number of EEG channels
REDUCED_CHANNELS = 8     # New channel dimension after applying manifold learning (UMAP)
SAMPLING_RATE = 200      # Hz (downsampled from 1000Hz)
SAMPLES_PER_EPOCH = 800  # Epoch length in samples
NUM_CLASSES = 4          # SEED-IV has 4 classes (0: Neutral, 1: Sad, 2: Fear, 3: Happy)

# -------- UMAP PARAMETERS (adjust these to experiment) --------
umap_params = {
    'n_neighbors': 30,
    'n_components': REDUCED_CHANNELS,
    'min_dist': 0.01,
    'metric': "euclidean",
    'random_state': 42,
    'batch_size': 2000  # Adjust batch size if needed
}

# Regex pattern to match EEG trial keys (e.g., "ha_eeg1", "cz_eeg24")
EEG_KEY_PATTERN = re.compile(r".+_eeg\d+$")
# Session 1-3 labels from ReadMe.txt
SESSION1_LABEL = np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3])
SESSION2_LABEL = np.array([2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1])
SESSION3_LABEL = np.array([1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0])

# Create a dictionary mapping session numbers to label arrays
SESSION_LABELS = {
    1: SESSION1_LABEL,
    2: SESSION2_LABEL,
    3: SESSION3_LABEL
}

# -------- GPU MEMORY GROWTH SETUP --------
if tf.config.list_physical_devices('GPU'):
    gpus = tf.config.list_physical_devices('GPU')
    print("Using GPU:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Error setting GPU memory growth:", e)
else:
    print("GPU not found. Using CPU.")

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
def load_seed_dataset(mat_path, session_number=1):
    """
    Load EEG data and labels from a .mat file.
    It detects trial keys, truncates each trial to the same length,
    splits into epochs, and assigns labels for the specified session.
    If a file doesn't contain enough samples for even one epoch, it is skipped.
    """
    try:
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise ValueError(f"Error loading {mat_path}: {e}")
    
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
        if num_epochs_trial == 0:
            print(f"Warning: Not enough data in file: {mat_path}. Skipping this trial.")
            continue
        trial = trial[:, :num_epochs_trial * segment_length]
        trial_epochs = trial.reshape(N_CHANNELS, num_epochs_trial, segment_length)
        trial_epochs = np.transpose(trial_epochs, (1, 0, 2))  # (num_epochs, channels, time)
        epochs_list.append(trial_epochs)
    if not epochs_list:
        raise ValueError(f"Not enough data in file: {mat_path}")
    eeg_data = np.concatenate(epochs_list, axis=0)  # (total_epochs, 62, segment_length)
    
    # Get the label array for the specified session.
    if session_number not in SESSION_LABELS:
        raise ValueError("session_number must be 1, 2, or 3.")
    expected_labels = SESSION_LABELS[session_number]
    
    if len(expected_labels) != len(sorted_keys):
        raise ValueError(f"Mismatch: Found {len(sorted_keys)} trials but {len(expected_labels)} labels in {mat_path}")
    
    epochs_per_trial = epochs_list[0].shape[0]
    labels = []
    for label in expected_labels:
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
        if epoch.shape[1] == 0:
            continue
        try:
            filtered = butter_bandpass_filter(epoch, lowcut=0.5, highcut=45, fs=SAMPLING_RATE, order=4)
            filtered = notch_filter(filtered, freq=50, fs=SAMPLING_RATE)
            preprocessed_epochs.append(filtered)
        except Exception as e:
            print(f"Warning: Skipping an epoch with shape {epoch.shape} due to filtering error: {e}")
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
    from sklearn.preprocessing import StandardScaler
    original_shape = X.shape
    X_reshaped = X.reshape(original_shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    return X_scaled.reshape(original_shape)

def extract_frequency_features(eeg_data):
    """Extract frequency domain features using FFT"""
    from scipy.fft import fft
    features = []
    for epoch in eeg_data:
        # Calculate power spectrum for each channel
        fft_features = np.abs(fft(epoch, axis=1)[:, :SAMPLES_PER_EPOCH//2])
        # Extract band powers (delta, theta, alpha, beta, gamma)
        delta = np.mean(fft_features[:, 1:4], axis=1)  # 0.5-4 Hz
        theta = np.mean(fft_features[:, 4:8], axis=1)  # 4-8 Hz
        alpha = np.mean(fft_features[:, 8:13], axis=1)  # 8-13 Hz
        beta = np.mean(fft_features[:, 13:30], axis=1)  # 13-30 Hz
        gamma = np.mean(fft_features[:, 30:45], axis=1)  # 30-45 Hz
        # Combine features
        combined = np.vstack([delta, theta, alpha, beta, gamma]).T
        features.append(combined)
    return np.array(features)

def augment_eeg(X, y, noise_level=0.05):
    """Add Gaussian noise to EEG data for augmentation"""
    X_aug = X.copy()
    # Add random noise
    noise = np.random.normal(0, noise_level, X_aug.shape)
    X_aug += noise
    return X_aug, y

# -------- MANIFOLD LEARNING: APPLY UMAP --------
def apply_umap(X, n_components=REDUCED_CHANNELS, batch_size=umap_params['batch_size']):
    total_epochs, time, channels, _ = X.shape
    X_reshaped = X.reshape(total_epochs * time, channels)

    reducer = UMAP(n_neighbors=umap_params['n_neighbors'],
                   n_components=n_components,
                   min_dist=umap_params['min_dist'],
                   metric=umap_params['metric'],
                   random_state=umap_params['random_state'])

    X_umap_list = []
    for i in range(0, X_reshaped.shape[0], batch_size):
        batch = X_reshaped[i : i + batch_size]
        X_umap_list.append(reducer.fit_transform(batch))
    X_umap = np.vstack(X_umap_list)
    X_umap = X_umap.reshape(total_epochs, time, n_components, 1)
    return X_umap


# -------- STEP 3: BUILD CNN MODEL --------
def build_eeg_model(input_shape=(SAMPLES_PER_EPOCH, REDUCED_CHANNELS), num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape + (1,))
    
    # Temporal convolutions (across time)
    x = tf.keras.layers.Conv2D(128, (10, 1), padding='same', activation='elu', kernel_regularizer=l2(1e-3))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Spatial convolutions (across channels)
    x = tf.keras.layers.Conv2D(128, (1, REDUCED_CHANNELS), padding='valid', activation='elu', kernel_regularizer=l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Further temporal processing
    x = tf.keras.layers.Conv2D(256, (10, 1), padding='same', activation='elu', kernel_regularizer=l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Classification layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='elu', kernel_regularizer=l2(1e-3))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-3))(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# -------- STEP 4: DATA PIPELINE --------
def seed_data_pipeline(mat_paths, session_number):
    X_data, y_labels = [], []
    for path in mat_paths:
        try:
            eeg_data_raw, labels_raw = load_seed_dataset(path, session_number=session_number)
        except ValueError as e:
            print(f"Skipping file {path}: {e}")
            continue
        preprocessed_data = preprocess_seed_eeg(eeg_data_raw)  # shape: (epochs, channels, time)
        X_data.extend(preprocessed_data)
        y_labels.extend(labels_raw)
    if len(X_data) == 0:
        raise ValueError("No valid EEG data was loaded from the provided paths.")
    X_array = np.array(X_data)  # (total_epochs, channels, time)
    X_array = np.transpose(X_array, (0, 2, 1))  # (total_epochs, time, channels)
    X_array = X_array[..., np.newaxis]          # (total_epochs, time, channels, 1)
    X_array = scale_dataset(X_array)
    X_array = apply_umap(X_array, n_components=REDUCED_CHANNELS)  # output shape: (total_epochs, time, REDUCED_CHANNELS, 1)
    y_array = to_categorical(np.array(y_labels, dtype=int), num_classes=NUM_CLASSES)
    X_array, y_array = oversample_training_data(X_array, y_array)
    targets = y_array.argmax(axis=1)
    stratify_option = targets if len(np.unique(targets)) > 1 else None
    return train_test_split(X_array, y_array, test_size=0.2, stratify=stratify_option)

# -------- OVERSAMPLING FUNCTION --------
def oversample_training_data(X_train, y_train):
    from imblearn.over_sampling import RandomOverSampler
    y_train_labels = np.argmax(y_train, axis=1)
    num_samples, time, channels, _ = X_train.shape
    X_train_flat = X_train.reshape(num_samples, -1)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train_flat, y_train_labels)
    X_resampled = X_resampled.reshape(-1, time, channels, 1)
    y_resampled = to_categorical(y_resampled, num_classes=NUM_CLASSES)
    print("Oversampled Training Data Shape:", X_resampled.shape)
    return X_resampled, y_resampled

def plot_history(history, title="Reduced dataset"):
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
    
def plot_reduced_dataset(X, y, title="Reduced dataset"):
    """
    Plot the UMAP-reduced data in 2D.
    X is expected to have shape (samples, time, REDUCED_CHANNELS, 1).
    Instead of averaging over time, this function uses the first time slice.
    If REDUCED_CHANNELS > 2, further reduce the features to 2D using UMAP.
    y is expected to be one-hot encoded.
    """
    # Choose a specific time slice from each sample (here, the first time slice)
    X_reduced = X[:, 0, :, 0]  # Shape: (samples, REDUCED_CHANNELS)
    
    # If the reduced dimension is more than 2, use UMAP to reduce to 2D.
    if X_reduced.shape[1] > 2:
        reducer2d = UMAP(
            n_neighbors=umap_params['n_neighbors'],
            n_components=2,
            min_dist=umap_params['min_dist'],
            metric=umap_params['metric'],
            random_state=umap_params['random_state']
        )
        X_plot = reducer2d.fit_transform(X_reduced)
    else:
        X_plot = X_reduced

    # Obtain class labels from the one-hot encoding
    labels = np.argmax(y, axis=1)

    # Plot using matplotlib
    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(X_plot[idx, 0], X_plot[idx, 1], label=f"Class {c}", alpha=0.6)
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.show()

    
def plot_3d_reduced_dataset(X, y, title="3D Reduced dataset"):
    # Average across time.
    X_mean = np.mean(X.squeeze(-1), axis=1)  # shape: (samples, REDUCED_CHANNELS)
    if X_mean.shape[1] > 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_plot = pca.fit_transform(X_mean)
    else:
        X_plot = X_mean
    labels = np.argmax(y, axis=1)
    data = np.column_stack((X_plot, labels))
    df = pd.DataFrame(data, columns=['Dim 1', 'Dim 2', 'Dim 3', 'Label'])
    fig = px.scatter_3d(df, x='Dim 1', y='Dim 2', z='Dim 3', color='Label',
                        title=title)
    fig.show()

def plot_pairwise(X, y, title="Pairwise Scatter Matrix"):
    X_mean = np.mean(X.squeeze(-1), axis=1)
    labels = np.argmax(y, axis=1)
    df = pd.DataFrame(X_mean, columns=[f"Dim {i+1}" for i in range(X_mean.shape[1])])
    df['Label'] = labels
    sns.pairplot(df, hue='Label', diag_kind='hist')
    plt.suptitle(title)
    plt.show()


# First create a custom learning rate scheduler
class GradualLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr=0.001, min_lr=0.00001, warmup_epochs=5, patience=4, factor=0.8):
        super(GradualLearningRateScheduler, self).init()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.factor = factor
        self.best_val_loss = float('inf')
        self.wait = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Gradual warmup phase
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
        else:
            # Get the current learning rate
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            
        # Print for logging
        print(f"\nEpoch {epoch+1}: Learning rate set to {lr}")
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            
    def on_epoch_end(self, epoch, logs=None):
        # Skip during warmup
        if epoch < self.warmup_epochs:
            return
            
        # After warmup, implement ReduceLROnPlateau-like behavior
        current_val_loss = logs.get('val_loss')
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = max(old_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"\nEpoch {epoch+1}: Reducing learning rate from {old_lr} to {new_lr}")
                self.wait = 0  # Reset wait counter

    
# -------- MAIN EXECUTION --------
if __name__ == "__main__":
    # Ensure GPU is used if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Using GPU:", gpus)
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Error setting GPU memory growth:", e)
    else:
        print("GPU not found. Using CPU.")


    # Define dataset paths (you can update with 30 paths if needed)
    dataset_paths = [
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\1_20160518.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\2_20150915.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\3_20150919.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\4_20151111.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\5_20160406.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\6_20150507.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\7_20150715.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\8_20151103.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\9_20151028.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\10_20151014.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\11_20150916.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\12_20150725.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\13_20151115.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\14_20151205.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1\15_20150508.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\1_20161125.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\2_20150920.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\3_20151018.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\4_20151118.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\5_20160413.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\6_20150511.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\7_20150717.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\8_20151110.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\9_20151119.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\10_20151021.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\11_20150921.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\12_20150804.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\13_20151125.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\14_20151208.mat",
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\15_20150514.mat",
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\1_20161126.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\2_20151012.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\3_20151101.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\4_20151123.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\5_20160420.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\6_20150512.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\7_20150721.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\8_20151117.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\9_20151209.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\10_20151023.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\11_20151011.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\12_20150807.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\13_20161130.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\14_20151215.mat',
        r'C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\3\15_20150527.mat'
    ]
    session_numbers = [1, 2, 3]
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

    for session in session_numbers:
        print(f"Processing session {session} data...")
        # Call seed_data_pipeline for each session separately. 
        X_train_s, X_test_s, y_train_s, y_test_s = seed_data_pipeline(dataset_paths, session_number=session)
        X_train_list.append(X_train_s)
        X_test_list.append(X_test_s)
        y_train_list.append(y_train_s)
        y_test_list.append(y_test_s)

    # Combine the data from all sessions
    X_train_combined = np.concatenate(X_train_list, axis=0)
    X_test_combined = np.concatenate(X_test_list, axis=0)
    y_train_combined = np.concatenate(y_train_list, axis=0)
    y_test_combined = np.concatenate(y_test_list, axis=0)

    print("Combined Training Data Shape:", X_train_combined.shape)
    print("Combined Testing Data Shape:", X_test_combined.shape)

    # Proceed with training using combined data:
    model = build_eeg_model(input_shape=(SAMPLES_PER_EPOCH, REDUCED_CHANNELS), num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) 
    # Create the callback
    gradual_lr_scheduler = GradualLearningRateScheduler(
        initial_lr=0.001,
        min_lr=0.00001,
        warmup_epochs=5,
        patience=4,
        factor=0.8  # Reduce by 20% each time
    )


    print("Training the combined model for 40 epochs...")
    history = model.fit(X_train_combined, y_train_combined, validation_split=0.2, shuffle=True,
                        epochs=40, batch_size=32, callbacks=[gradual_lr_scheduler, early_stop])
    
    # Save the retrained model
    retrained_model_path = r"C:\sandhyaa\AI-ve\my_ml_project\my_model_trained_4.h5"
    model.save(retrained_model_path)
    print("Model saved as", retrained_model_path)
                        
    plot_history(history, title="Combined Training Process")
    
    # Plot the UMAP-reduced training data
    plot_reduced_dataset(X_train_combined, y_train_combined, title="UMAP Reduced Training Data")
    plot_3d_reduced_dataset(X_train_combined, y_train_combined, title="3D Reduced dataset")
    plot_pairwise(X_train_combined, y_train_combined, title="Pairwise Scatter Matrix")
