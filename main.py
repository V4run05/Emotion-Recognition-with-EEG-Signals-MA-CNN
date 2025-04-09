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
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import warnings
from umap.umap_ import UMAP  # Ensure umap-learn is installed
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Directory containing .mat files for SEED-IV (session 1)
DATASET_DIR = r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\1"

# Regex pattern to match EEG trial keys (e.g., "ha_eeg1", "cz_eeg24")
EEG_KEY_PATTERN = re.compile(r".+_eeg\d+$")
# Session 1 labels from ReadMe.txt
SESSION1_LABEL = np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3])

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

def create_ensemble(X_train, y_train, X_test, num_models=3):
    models = []
    for i in range(num_models):
        model = build_eeg_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        models.append(model)
    
    # Make predictions with each model
    predictions = [model.predict(X_test) for model in models]
    # Average the predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# -------- STEP 1: LOAD DATASET --------
def load_seed_dataset(mat_path):
    """
    Load EEG data and labels from a .mat file for SEED-IV session 1.
    It detects trial keys, truncates each trial to the same length,
    splits into epochs, and assigns labels.
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
    
    if len(SESSION1_LABEL) != len(sorted_keys):
        raise ValueError(f"Mismatch: Found {len(sorted_keys)} trials but {len(SESSION1_LABEL)} labels in {mat_path}")
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
    x = tf.keras.layers.Conv2D(128, (10, 1), padding='same', activation='elu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Spatial convolutions (across channels)
    x = tf.keras.layers.Conv2D(128, (1, REDUCED_CHANNELS), padding='valid', activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Further temporal processing
    x = tf.keras.layers.Conv2D(256, (10, 1), padding='same', activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Classification layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='elu', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Use a learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch > 20:
        return initial_lr * 0.1
    return initial_lr

# -------- STEP 4: DATA PIPELINE --------
def seed_data_pipeline(mat_paths):
    X_data, y_labels = [], []
    for path in mat_paths:
        try:
            eeg_data_raw, labels_raw = load_seed_dataset(path)
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
    We average over the time dimension so that each sample is represented by a vector.
    If the number of UMAP dimensions is greater than 2, apply t-SNE to reduce to 2D.
    y is expected to be one-hot encoded.
    """
    # Average over the time dimension.
    X_mean = np.mean(X.squeeze(-1), axis=1)  # shape: (samples, REDUCED_CHANNELS)
    
    # If the data has more than 2 dimensions, use t-SNE to reduce it to 2D.
    if X_mean.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        X_plot = tsne.fit_transform(X_mean)
    else:
        X_plot = X_mean
    
    labels = np.argmax(y, axis=1)
    
    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(X_plot[idx, 0], X_plot[idx, 1], label=f"Class {c}", alpha=0.6)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
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
        r"C:\sandhyaa\AI-ve\my_ml_project\seed_iv\eeg_raw_data\2\15_20150514.mat"
    ]
    X_train, X_test, y_train, y_test = seed_data_pipeline(dataset_paths)
    print("Initial Training Data Shape:", X_train.shape)
    print("Initial Testing Data Shape:", X_test.shape)
    
    X_train_aug, y_train_aug = augment_eeg(X_train, y_train)
    X_train = np.concatenate([X_train, X_train_aug], axis=0)
    y_train = np.concatenate([y_train, y_train_aug], axis=0)
    print("Augmented Training Data Shape:", X_train.shape)
    
    # Creating new model    
    print("Building a new model.")
    model = build_eeg_model(input_shape=(SAMPLES_PER_EPOCH, REDUCED_CHANNELS), num_classes=NUM_CLASSES)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    print("Training the model for 40 epochs...")
    history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
                        epochs=40, batch_size=32, callbacks=[lr_scheduler, early_stop])
    
    plot_history(history, title="Training Process")
    
    # Plot the UMAP-reduced training data
    plot_reduced_dataset(X_train, y_train, title="UMAP Reduced Training Data")
    plot_3d_reduced_dataset(X_train, y_train, title="3D Reduced dataset")
    plot_pairwise(X_train, y_train, title="Pairwise Scatter Matrix")
    
    #KFold cross-validation code
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold+1}/5...")
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = build_eeg_model()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(X_fold_train, y_fold_train, 
                            validation_data=(X_fold_val, y_fold_val),
                            epochs=50, batch_size=32, 
                            callbacks=[early_stop, lr_scheduler])
        
        fold_accuracies.append(max(history.history['val_accuracy']))
        
    print(f"Average validation accuracy: {np.mean(fold_accuracies):.4f}")
    
    # Save the retrained model
    retrained_model_path = r"C:\sandhyaa\AI-ve\my_ml_project\my_model_trained_4.h5"
    model.save(retrained_model_path)
    print("Model saved as", retrained_model_path)
