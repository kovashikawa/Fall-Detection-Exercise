import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch

def load_trial(filepath, label):
    """Load a single trial Excel file and add label."""
    df = pd.read_excel(filepath)
    df['label'] = label
    return df

def load_subject_data(subject_dir):
    """Load all trials for a single subject."""
    data = []
    for trial_type in ['ADLs', 'Falls', 'Near_Falls']:
        label = 1 if trial_type == 'Falls' else 0
        trial_paths = glob(os.path.join(subject_dir, trial_type, '*.xlsx'))
        for path in trial_paths:
            trial_df = load_trial(path, label)
            data.append(trial_df)
    return pd.concat(data, ignore_index=True)

def load_all_subjects(data_dir):
    """Load data from all subjects."""
    subjects = [f for f in glob(os.path.join(data_dir, 'sub*')) if os.path.isdir(f)]
    all_data = [load_subject_data(sub) for sub in subjects]
    return pd.concat(all_data, ignore_index=True)

def compute_sensor_magnitudes(data):
    """Compute magnitude features for each sensor."""
    sensor_locations = ['r.ankle', 'l.ankle', 'r.thigh', 'l.thigh', 'head', 'sternum', 'waist']
    for sensor in sensor_locations:
        data[f'{sensor}_acc_mag'] = np.sqrt(
            data[f'{sensor} Acceleration X (m/s^2)']**2 +
            data[f'{sensor} Acceleration Y (m/s^2)']**2 +
            data[f'{sensor} Acceleration Z (m/s^2)']**2
        )
    return data

def create_sequences(X, y, seq_length=100):
    """Create sequences for LSTM input."""
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length-1])
    return torch.tensor(sequences).float(), torch.tensor(labels).float()

def preprocess_data(data_dir, seq_length=100):
    """Main preprocessing function."""
    # Load data
    data = load_all_subjects(data_dir)
    
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    
    # Feature engineering
    data = compute_sensor_magnitudes(data)
    
    # Select features
    features = [col for col in data.columns if 'Acceleration' in col or 'Angular Velocity' in col or '_acc_mag' in col]
    X = data[features].values
    y = data['label'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    
    return {
        'train': (X_train_seq, y_train_seq),
        'val': (X_val_seq, y_val_seq),
        'test': (X_test_seq, y_test_seq),
        'feature_names': features
    }

if __name__ == '__main__':
    # Example usage
    data_dir = 'data'  # Update this path
    processed_data = preprocess_data(data_dir)
    print(f"Training data shape: {processed_data['train'][0].shape}")
    print(f"Validation data shape: {processed_data['val'][0].shape}")
    print(f"Test data shape: {processed_data['test'][0].shape}") 