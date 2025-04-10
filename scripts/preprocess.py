import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from logger_config import setup_logger

# Setup logger
logger = setup_logger('preprocess', 'preprocessing')

def load_trial(filepath, label):
    """Load a single trial Excel file and add label."""
    logger.debug(f"Loading trial from {filepath}")
    df = pd.read_excel(filepath)
    df['label'] = label
    return df

def load_subject_data(subject_dir):
    """Load all trials for a single subject."""
    logger.info(f"Loading data for subject: {os.path.basename(subject_dir)}")
    data = []
    for trial_type in ['ADLs', 'Falls', 'Near_Falls']:
        label = 1 if trial_type == 'Falls' else 0
        trial_paths = glob(os.path.join(subject_dir, trial_type, '*.xlsx'))
        logger.debug(f"Found {len(trial_paths)} {trial_type} trials")
        for path in trial_paths:
            trial_df = load_trial(path, label)
            data.append(trial_df)
    return pd.concat(data, ignore_index=True)

def load_all_subjects(data_dir):
    """Load data from all subjects."""
    logger.info("Loading data from all subjects")
    subjects = [f for f in glob(os.path.join(data_dir, 'sub*')) if os.path.isdir(f)]
    logger.info(f"Found {len(subjects)} subjects")
    all_data = [load_subject_data(sub) for sub in subjects]
    return pd.concat(all_data, ignore_index=True)

def compute_sensor_magnitudes(data):
    """Compute magnitude features for each sensor."""
    logger.info("Computing sensor magnitudes")
    sensor_locations = ['r.ankle', 'l.ankle', 'r.thigh', 'l.thigh', 'head', 'sternum', 'waist']
    for sensor in sensor_locations:
        logger.debug(f"Computing magnitude for {sensor}")
        data[f'{sensor}_acc_mag'] = np.sqrt(
            data[f'{sensor} Acceleration X (m/s^2)']**2 +
            data[f'{sensor} Acceleration Y (m/s^2)']**2 +
            data[f'{sensor} Acceleration Z (m/s^2)']**2
        )
    return data

def create_sequences(X, y, seq_length=100):
    """Create sequences for LSTM input."""
    logger.info(f"Creating sequences with length {seq_length}")
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length-1])
    logger.info(f"Created {len(sequences)} sequences")
    return torch.tensor(sequences).float(), torch.tensor(labels).float()

def preprocess_data(data_dir, seq_length=100):
    """Main preprocessing function."""
    logger.info("Starting data preprocessing")
    
    # Load data
    logger.info("Loading data")
    data = load_all_subjects(data_dir)
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Handle missing values
    logger.info("Handling missing values")
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    logger.info(f"Data shape after handling missing values: {data.shape}")
    
    # Feature engineering
    logger.info("Performing feature engineering")
    data = compute_sensor_magnitudes(data)
    
    # Select features
    logger.info("Selecting features")
    features = [col for col in data.columns if 'Acceleration' in col or 'Angular Velocity' in col or '_acc_mag' in col]
    X = data[features].values
    y = data['label'].values
    logger.info(f"Selected {len(features)} features")
    
    # Split data
    logger.info("Splitting data into train/val/test sets")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Create sequences
    logger.info("Creating sequences for LSTM input")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    
    logger.info("Preprocessing completed successfully")
    
    return {
        'train': (X_train_seq, y_train_seq),
        'val': (X_val_seq, y_val_seq),
        'test': (X_test_seq, y_test_seq),
        'feature_names': features
    }

if __name__ == '__main__':
    try:
        # Example usage
        data_dir = 'data'
        logger.info(f"Starting preprocessing with data directory: {data_dir}")
        processed_data = preprocess_data(data_dir)
        logger.info(f"Training data shape: {processed_data['train'][0].shape}")
        logger.info(f"Validation data shape: {processed_data['val'][0].shape}")
        logger.info(f"Test data shape: {processed_data['test'][0].shape}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise 