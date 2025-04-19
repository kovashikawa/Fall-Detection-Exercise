import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from logger_config import setup_logger
import random
from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path
import logging

# Setup logger
logger = setup_logger('preprocess', 'preprocessing')

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define column mapping for standardization
COLUMN_MAPPING = {
    # Right ankle
    'r.ankle Acceleration X (m/s^2)': 'r_ankle_acc_x',
    'r.ankle Acceleration Y (m/s^2)': 'r_ankle_acc_y',
    'r.ankle Acceleration Z (m/s^2)': 'r_ankle_acc_z',
    'r.ankle Angular Velocity X (rad/s)': 'r_ankle_gyro_x',
    'r.ankle Angular Velocity Y (rad/s)': 'r_ankle_gyro_y',
    'r.ankle Angular Velocity Z (rad/s)': 'r_ankle_gyro_z',
    'r.ankle Magnetic Field X (uT)': 'r_ankle_mag_x',
    'r.ankle Magnetic Field Y (uT)': 'r_ankle_mag_y',
    'r.ankle Magnetic Field Z (uT)': 'r_ankle_mag_z',
    
    # Left ankle
    'l.ankle Acceleration X (m/s^2)': 'l_ankle_acc_x',
    'l.ankle Acceleration Y (m/s^2)': 'l_ankle_acc_y',
    'l.ankle Acceleration Z (m/s^2)': 'l_ankle_acc_z',
    'l.ankle Angular Velocity X (rad/s)': 'l_ankle_gyro_x',
    'l.ankle Angular Velocity Y (rad/s)': 'l_ankle_gyro_y',
    'l.ankle Angular Velocity Z (rad/s)': 'l_ankle_gyro_z',
    'l.ankle Magnetic Field X (uT)': 'l_ankle_mag_x',
    'l.ankle Magnetic Field Y (uT)': 'l_ankle_mag_y',
    'l.ankle Magnetic Field Z (uT)': 'l_ankle_mag_z',
    
    # Right thigh
    'r.thigh Acceleration X (m/s^2)': 'r_thigh_acc_x',
    'r.thigh Acceleration Y (m/s^2)': 'r_thigh_acc_y',
    'r.thigh Acceleration Z (m/s^2)': 'r_thigh_acc_z',
    'r.thigh Angular Velocity X (rad/s)': 'r_thigh_gyro_x',
    'r.thigh Angular Velocity Y (rad/s)': 'r_thigh_gyro_y',
    'r.thigh Angular Velocity Z (rad/s)': 'r_thigh_gyro_z',
    'r.thigh Magnetic Field X (uT)': 'r_thigh_mag_x',
    'r.thigh Magnetic Field Y (uT)': 'r_thigh_mag_y',
    'r.thigh Magnetic Field Z (uT)': 'r_thigh_mag_z',
    
    # Left thigh
    'l.thigh Acceleration X (m/s^2)': 'l_thigh_acc_x',
    'l.thigh Acceleration Y (m/s^2)': 'l_thigh_acc_y',
    'l.thigh Acceleration Z (m/s^2)': 'l_thigh_acc_z',
    'l.thigh Angular Velocity X (rad/s)': 'l_thigh_gyro_x',
    'l.thigh Angular Velocity Y (rad/s)': 'l_thigh_gyro_y',
    'l.thigh Angular Velocity Z (rad/s)': 'l_thigh_gyro_z',
    'l.thigh Magnetic Field X (uT)': 'l_thigh_mag_x',
    'l.thigh Magnetic Field Y (uT)': 'l_thigh_mag_y',
    'l.thigh Magnetic Field Z (uT)': 'l_thigh_mag_z',
    
    # Head
    'head Acceleration X (m/s^2)': 'head_acc_x',
    'head Acceleration Y (m/s^2)': 'head_acc_y',
    'head Acceleration Z (m/s^2)': 'head_acc_z',
    'head Angular Velocity X (rad/s)': 'head_gyro_x',
    'head Angular Velocity Y (rad/s)': 'head_gyro_y',
    'head Angular Velocity Z (rad/s)': 'head_gyro_z',
    'head Magnetic Field X (uT)': 'head_mag_x',
    'head Magnetic Field Y (uT)': 'head_mag_y',
    'head Magnetic Field Z (uT)': 'head_mag_z',
    
    # Sternum
    'sternum Acceleration X (m/s^2)': 'sternum_acc_x',
    'sternum Acceleration Y (m/s^2)': 'sternum_acc_y',
    'sternum Acceleration Z (m/s^2)': 'sternum_acc_z',
    'sternum Angular Velocity X (rad/s)': 'sternum_gyro_x',
    'sternum Angular Velocity Y (rad/s)': 'sternum_gyro_y',
    'sternum Angular Velocity Z (rad/s)': 'sternum_gyro_z',
    'sternum Magnetic Field X (uT)': 'sternum_mag_x',
    'sternum Magnetic Field Y (uT)': 'sternum_mag_y',
    'sternum Magnetic Field Z (uT)': 'sternum_mag_z',
    
    # Waist
    'waist Acceleration X (m/s^2)': 'waist_acc_x',
    'waist Acceleration Y (m/s^2)': 'waist_acc_y',
    'waist Acceleration Z (m/s^2)': 'waist_acc_z',
    'waist Angular Velocity X (rad/s)': 'waist_gyro_x',
    'waist Angular Velocity Y (rad/s)': 'waist_gyro_y',
    'waist Angular Velocity Z (rad/s)': 'waist_gyro_z',
    'waist Magnetic Field X (uT)': 'waist_mag_x',
    'waist Magnetic Field Y (uT)': 'waist_mag_y',
    'waist Magnetic Field Z (uT)': 'waist_mag_z',
    
    # Time column
    'Time': 'time'
}

def standardize_column_names(df):
    """Standardize column names using the mapping."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df.rename(columns=COLUMN_MAPPING)

def load_trial(filepath, label):
    """Load a single trial from an Excel file and standardize column names."""
    try:
        df = pd.read_excel(filepath)
        df = standardize_column_names(df)
        df['label'] = label
        df['trial'] = filepath.stem
        return df
    except Exception as e:
        logger.error(f"Error loading trial {filepath}: {str(e)}")
        return None

def load_subject_data(subject_dir):
    """Load all trials for a subject."""
    subject_dir = Path(subject_dir)
    falls_dir = subject_dir / 'Falls'
    adls_dir = subject_dir / 'ADLs'
    
    all_trials = []
    
    # Load fall trials
    if falls_dir.exists():
        for trial_file in falls_dir.glob('*.xlsx'):
            df = load_trial(trial_file, label=1)
            if df is not None:
                all_trials.append(df)
    
    # Load ADL trials
    if adls_dir.exists():
        for trial_file in adls_dir.glob('*.xlsx'):
            df = load_trial(trial_file, label=0)
            if df is not None:
                all_trials.append(df)
    
    if not all_trials:
        logger.warning(f"No data found in {subject_dir}")
        return None
    
    return pd.concat(all_trials, ignore_index=True)

def parse_args():
    p = argparse.ArgumentParser(description="Fall-Detection preprocessing")
    p.add_argument('--sample_frac', type=float, default=1.0,
                   help="If <1.0, only keep this fraction of rows after loading for quick debug")
    p.add_argument('--data_dir', type=str, default='data',
                   help="Directory containing the data files")
    return p.parse_args()

def load_all_subjects(data_dir, sample_frac=1.0):
    """Load and combine data from all subjects"""
    logger.info("Loading data from all subjects")
    all_data = []
    
    # List all subject directories
    subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subject_dirs:
        logger.warning(f"No subject directories found in {data_dir}. Generating sample data...")
        return generate_sample_data()
    
    # If sample_frac < 1.0, only process a subset of subjects
    if sample_frac < 1.0:
        n_keep = max(1, int(len(subject_dirs) * sample_frac))
        subject_dirs = subject_dirs[:n_keep]
        logger.info(f"[DEBUG] Only processing {n_keep} subjects for quick turn-around")
    
    for subject_dir in subject_dirs:
        subject_path = os.path.join(data_dir, subject_dir)
        subject_data = load_subject_data(subject_path)
        if subject_data is not None:
            all_data.append(subject_data)
    
    if not all_data:
        logger.warning("No data was loaded. Generating sample data...")
        return generate_sample_data()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # If sample_frac < 1.0, further subsample the combined data
    if sample_frac < 1.0:
        combined_data = combined_data.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        logger.info(f"[DEBUG] Sampled down to {len(combined_data)} rows ({sample_frac*100:.1f}% of original)")
    
    logger.info(f"Loaded data from {len(subject_dirs)} subjects with {len(all_data)} trials")
    logger.info(f"Total data shape: {combined_data.shape}")
    
    return combined_data

def compute_derived_features(data):
    """Compute derived features from sensor data."""
    feature_data = data.copy()
    
    # List of sensors
    sensors = ['r_ankle', 'l_ankle', 'r_thigh', 'l_thigh', 'head', 'sternum', 'waist']
    
    for sensor in sensors:
        # Compute acceleration magnitude
        acc_cols = [f'{sensor}_acc_x', f'{sensor}_acc_y', f'{sensor}_acc_z']
        feature_data[f'{sensor}_acc_mag'] = np.sqrt(
            feature_data[acc_cols].pow(2).sum(axis=1)
        )
        
        # Compute angular velocity magnitude
        gyro_cols = [f'{sensor}_gyro_x', f'{sensor}_gyro_y', f'{sensor}_gyro_z']
        feature_data[f'{sensor}_gyro_mag'] = np.sqrt(
            feature_data[gyro_cols].pow(2).sum(axis=1)
        )
        
        # Compute magnetic field magnitude
        mag_cols = [f'{sensor}_mag_x', f'{sensor}_mag_y', f'{sensor}_mag_z']
        feature_data[f'{sensor}_mag_mag'] = np.sqrt(
            feature_data[mag_cols].pow(2).sum(axis=1)
        )
    
    return feature_data

def create_sequences_generator(X, y, seq_length=100, stride=1):
    """Create sequences using a generator to save memory."""
    for trial in X['trial'].unique():
        trial_data = X[X['trial'] == trial].copy()
        trial_labels = y[X['trial'] == trial].copy()
        
        # Drop trial column before creating sequences
        trial_data = trial_data.drop(columns=['trial'])
        
        # Process in smaller chunks
        chunk_size = 500
        for i in range(0, len(trial_data), chunk_size):
            chunk_data = trial_data.iloc[i:i + chunk_size].copy()
            chunk_labels = trial_labels.iloc[i:i + chunk_size].copy()
            
            for j in range(0, len(chunk_data) - seq_length + 1, stride):
                seq = chunk_data.iloc[j:j + seq_length].values
                # Use the majority label in the sequence
                label = chunk_labels.iloc[j:j + seq_length].mode()[0]
                yield seq, label
            
            # Clear memory
            del chunk_data
            del chunk_labels
        
        # Clear memory
        del trial_data
        del trial_labels

def collect_sequences(generator, max_sequences=10000):
    """Collect sequences from generator up to a maximum count."""
    sequences = []
    labels = []
    count = 0
    
    for seq, label in generator:
        sequences.append(seq)
        labels.append(label)
        count += 1
        
        if count >= max_sequences:
            break
    
    return np.array(sequences), np.array(labels)

def preprocess_data(data_dir, sample_frac=0.1, max_sequences=10000):
    """Preprocess data for training."""
    logger.info("Starting data preprocessing")
    
    # Load data
    logger.info("Loading data")
    data = load_all_subjects(data_dir, sample_frac)
    
    # Standardize column names
    logger.info("Standardizing column names")
    data = standardize_column_names(data)
    
    # Compute derived features
    data = compute_derived_features(data)
    
    # Select only numeric columns for scaling
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['label']]
    
    # Scale features in chunks
    logger.info("Scaling features")
    scaler = StandardScaler()
    chunk_size = 5000
    
    # First pass to fit the scaler
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size].copy()
        X = chunk[feature_cols].values
        scaler.partial_fit(X)
        del chunk, X
    
    # Second pass to transform the data
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size].copy()
        X = chunk[feature_cols].values
        X_scaled = scaler.transform(X)
        data.iloc[i:i + chunk_size, data.columns.get_indexer(feature_cols)] = X_scaled
        del chunk, X, X_scaled
    
    # Select features and labels
    logger.info("Selecting features and labels")
    X = data[feature_cols + ['trial']].copy()
    y = data['label'].copy()
    logger.info(f"Selected {len(feature_cols)} features")
    
    # Clear original data
    del data
    
    # Split data
    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Clear memory
    del X
    del y
    
    # Create sequences using generators
    logger.info("Creating sequences")
    train_gen = create_sequences_generator(X_train, y_train)
    val_gen = create_sequences_generator(X_val, y_val)
    test_gen = create_sequences_generator(X_test, y_test)
    
    X_train_seq, y_train_seq = collect_sequences(train_gen, max_sequences)
    X_val_seq, y_val_seq = collect_sequences(val_gen, max_sequences // 5)
    X_test_seq, y_test_seq = collect_sequences(test_gen, max_sequences // 5)
    
    # Clear memory
    del X_train, y_train, X_val, y_val, X_test, y_test
    
    # Convert to tensors
    logger.info("Converting to tensors")
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.FloatTensor(y_val_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)
    
    # Clear memory
    del X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq
    
    return {
        'train': (X_train_tensor, y_train_tensor),
        'val': (X_val_tensor, y_val_tensor),
        'test': (X_test_tensor, y_test_tensor),
        'feature_names': feature_cols,
        'input_size': len(feature_cols),
        'scaler': scaler
    }

if __name__ == '__main__':
    try:
        args = parse_args()
        logger.info(f"Starting preprocessing with data directory: {args.data_dir}")
        logger.info(f"Sample fraction: {args.sample_frac}")
        
        processed_data = preprocess_data(args.data_dir, sample_frac=args.sample_frac)
        logger.info(f"Training data shape: {processed_data['train'][0].shape}")
        logger.info(f"Validation data shape: {processed_data['val'][0].shape}")
        logger.info(f"Test data shape: {processed_data['test'][0].shape}")
        logger.info(f"Number of features: {processed_data['input_size']}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise 