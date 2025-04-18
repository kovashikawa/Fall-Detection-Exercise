import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from logger_config import setup_logger
import random

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

# Column name mapping for standardization
COLUMN_MAPPING = {
    # Acceleration columns
    'Acceleration X (m/s^2)': 'acc_x',
    'Acceleration Y (m/s^2)': 'acc_y',
    'Acceleration Z (m/s^2)': 'acc_z',
    # Angular velocity columns
    'Angular Velocity X (rad/s)': 'ang_vel_x',
    'Angular Velocity Y (rad/s)': 'ang_vel_y',
    'Angular Velocity Z (rad/s)': 'ang_vel_z',
    # Magnetometer columns
    'Magnetic Field X (μT)': 'mag_x',
    'Magnetic Field Y (μT)': 'mag_y',
    'Magnetic Field Z (μT)': 'mag_z'
}

def standardize_column_names(df):
    """Standardize column names using the mapping"""
    return df.rename(columns=COLUMN_MAPPING)

def load_trial(filepath, label):
    """Load a single trial Excel file and add label."""
    logger.debug(f"Loading trial from {filepath}")
    df = pd.read_excel(filepath)
    df = standardize_column_names(df)
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

def compute_derived_features(data):
    """Compute derived features from raw sensor data"""
    logger.info("Computing derived features")
    
    # Basic features
    features = []
    
    # For each sensor type (acc, gyro, mag)
    for sensor in ['acc', 'ang_vel', 'mag']:
        # Get the three components
        x = data[f'{sensor}_x']
        y = data[f'{sensor}_y']
        z = data[f'{sensor}_z']
        
        # Compute magnitude
        mag = np.sqrt(x**2 + y**2 + z**2)
        features.append(mag)
        
        # Compute velocity (integral of acceleration)
        if sensor == 'acc':
            vel_x = np.cumsum(x)
            vel_y = np.cumsum(y)
            vel_z = np.cumsum(z)
            features.extend([vel_x, vel_y, vel_z])
            
            # Compute jerk (derivative of acceleration)
            jerk_x = np.gradient(x)
            jerk_y = np.gradient(y)
            jerk_z = np.gradient(z)
            features.extend([jerk_x, jerk_y, jerk_z])
            
            # Compute energy
            energy = x**2 + y**2 + z**2
            features.append(energy)
        
        # Compute statistical features
        window_size = 10
        for component in [x, y, z, mag]:
            # Rolling mean
            mean = component.rolling(window=window_size).mean()
            features.append(mean)
            
            # Rolling std
            std = component.rolling(window=window_size).std()
            features.append(std)
            
            # Rolling max
            max_val = component.rolling(window=window_size).max()
            features.append(max_val)
    
    # Combine all features
    feature_names = []
    for i, feature in enumerate(features):
        feature_names.append(f'feature_{i}')
        data[f'feature_{i}'] = feature
    
    logger.info(f"Computed {len(feature_names)} derived features")
    return data, feature_names

def create_sequences(X, y, seq_length=100, stride=50):
    """Create sequences for LSTM input."""
    logger.info(f"Creating sequences with length {seq_length} and stride {stride}")
    sequences, labels = [], []
    
    for i in range(0, len(X) - seq_length + 1, stride):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length-1])
    
    # Convert lists to numpy arrays before creating tensors
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    logger.info(f"Created {len(sequences)} sequences")
    return sequences, labels

def preprocess_data(data_dir, seq_length=100, stride=50):
    """Main preprocessing function."""
    logger.info("Starting data preprocessing")
    
    # Set random seeds
    set_seeds()
    
    # Load data
    logger.info("Loading data")
    data = load_all_subjects(data_dir)
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Handle missing values
    logger.info("Handling missing values")
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    logger.info(f"Data shape after handling missing values: {data.shape}")
    
    # Compute derived features
    data, feature_names = compute_derived_features(data)
    
    # Select features and labels
    logger.info("Selecting features and labels")
    X = data[feature_names].values
    y = data['label'].values
    logger.info(f"Selected {len(feature_names)} features")
    
    # Split data
    logger.info("Splitting data into train/val/test sets")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Create sequences
    logger.info("Creating sequences for LSTM input")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length, stride)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length, stride)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length, stride)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.FloatTensor(y_val_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)
    
    logger.info("Preprocessing completed successfully")
    
    return {
        'train': (X_train_tensor, y_train_tensor),
        'val': (X_val_tensor, y_val_tensor),
        'test': (X_test_tensor, y_test_tensor),
        'feature_names': feature_names,
        'input_size': len(feature_names)
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
        logger.info(f"Number of features: {processed_data['input_size']}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise 