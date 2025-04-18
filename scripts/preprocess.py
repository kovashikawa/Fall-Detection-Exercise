import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from logger_config import setup_logger
import random
from sklearn.preprocessing import StandardScaler

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
    'Magnetic Field X (uT)': 'mag_x',
    'Magnetic Field Y (uT)': 'mag_y',
    'Magnetic Field Z (uT)': 'mag_z'
}

def standardize_column_names(df):
    """Standardize column names using the mapping"""
    # Create a new mapping that handles any sensor prefix
    prefix_mapping = {}
    for col in df.columns:
        if col == 'Time':
            prefix_mapping[col] = 'timestamp'
            continue
        for old_name, new_name in COLUMN_MAPPING.items():
            if old_name in col:
                # Extract sensor prefix (e.g., 'r.ankle', 'l.ankle')
                prefix = col.split(' ')[0]
                prefix_mapping[col] = f'{prefix}_{new_name}'
    
    # Rename columns
    df = df.rename(columns=prefix_mapping)
    
    # Add label column based on directory structure
    if 'Falls' in df['trial'].iloc[0]:
        df['label'] = 1
    elif 'Near_Falls' in df['trial'].iloc[0]:
        df['label'] = 0.5  # Intermediate severity for near falls
    else:
        df['label'] = 0
    
    return df

def load_trial(filepath, label):
    """Load and preprocess a single trial file"""
    logger.info(f"Loading trial: {filepath}")
    
    try:
        # Read trial data
        df = pd.read_excel(filepath)
        
        # Add trial information
        df['trial'] = os.path.basename(filepath).replace('.xlsx', '')
        df['label'] = label
        
        # Standardize column names
        df = standardize_column_names(df)
        
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

def load_subject_data(subject_path):
    """Load all trials for a subject"""
    logger.info(f"Loading data for subject: {os.path.basename(subject_path)}")
    all_trials = []
    
    # Load falls
    falls_dir = os.path.join(subject_path, 'Falls')
    if os.path.exists(falls_dir):
        for trial_file in os.listdir(falls_dir):
            if trial_file.endswith('.xlsx'):
                trial_path = os.path.join(falls_dir, trial_file)
                trial_df = load_trial(trial_path, label=1)  # Falls are labeled as 1
                if trial_df is not None:
                    all_trials.append(trial_df)
    
    # Load near falls
    near_falls_dir = os.path.join(subject_path, 'Near_Falls')
    if os.path.exists(near_falls_dir):
        for trial_file in os.listdir(near_falls_dir):
            if trial_file.endswith('.xlsx'):
                trial_path = os.path.join(near_falls_dir, trial_file)
                trial_df = load_trial(trial_path, label=0.5)  # Near falls are labeled as 0.5
                if trial_df is not None:
                    all_trials.append(trial_df)
    
    # Load ADLs
    adls_dir = os.path.join(subject_path, 'ADLs')
    if os.path.exists(adls_dir):
        for trial_file in os.listdir(adls_dir):
            if trial_file.endswith('.xlsx'):
                trial_path = os.path.join(adls_dir, trial_file)
                trial_df = load_trial(trial_path, label=0)  # ADLs are labeled as 0
                if trial_df is not None:
                    all_trials.append(trial_df)
    
    if not all_trials:
        logger.warning(f"No valid trials found for subject {os.path.basename(subject_path)}")
        return None
    
    # Combine all trials
    subject_data = pd.concat(all_trials, ignore_index=True)
    logger.info(f"Loaded {len(all_trials)} trials for subject {os.path.basename(subject_path)}")
    
    return subject_data

def load_all_subjects(data_dir):
    """Load and combine data from all subjects"""
    logger.info("Loading data from all subjects")
    all_data = []
    
    # List all subject directories
    subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subject_dirs:
        logger.warning(f"No subject directories found in {data_dir}. Generating sample data...")
        return generate_sample_data()
    
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
    logger.info(f"Loaded data from {len(subject_dirs)} subjects with {len(all_data)} trials")
    logger.info(f"Total data shape: {combined_data.shape}")
    
    return combined_data

def compute_derived_features(data):
    """Compute derived features from raw sensor data"""
    logger.info("Computing derived features")
    feature_names = []
    
    # Get unique sensor prefixes
    sensor_prefixes = set(col.split('_')[0] for col in data.columns if '_' in col)
    
    for prefix in sensor_prefixes:
        # Get sensor data
        acc_x = data[f'{prefix}_acc_x']
        acc_y = data[f'{prefix}_acc_y']
        acc_z = data[f'{prefix}_acc_z']
        ang_vel_x = data[f'{prefix}_ang_vel_x']
        ang_vel_y = data[f'{prefix}_ang_vel_y']
        ang_vel_z = data[f'{prefix}_ang_vel_z']
        mag_x = data[f'{prefix}_mag_x']
        mag_y = data[f'{prefix}_mag_y']
        mag_z = data[f'{prefix}_mag_z']
        
        # Compute magnitudes
        data[f'{prefix}_acc_mag'] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        data[f'{prefix}_ang_vel_mag'] = np.sqrt(ang_vel_x**2 + ang_vel_y**2 + ang_vel_z**2)
        data[f'{prefix}_mag_mag'] = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        
        # Compute velocities (integrate acceleration)
        data[f'{prefix}_vel_x'] = acc_x.cumsum()
        data[f'{prefix}_vel_y'] = acc_y.cumsum()
        data[f'{prefix}_vel_z'] = acc_z.cumsum()
        
        # Compute jerk (derivative of acceleration)
        data[f'{prefix}_jerk_x'] = acc_x.diff()
        data[f'{prefix}_jerk_y'] = acc_y.diff()
        data[f'{prefix}_jerk_z'] = acc_z.diff()
        
        # Compute energy
        data[f'{prefix}_energy'] = data[f'{prefix}_acc_mag']**2 + data[f'{prefix}_ang_vel_mag']**2
        
        # Compute rolling statistics
        window_size = 10
        for axis in ['x', 'y', 'z']:
            # Acceleration statistics
            data[f'{prefix}_acc_{axis}_mean'] = data[f'{prefix}_acc_{axis}'].rolling(window=window_size).mean()
            data[f'{prefix}_acc_{axis}_std'] = data[f'{prefix}_acc_{axis}'].rolling(window=window_size).std()
            data[f'{prefix}_acc_{axis}_max'] = data[f'{prefix}_acc_{axis}'].rolling(window=window_size).max()
            
            # Angular velocity statistics
            data[f'{prefix}_ang_vel_{axis}_mean'] = data[f'{prefix}_ang_vel_{axis}'].rolling(window=window_size).mean()
            data[f'{prefix}_ang_vel_{axis}_std'] = data[f'{prefix}_ang_vel_{axis}'].rolling(window=window_size).std()
            data[f'{prefix}_ang_vel_{axis}_max'] = data[f'{prefix}_ang_vel_{axis}'].rolling(window=window_size).max()
            
            # Magnetometer statistics
            data[f'{prefix}_mag_{axis}_mean'] = data[f'{prefix}_mag_{axis}'].rolling(window=window_size).mean()
            data[f'{prefix}_mag_{axis}_std'] = data[f'{prefix}_mag_{axis}'].rolling(window=window_size).std()
            data[f'{prefix}_mag_{axis}_max'] = data[f'{prefix}_mag_{axis}'].rolling(window=window_size).max()
        
        # Add feature names
        feature_names.extend([
            f'{prefix}_acc_x', f'{prefix}_acc_y', f'{prefix}_acc_z',
            f'{prefix}_ang_vel_x', f'{prefix}_ang_vel_y', f'{prefix}_ang_vel_z',
            f'{prefix}_mag_x', f'{prefix}_mag_y', f'{prefix}_mag_z',
            f'{prefix}_acc_mag', f'{prefix}_ang_vel_mag', f'{prefix}_mag_mag',
            f'{prefix}_vel_x', f'{prefix}_vel_y', f'{prefix}_vel_z',
            f'{prefix}_jerk_x', f'{prefix}_jerk_y', f'{prefix}_jerk_z',
            f'{prefix}_energy',
            f'{prefix}_acc_x_mean', f'{prefix}_acc_y_mean', f'{prefix}_acc_z_mean',
            f'{prefix}_acc_x_std', f'{prefix}_acc_y_std', f'{prefix}_acc_z_std',
            f'{prefix}_acc_x_max', f'{prefix}_acc_y_max', f'{prefix}_acc_z_max',
            f'{prefix}_ang_vel_x_mean', f'{prefix}_ang_vel_y_mean', f'{prefix}_ang_vel_z_mean',
            f'{prefix}_ang_vel_x_std', f'{prefix}_ang_vel_y_std', f'{prefix}_ang_vel_z_std',
            f'{prefix}_ang_vel_x_max', f'{prefix}_ang_vel_y_max', f'{prefix}_ang_vel_z_max',
            f'{prefix}_mag_x_mean', f'{prefix}_mag_y_mean', f'{prefix}_mag_z_mean',
            f'{prefix}_mag_x_std', f'{prefix}_mag_y_std', f'{prefix}_mag_z_std',
            f'{prefix}_mag_x_max', f'{prefix}_mag_y_max', f'{prefix}_mag_z_max'
        ])
    
    # Fill NaN values
    data = data.ffill().bfill()
    
    logger.info(f"Computed {len(feature_names)} derived features")
    return data, feature_names

def create_sequences(X, y, seq_length=100, stride=50):
    """Create sequences for LSTM input."""
    logger.info(f"Creating sequences with length {seq_length} and stride {stride}")
    sequences, labels = [], []
    
    # Ensure we don't cross trial boundaries
    trial_boundaries = np.where(np.diff(y) != 0)[0] + 1
    trial_boundaries = np.concatenate([[0], trial_boundaries, [len(y)]])
    
    for i in range(len(trial_boundaries) - 1):
        start_idx = trial_boundaries[i]
        end_idx = trial_boundaries[i + 1]
        trial_X = X[start_idx:end_idx]
        trial_y = y[start_idx:end_idx]
        
        # Create sequences within this trial
        for j in range(0, len(trial_X) - seq_length + 1, stride):
            sequences.append(trial_X[j:j+seq_length])
            labels.append(trial_y[j+seq_length-1])
    
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
    
    # Scale features
    logger.info("Scaling features")
    scaler = StandardScaler()
    X = data[feature_names].values
    X_scaled = scaler.fit_transform(X)
    data[feature_names] = X_scaled
    
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
        'input_size': len(feature_names),
        'scaler': scaler
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