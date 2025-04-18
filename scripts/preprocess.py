import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from logger_config import setup_logger

# Setup logger
logger = setup_logger('preprocess', 'preprocessing')

def load_all_subjects(data_dir):
    """Load and combine data from all subjects with proper activity labels"""
    logger.info("Loading data from all subjects")
    all_data = []
    
    # List all subject directories
    subject_dirs = [d for d in os.listdir(data_dir) if d.startswith('sub') and os.path.isdir(os.path.join(data_dir, d))]
    
    if not subject_dirs:
        logger.error(f"No subject directories found in {data_dir}")
        raise ValueError("No data found. Please ensure the data directory contains subject folders (sub1, sub2, etc.)")
    
    for subject_dir in subject_dirs:
        subject_path = os.path.join(data_dir, subject_dir)
        logger.info(f"Processing subject: {subject_dir}")
        
        # Process each activity type (Falls, Near_Falls, ADLs)
        activity_types = ['Falls', 'Near_Falls', 'ADLs']
        
        for activity_type in activity_types:
            activity_path = os.path.join(subject_path, activity_type)
            
            if not os.path.exists(activity_path):
                logger.warning(f"Activity directory not found: {activity_path}")
                continue
            
            # List all Excel files in the activity directory
            trial_files = [f for f in os.listdir(activity_path) if f.endswith('.xlsx')]
            
            if not trial_files:
                logger.warning(f"No Excel files found in {activity_path}")
                continue
            
            for trial_file in trial_files:
                trial_path = os.path.join(activity_path, trial_file)
                logger.info(f"Loading trial: {trial_file} from {activity_type}")
                
                try:
                    # Read Excel data
                    trial_data = pd.read_excel(trial_path)
                    
                    # Add metadata
                    trial_data['subject'] = subject_dir
                    trial_data['trial'] = trial_file.replace('.xlsx', '')
                    trial_data['activity_type'] = activity_type
                    
                    # Extract fall type from filename (e.g., 'slip', 'trip', etc.)
                    fall_type = trial_file.split('_')[1].lower()
                    trial_data['fall_type'] = fall_type
                    
                    # Set binary label (1 for Falls, 0 for others)
                    trial_data['label'] = 1 if activity_type == 'Falls' else 0
                    
                    all_data.append(trial_data)
                except Exception as e:
                    logger.error(f"Error loading {trial_path}: {str(e)}")
                    continue
    
    if not all_data:
        logger.error("No data was loaded from any subject")
        raise ValueError("Failed to load any data. Please check the data directory structure and file formats.")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Successfully loaded data from {len(subject_dirs)} subjects")
    logger.info(f"Total data shape: {combined_data.shape}")
    logger.info(f"Activity distribution:")
    logger.info(combined_data.groupby('activity_type').size())
    logger.info(f"Fall type distribution:")
    logger.info(combined_data.groupby('fall_type').size())
    logger.info(f"Label distribution:")
    logger.info(combined_data['label'].value_counts())
    
    return combined_data

def compute_derived_features(data):
    """Compute derived features from raw sensor data"""
    logger.info("Computing derived features")
    
    # Get all sensor columns
    acc_cols = [col for col in data.columns if 'Acceleration' in col]
    gyro_cols = [col for col in data.columns if 'Angular Velocity' in col]
    
    # Compute magnitude for each sensor
    for i in range(0, len(acc_cols), 3):
        sensor_name = acc_cols[i].split('_')[0]  # e.g., 'Ankle', 'Thigh', etc.
        x, y, z = acc_cols[i], acc_cols[i+1], acc_cols[i+2]
        data[f'{sensor_name}_acc_mag'] = np.sqrt(data[x]**2 + data[y]**2 + data[z]**2)
    
    for i in range(0, len(gyro_cols), 3):
        sensor_name = gyro_cols[i].split('_')[0]
        x, y, z = gyro_cols[i], gyro_cols[i+1], gyro_cols[i+2]
        data[f'{sensor_name}_gyro_mag'] = np.sqrt(data[x]**2 + data[y]**2 + data[z]**2)
    
    # Compute jerk (derivative of acceleration)
    for i in range(0, len(acc_cols), 3):
        sensor_name = acc_cols[i].split('_')[0]
        x, y, z = acc_cols[i], acc_cols[i+1], acc_cols[i+2]
        data[f'{sensor_name}_jerk_x'] = data[x].diff()
        data[f'{sensor_name}_jerk_y'] = data[y].diff()
        data[f'{sensor_name}_jerk_z'] = data[z].diff()
        data[f'{sensor_name}_jerk_mag'] = np.sqrt(
            data[f'{sensor_name}_jerk_x']**2 + 
            data[f'{sensor_name}_jerk_y']**2 + 
            data[f'{sensor_name}_jerk_z']**2
        )
    
    # Fill NaN values (from diff operation)
    data = data.fillna(0)
    
    return data

def create_sequences_per_trial(data, window_size=100, stride=50):
    """Create sequences for each trial separately"""
    logger.info(f"Creating sequences with window_size={window_size}, stride={stride}")
    
    all_sequences = []
    all_labels = []
    all_severity = []
    
    # Group by subject and trial
    grouped = data.groupby(['subject', 'trial'])
    
    for (subject, trial), group in grouped:
        logger.info(f"Processing {subject}/{trial}")
        
        # Get feature columns (excluding metadata and label)
        feature_cols = [col for col in group.columns 
                       if col not in ['subject', 'trial', 'activity_type', 
                                    'fall_type', 'label', 'Time']]
        
        # Scale features for this trial
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(group[feature_cols])
        
        # Create sequences for this trial
        for i in range(0, len(group) - window_size + 1, stride):
            sequence = scaled_features[i:i + window_size]
            label = group['label'].iloc[i + window_size - 1]
            
            # Compute severity score for this window
            window_data = group.iloc[i:i + window_size]
            severity = compute_severity_score(window_data)
            
            all_sequences.append(sequence)
            all_labels.append(label)
            all_severity.append(severity)
    
    return np.array(all_sequences), np.array(all_labels), np.array(all_severity)

def compute_severity_score(data):
    """Compute fall severity score based on acceleration and impact features"""
    # Get acceleration columns for all sensors
    acc_cols = [col for col in data.columns if 'Acceleration' in col]
    
    # Calculate total acceleration magnitude for each sensor
    acc_magnitudes = []
    for i in range(0, len(acc_cols), 3):  # Process X, Y, Z components together
        acc_x = data[acc_cols[i]].astype(float)
        acc_y = data[acc_cols[i+1]].astype(float)
        acc_z = data[acc_cols[i+2]].astype(float)
        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        acc_magnitudes.append(magnitude)
    
    # Combine magnitudes from all sensors
    total_acc_magnitude = np.mean(acc_magnitudes, axis=0)
    
    # Calculate impact force (using maximum acceleration magnitude)
    impact_force = np.max(total_acc_magnitude)
    
    # Calculate duration of high acceleration
    high_acc_threshold = np.percentile(total_acc_magnitude, 95)
    duration = np.sum(total_acc_magnitude > high_acc_threshold)
    
    # Get angular velocity columns for all sensors
    gyro_cols = [col for col in data.columns if 'Angular Velocity' in col]
    
    # Calculate total angular velocity magnitude for each sensor
    gyro_magnitudes = []
    for i in range(0, len(gyro_cols), 3):  # Process X, Y, Z components together
        gyro_x = data[gyro_cols[i]].astype(float)
        gyro_y = data[gyro_cols[i+1]].astype(float)
        gyro_z = data[gyro_cols[i+2]].astype(float)
        magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        gyro_magnitudes.append(magnitude)
    
    # Combine magnitudes from all sensors
    total_gyro_magnitude = np.mean(gyro_magnitudes, axis=0)
    max_angular_velocity = np.max(total_gyro_magnitude)
    
    # Combine into severity score (normalized between 0 and 1)
    severity = (0.5 * impact_force + 0.3 * duration + 0.2 * max_angular_velocity) / \
              (np.max([impact_force, 1]) * len(data))
    
    return float(severity)

def preprocess_data(data_dir='data', split='train', window_size=100, stride=50):
    """Main preprocessing function"""
    try:
        # Load data
        data = load_all_subjects(data_dir)
        
        # Compute derived features
        data = compute_derived_features(data)
        
        # Create sequences per trial
        X, y, severity = create_sequences_per_trial(data, window_size, stride)
        
        # Split data by subject (to avoid data leakage)
        subjects = data['subject'].unique()
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=0.2, random_state=42
        )
        
        # Get indices for train/test split
        train_mask = data['subject'].isin(train_subjects)
        test_mask = data['subject'].isin(test_subjects)
        
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        if split == 'train':
            X_train, y_train, severity_train = create_sequences_per_trial(
                train_data, window_size, stride
            )
            return X_train, y_train, severity_train
        
        elif split == 'test':
            X_test, y_test, severity_test = create_sequences_per_trial(
                test_data, window_size, stride
            )
            return X_test, y_test, severity_test
        
        else:
            raise ValueError("split must be either 'train' or 'test'")
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    preprocess_data() 