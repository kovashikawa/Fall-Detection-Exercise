import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
from logger_config import setup_logger
from preprocess import preprocess_data
from model_comparison import ModelComparison
from sklearn.model_selection import KFold
import pandas as pd

# Setup logger
logger = setup_logger('train', 'training')

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

def create_sliding_sequences(X, y, window_size=100, stride=50):
    """Create sequences using sliding window approach"""
    sequences = []
    labels = []
    
    for i in range(0, len(X) - window_size + 1, stride):
        sequences.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])  # Use the label at the end of the window
    
    return np.array(sequences), np.array(labels)

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Regression head for severity
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification and regression outputs
        classification = self.classifier(context)
        severity = self.regressor(context)
        
        return classification, severity

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

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

def train_epoch(model, train_loader, criterion_cls, criterion_reg, optimizer, device):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    for batch_X, batch_y, batch_severity in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_severity = batch_severity.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_cls, pred_severity = model(batch_X)
        
        # Compute losses
        cls_loss = criterion_cls(pred_cls, batch_y.unsqueeze(1))
        reg_loss = criterion_reg(pred_severity, batch_severity.unsqueeze(1))
        loss = cls_loss + reg_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
    
    return total_loss / len(train_loader), total_cls_loss / len(train_loader), total_reg_loss / len(train_loader)

def validate(model, val_loader, criterion_cls, criterion_reg, device):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    all_preds = []
    all_labels = []
    all_severity = []
    all_pred_severity = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_severity in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_severity = batch_severity.to(device)
            
            # Forward pass
            pred_cls, pred_severity = model(batch_X)
            
            # Compute losses
            cls_loss = criterion_cls(pred_cls, batch_y.unsqueeze(1))
            reg_loss = criterion_reg(pred_severity, batch_severity.unsqueeze(1))
            loss = cls_loss + reg_loss
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            
            # Store predictions
            all_preds.extend((pred_cls > 0.5).float().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_severity.extend(batch_severity.cpu().numpy())
            all_pred_severity.extend(pred_severity.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_severity = np.array(all_severity)
    all_pred_severity = np.array(all_pred_severity)
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'cls_loss': total_cls_loss / len(val_loader),
        'reg_loss': total_reg_loss / len(val_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'severity_mse': np.mean((all_severity - all_pred_severity) ** 2)
    }
    
    return metrics

def train_model(model, train_loader, val_loader, criterion_cls, criterion_reg, optimizer, 
                scheduler, early_stopping, device, num_epochs=100):
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cls_loss': [],
        'val_cls_loss': [],
        'train_reg_loss': [],
        'val_reg_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_severity_mse': []
    }
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_cls_loss, train_reg_loss = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )
        
        # Validation
        val_metrics = validate(model, val_loader, criterion_cls, criterion_reg, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Early stopping check
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), f'models/classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_cls_loss'].append(train_cls_loss)
        history['val_cls_loss'].append(val_metrics['cls_loss'])
        history['train_reg_loss'].append(train_reg_loss)
        history['val_reg_loss'].append(val_metrics['reg_loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_severity_mse'].append(val_metrics['severity_mse'])
        
        # Log progress
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f} (Cls: {val_metrics['cls_loss']:.4f}, Reg: {val_metrics['reg_loss']:.4f})")
        logger.info(f"Val Metrics - Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
                   f"Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        logger.info(f"Val Severity MSE: {val_metrics['severity_mse']:.4f}")
    
    return history

def cross_validate(data, device, metrics_history=None, n_splits=5, seq_lengths=[50, 100, 150]):
    logger.info(f"Starting cross-validation with {n_splits} splits")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_metrics = {}
    
    for seq_length in seq_lengths:
        logger.info(f"Testing sequence length: {seq_length}")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            # Split and preprocess data
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            # Ensure all feature data is numeric and handle any missing values
            X_train = train_data[['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Angular_Velocity_X', 'Angular_Velocity_Y', 'Angular_Velocity_Z']].astype(float).fillna(0)
            y_train = train_data['label'].astype(float).values
            X_val = val_data[['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Angular_Velocity_X', 'Angular_Velocity_Y', 'Angular_Velocity_Z']].astype(float).fillna(0)
            y_val = val_data['label'].astype(float).values
            
            # Create sequences
            X_train_seq, y_train_seq = create_sliding_sequences(
                X_train.values, 
                y_train,
                window_size=seq_length,
                stride=seq_length//2
            )
            
            X_val_seq, y_val_seq = create_sliding_sequences(
                X_val.values,
                y_val,
                window_size=seq_length,
                stride=seq_length//2
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
            y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(device)
            
            # Compute severity scores
            severity_train = []
            for i in range(0, len(train_data)-seq_length+1, seq_length//2):
                window_data = train_data.iloc[i:i+seq_length]
                severity = compute_severity_score(window_data)
                severity_train.append(severity)
            severity_train_tensor = torch.FloatTensor(severity_train).unsqueeze(1).to(device)
            
            severity_val = []
            for i in range(0, len(val_data)-seq_length+1, seq_length//2):
                window_data = val_data.iloc[i:i+seq_length]
                severity = compute_severity_score(window_data)
                severity_val.append(severity)
            severity_val_tensor = torch.FloatTensor(severity_val).unsqueeze(1).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor, severity_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, severity_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64)
            
            # Initialize model
            input_size = X_train_tensor.shape[2]  # Number of features
            model = AttentionBiLSTM(input_size=input_size, hidden_size=64).to(device)
            
            # Initialize loss functions and optimizer
            criterion_cls = nn.BCELoss()
            criterion_reg = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            early_stopping = EarlyStopping(patience=5)
            
            # Train model
            logger.info("Starting training...")
            history = train_model(
                model, train_loader, val_loader, criterion_cls, criterion_reg,
                optimizer, scheduler, early_stopping, device
            )
            
            fold_metrics.append({
                'clf_val_loss': history['val_loss'][-1],
                'reg_val_loss': history['val_reg_loss'][-1],
                'val_acc': history['val_accuracy'][-1],
                'clf_train_loss': history['train_loss'][-1],
                'reg_train_loss': history['train_reg_loss'][-1],
                'train_acc': history['train_accuracy'][-1]
            })
        
        # Average metrics across folds
        avg_metrics = {
            k: np.mean([m[k] for m in fold_metrics])
            for k in fold_metrics[0].keys()
        }
        
        if not best_metrics or avg_metrics['val_acc'] > best_metrics['val_acc']:
            best_metrics = avg_metrics
            best_seq_length = seq_length
    
    return best_metrics, best_seq_length

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data_dir = 'data'
        data = load_all_subjects(data_dir)
        
        # Perform cross-validation
        best_metrics, best_seq_length = cross_validate(data, device)
        logger.info(f"Best sequence length: {best_seq_length}")
        logger.info(f"Best metrics: {best_metrics}")
        
        # Train final models with best parameters
        logger.info("Training final models with best parameters")
        classifier = AttentionBiLSTM(
            input_size=6,
            hidden_size=64
        ).to(device)
        
        # Save models and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classifier_path = f'models/classifier_{timestamp}.pth'
        
        torch.save(classifier.state_dict(), classifier_path)
        
        # Add to model comparison
        comparison = ModelComparison()
        comparison.add_model_result(
            'AttentionBiLSTM_Classifier',
            {k: v for k, v in best_metrics.items() if 'clf' in k or 'acc' in k},
            {
                'input_size': 6,
                'hidden_size': 64,
                'sequence_length': best_seq_length
            },
            classifier_path
        )
        
        # Compare with previous models
        comparison.compare_models('val_acc')
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()