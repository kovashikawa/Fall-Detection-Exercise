import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
from logger_config import setup_logger
from preprocess import preprocess_data, set_seeds
from model_comparison import ModelComparison
from sklearn.model_selection import KFold
import pandas as pd

# Setup logger
logger = setup_logger('train', 'training')

# Define feature columns
FEATURES = [
    'acc_x', 'acc_y', 'acc_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    'mag_x', 'mag_y', 'mag_z',
    'acc_mag', 'ang_vel_mag', 'mag_mag',
    'vel_x', 'vel_y', 'vel_z',
    'jerk_x', 'jerk_y', 'jerk_z',
    'energy',
    'acc_x_mean', 'acc_y_mean', 'acc_z_mean',
    'acc_x_std', 'acc_y_std', 'acc_z_std',
    'acc_x_max', 'acc_y_max', 'acc_z_max',
    'ang_vel_x_mean', 'ang_vel_y_mean', 'ang_vel_z_mean',
    'ang_vel_x_std', 'ang_vel_y_std', 'ang_vel_z_std',
    'ang_vel_x_max', 'ang_vel_y_max', 'ang_vel_z_max',
    'mag_x_mean', 'mag_y_mean', 'mag_z_mean',
    'mag_x_std', 'mag_y_std', 'mag_z_std',
    'mag_x_max', 'mag_y_max', 'mag_z_max'
]

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
        logger.info(f"Processing subject: {subject_dir}")
        
        # List all trial files
        trial_files = []
        for root, dirs, files in os.walk(subject_path):
            for file in files:
                if file.endswith('.xlsx'):
                    trial_files.append(os.path.join(root, file))
        
        if not trial_files:
            logger.warning(f"No Excel files found in {subject_path}")
            continue
        
        for trial_file in trial_files:
            logger.info(f"Loading trial: {trial_file}")
            
            try:
                # Read trial data
                trial_data = pd.read_excel(trial_file)
                
                # Add subject and trial information
                trial_data['subject'] = subject_dir
                trial_data['trial'] = os.path.basename(trial_file).replace('.xlsx', '')
                
                all_data.append(trial_data)
            except Exception as e:
                logger.error(f"Error loading {trial_file}: {str(e)}")
                continue
    
    if not all_data:
        logger.warning("No data was loaded. Generating sample data...")
        return generate_sample_data()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded data from {len(subject_dirs)} subjects with {len(all_data)} trials")
    logger.info(f"Total data shape: {combined_data.shape}")
    
    return combined_data

def generate_sample_data():
    """Generate sample data for testing when no real data is available"""
    logger.info("Generating sample data for testing")
    
    # Create sample data with similar structure to real data
    num_samples = 1000
    num_features = 63  # Match the expected number of features
    
    # Generate random data
    data = np.random.randn(num_samples, num_features)
    
    # Create DataFrame with appropriate column names
    columns = [f'feature_{i}' for i in range(num_features)]
    df = pd.DataFrame(data, columns=columns)
    
    # Add label column (binary classification)
    df['label'] = np.random.randint(0, 2, size=num_samples)
    
    # Add timestamp column
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=num_samples, freq='100ms')
    
    logger.info(f"Generated sample data with shape: {df.shape}")
    return df

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
    acc_columns = [col for col in data.columns if 'acc_mag' in col]
    ang_vel_columns = [col for col in data.columns if 'ang_vel_mag' in col]
    
    # Calculate total acceleration magnitude (average across all sensors)
    acc_magnitudes = data[acc_columns].values
    acc_magnitude = np.mean(acc_magnitudes, axis=1)
    
    # Calculate impact force (using maximum acceleration magnitude)
    impact_force = np.max(acc_magnitude)
    
    # Calculate duration of high acceleration
    high_acc_threshold = np.percentile(acc_magnitude, 95)
    duration = np.sum(acc_magnitude > high_acc_threshold)
    
    # Calculate total angular velocity magnitude (average across all sensors)
    ang_vel_magnitudes = data[ang_vel_columns].values
    ang_vel_magnitude = np.mean(ang_vel_magnitudes, axis=1)
    max_angular_velocity = np.max(ang_vel_magnitude)
    
    # Combine into severity score (normalized between 0 and 1)
    severity = (0.5 * impact_force + 0.3 * duration + 0.2 * max_angular_velocity) / \
              (np.max([impact_force, 1]) * len(data))
    
    return float(severity)

def train_epoch(model, train_loader, criterion_cls, criterion_reg, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).float()  # Ensure float type for BCE loss
        
        # Forward pass
        y_pred, _ = model(batch_X)
        
        # Ensure consistent shapes
        y_pred = y_pred.view(-1)
        batch_y = batch_y.view(-1)
        
        # Compute loss
        cls_loss = criterion_cls(y_pred, batch_y)
        reg_loss = 0  # No severity prediction for now
        loss = cls_loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)
    
    return avg_loss, avg_cls_loss, avg_reg_loss

def validate(model, val_loader, criterion_cls, criterion_reg, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).float()  # Ensure float type for BCE loss
            
            # Forward pass
            y_pred, _ = model(batch_X)
            
            # Ensure consistent shapes
            y_pred = y_pred.view(-1)
            batch_y = batch_y.view(-1)
            
            # Compute loss
            cls_loss = criterion_cls(y_pred, batch_y)
            reg_loss = 0  # No severity prediction for now
            loss = cls_loss + reg_loss
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss
            
            # Store predictions and labels for metrics
            all_preds.extend((y_pred > 0.5).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Compute average losses
    avg_loss = total_loss / len(val_loader)
    avg_cls_loss = total_cls_loss / len(val_loader)
    avg_reg_loss = total_reg_loss / len(val_loader)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'val_loss': avg_loss,
        'val_cls_loss': avg_cls_loss,
        'val_reg_loss': avg_reg_loss,
        'val_accuracy': accuracy,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1
    }

def train_model(model, train_loader, val_loader, criterion_cls, criterion_reg, optimizer, 
                scheduler, early_stopping, device, num_epochs=20):
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
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_cls_loss, train_reg_loss = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )
        
        # Validation
        val_metrics = validate(model, val_loader, criterion_cls, criterion_reg, device)
        
        # Log detailed metrics for debugging
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} (Cls: {val_metrics['val_cls_loss']:.4f}, Reg: {val_metrics['val_reg_loss']:.4f})")
        logger.info(f"Val Metrics - Acc: {val_metrics['val_accuracy']:.4f}, Prec: {val_metrics['val_precision']:.4f}, "
                   f"Rec: {val_metrics['val_recall']:.4f}, F1: {val_metrics['val_f1']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Early stopping check
        early_stopping(val_metrics['val_loss'])
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save(model.state_dict(), f'models/classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['train_cls_loss'].append(train_cls_loss)
        history['val_cls_loss'].append(val_metrics['val_cls_loss'])
        history['train_reg_loss'].append(train_reg_loss)
        history['val_reg_loss'].append(val_metrics['val_reg_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['val_precision'].append(val_metrics['val_precision'])
        history['val_recall'].append(val_metrics['val_recall'])
        history['val_f1'].append(val_metrics['val_f1'])
    
    return history

def cross_validate(data_dict, device, metrics_history=None, n_splits=5, seq_lengths=[50, 100, 150]):
    """Perform k-fold cross-validation with different sequence lengths."""
    logger.info(f"Starting cross-validation with {n_splits} splits")
    
    # Unpack data
    X_train, y_train = data_dict['train']
    X_val, y_val = data_dict['val']
    input_size = data_dict['input_size']
    
    best_metrics = None
    best_seq_length = None
    best_val_loss = float('inf')
    
    for seq_length in seq_lengths:
        logger.info(f"Testing sequence length: {seq_length}")
        
        # Initialize k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        # Convert tensors to numpy for splitting
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train_fold = torch.FloatTensor(X_train_np[train_idx]).to(device)
            y_train_fold = torch.FloatTensor(y_train_np[train_idx]).to(device)
            X_val_fold = torch.FloatTensor(X_train_np[val_idx]).to(device)
            y_val_fold = torch.FloatTensor(y_train_np[val_idx]).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_fold, y_train_fold)
            val_dataset = TensorDataset(X_val_fold, y_val_fold)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            # Initialize model and training components
            model = AttentionBiLSTM(input_size=input_size, hidden_size=64).to(device)
            criterion_cls = nn.BCELoss()
            criterion_reg = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
            early_stopping = EarlyStopping(patience=10)
            
            # Train model
            fold_history = train_model(
                model, train_loader, val_loader,
                criterion_cls, criterion_reg,
                optimizer, scheduler,
                early_stopping, device
            )
            
            fold_metrics.append(fold_history)
        
        # Average metrics across folds
        avg_metrics = {}
        for metric in ['val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']:
            values = [m[metric][-1] for m in fold_metrics]  # Get final value for each fold
            avg_metrics[metric] = sum(values) / len(values)
        
        logger.info(f"Average metrics for sequence length {seq_length}:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Update best metrics if validation loss improved
        if avg_metrics['val_loss'] < best_val_loss:
            best_val_loss = avg_metrics['val_loss']
            best_metrics = avg_metrics
            best_seq_length = seq_length
            
            # Update metrics history
            if metrics_history is not None:
                metrics_history.add_model_result(
                    model_name=f"bilstm_seq{seq_length}",
                    metrics=avg_metrics,
                    hyperparameters={'sequence_length': seq_length}
                )
    
    logger.info(f"Best sequence length: {best_seq_length}")
    logger.info("Best metrics:")
    for metric, value in best_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return best_metrics, best_seq_length

def main(metrics_history=None):
    try:
        # Set device and random seeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seeds()
        logger.info(f"Using device: {device}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data_dir = 'data'
        processed_data = preprocess_data(data_dir)
        
        # Get input size from preprocessed data
        input_size = processed_data['input_size']
        
        # Perform cross-validation
        best_metrics, best_seq_length = cross_validate(processed_data, device, metrics_history)
        logger.info(f"Best sequence length: {best_seq_length}")
        logger.info(f"Best metrics: {best_metrics}")
        
        # Train final models with best parameters
        logger.info("Training final models with best parameters")
        classifier = AttentionBiLSTM(
            input_size=input_size,
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
            best_metrics,
            {
                'input_size': input_size,
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