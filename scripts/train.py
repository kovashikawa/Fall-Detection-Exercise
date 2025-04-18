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
        trial_files = [f for f in os.listdir(subject_path) if f.endswith('.csv')]
        
        if not trial_files:
            logger.warning(f"No CSV files found in {subject_path}")
            continue
        
        for trial_file in trial_files:
            trial_path = os.path.join(subject_path, trial_file)
            logger.info(f"Loading trial: {trial_file}")
            
            try:
                # Read trial data
                trial_data = pd.read_csv(trial_path)
                
                # Add subject and trial information
                trial_data['subject'] = subject_dir
                trial_data['trial'] = trial_file.replace('.csv', '')
                
                all_data.append(trial_data)
            except Exception as e:
                logger.error(f"Error loading {trial_path}: {str(e)}")
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
    # Get acceleration columns
    acc_x = data['acc_x'].values
    acc_y = data['acc_y'].values
    acc_z = data['acc_z'].values
    
    # Calculate total acceleration magnitude
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Calculate impact force (using maximum acceleration magnitude)
    impact_force = np.max(acc_magnitude)
    
    # Calculate duration of high acceleration
    high_acc_threshold = np.percentile(acc_magnitude, 95)
    duration = np.sum(acc_magnitude > high_acc_threshold)
    
    # Get angular velocity columns
    ang_vel_x = data['ang_vel_x'].values
    ang_vel_y = data['ang_vel_y'].values
    ang_vel_z = data['ang_vel_z'].values
    
    # Calculate total angular velocity magnitude
    ang_vel_magnitude = np.sqrt(ang_vel_x**2 + ang_vel_y**2 + ang_vel_z**2)
    max_angular_velocity = np.max(ang_vel_magnitude)
    
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
            
            # Create sequences and compute severity scores
            X_train, y_train = create_sliding_sequences(
                train_data[features].values, 
                train_data['label'].values,
                window_size=seq_length,
                stride=seq_length//2
            )
            severity_train = np.array([compute_severity_score(train_data.iloc[i:i+seq_length]) 
                                     for i in range(0, len(train_data)-seq_length+1, seq_length//2)])
            
            X_val, y_val = create_sliding_sequences(
                val_data[features].values,
                val_data['label'].values,
                window_size=seq_length,
                stride=seq_length//2
            )
            severity_val = np.array([compute_severity_score(val_data.iloc[i:i+seq_length]) 
                                   for i in range(0, len(val_data)-seq_length+1, seq_length//2)])
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train, torch.tensor(severity_train).float())
            val_dataset = TensorDataset(X_val, y_val, torch.tensor(severity_val).float())
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64)
            
            # Initialize model
            input_size = X_train.shape[2]  # Number of features
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