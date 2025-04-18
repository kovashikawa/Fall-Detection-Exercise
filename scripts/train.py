import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_data
from logger_config import setup_logger
from model_comparison import ModelComparison
from sklearn.model_selection import KFold
import numpy as np
from datetime import datetime

# Setup logger with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger('train', f'training_{timestamp}')

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(AttentionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Compute attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.fc(attended)
        return self.sigmoid(out)

class FallSeverityRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(FallSeverityRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(attended)

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
    # Calculate impact force (using acceleration magnitude)
    acc_magnitude = np.sqrt(np.sum(data[['acc_x', 'acc_y', 'acc_z']]**2, axis=1))
    impact_force = np.max(acc_magnitude)
    
    # Calculate duration of high acceleration
    high_acc_threshold = np.percentile(acc_magnitude, 95)
    duration = np.sum(acc_magnitude > high_acc_threshold)
    
    # Combine into severity score (normalized between 0 and 1)
    severity = (impact_force * duration) / (np.max(acc_magnitude) * len(data))
    return severity

def train_epoch(models, train_loader, criteria, optimizers, device):
    classifier, regressor = models
    clf_criterion, reg_criterion = criteria
    clf_optimizer, reg_optimizer = optimizers
    
    classifier.train()
    regressor.train()
    
    clf_total_loss = 0
    reg_total_loss = 0
    clf_correct = 0
    total = 0
    
    for x_batch, y_batch, severity_batch in train_loader:
        x_batch, y_batch, severity_batch = x_batch.to(device), y_batch.to(device), severity_batch.to(device)
        
        # Train classifier
        clf_optimizer.zero_grad()
        clf_outputs = classifier(x_batch)
        clf_loss = clf_criterion(clf_outputs, y_batch)
        clf_loss.backward()
        clf_optimizer.step()
        
        # Train regressor
        reg_optimizer.zero_grad()
        reg_outputs = regressor(x_batch)
        reg_loss = reg_criterion(reg_outputs, severity_batch)
        reg_loss.backward()
        reg_optimizer.step()
        
        clf_total_loss += clf_loss.item()
        reg_total_loss += reg_loss.item()
        predicted = (clf_outputs > 0.5).float()
        total += y_batch.size(0)
        clf_correct += (predicted == y_batch).sum().item()
    
    return (clf_total_loss / len(train_loader), 
            reg_total_loss / len(train_loader), 
            clf_correct / total)

def validate_epoch(models, val_loader, criteria, device):
    classifier, regressor = models
    clf_criterion, reg_criterion = criteria
    
    classifier.eval()
    regressor.eval()
    
    clf_total_loss = 0
    reg_total_loss = 0
    clf_correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch, severity_batch in val_loader:
            x_batch, y_batch, severity_batch = x_batch.to(device), y_batch.to(device), severity_batch.to(device)
            
            clf_outputs = classifier(x_batch)
            reg_outputs = regressor(x_batch)
            
            clf_loss = clf_criterion(clf_outputs, y_batch)
            reg_loss = reg_criterion(reg_outputs, severity_batch)
            
            clf_total_loss += clf_loss.item()
            reg_total_loss += reg_loss.item()
            predicted = (clf_outputs > 0.5).float()
            total += y_batch.size(0)
            clf_correct += (predicted == y_batch).sum().item()
    
    return (clf_total_loss / len(val_loader),
            reg_total_loss / len(val_loader),
            clf_correct / total)

def cross_validate(data, n_splits=5, seq_lengths=[50, 100, 150]):
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
            
            # Initialize models
            classifier = AttentionBiLSTM(
                input_dim=X_train.shape[2],
                hidden_dim=64,
                num_layers=2
            ).to(device)
            
            regressor = FallSeverityRegressor(
                input_dim=X_train.shape[2],
                hidden_dim=64,
                num_layers=2
            ).to(device)
            
            clf_criterion = nn.BCELoss()
            reg_criterion = nn.MSELoss()
            
            clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
            reg_optimizer = optim.Adam(regressor.parameters(), lr=0.001)
            
            clf_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                clf_optimizer, mode='min', factor=0.1, patience=3, verbose=True
            )
            reg_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                reg_optimizer, mode='min', factor=0.1, patience=3, verbose=True
            )
            
            early_stopping = EarlyStopping(patience=5)
            
            # Training loop
            clf_train_losses = []
            reg_train_losses = []
            clf_val_losses = []
            reg_val_losses = []
            train_accs = []
            val_accs = []
            
            for epoch in range(15):
                clf_train_loss, reg_train_loss, train_acc = train_epoch(
                    (classifier, regressor),
                    train_loader,
                    (clf_criterion, reg_criterion),
                    (clf_optimizer, reg_optimizer),
                    device
                )
                
                clf_val_loss, reg_val_loss, val_acc = validate_epoch(
                    (classifier, regressor),
                    val_loader,
                    (clf_criterion, reg_criterion),
                    device
                )
                
                clf_scheduler.step(clf_val_loss)
                reg_scheduler.step(reg_val_loss)
                early_stopping(clf_val_loss + reg_val_loss)
                
                clf_train_losses.append(clf_train_loss)
                reg_train_losses.append(reg_train_loss)
                clf_val_losses.append(clf_val_loss)
                reg_val_losses.append(reg_val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break
            
            fold_metrics.append({
                'clf_val_loss': min(clf_val_losses),
                'reg_val_loss': min(reg_val_losses),
                'val_acc': max(val_accs),
                'clf_train_loss': min(clf_train_losses),
                'reg_train_loss': min(reg_train_losses),
                'train_acc': max(train_accs)
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
        best_metrics, best_seq_length = cross_validate(data)
        logger.info(f"Best sequence length: {best_seq_length}")
        logger.info(f"Best metrics: {best_metrics}")
        
        # Train final models with best parameters
        logger.info("Training final models with best parameters")
        classifier = AttentionBiLSTM(
            input_dim=63,
            hidden_dim=64,
            num_layers=2
        ).to(device)
        
        regressor = FallSeverityRegressor(
            input_dim=63,
            hidden_dim=64,
            num_layers=2
        ).to(device)
        
        # Save models and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classifier_path = f'models/classifier_{timestamp}.pth'
        regressor_path = f'models/regressor_{timestamp}.pth'
        
        torch.save(classifier.state_dict(), classifier_path)
        torch.save(regressor.state_dict(), regressor_path)
        
        # Add to model comparison
        comparison = ModelComparison()
        comparison.add_model_result(
            'AttentionBiLSTM_Classifier',
            {k: v for k, v in best_metrics.items() if 'clf' in k or 'acc' in k},
            {
                'input_dim': 63,
                'hidden_dim': 64,
                'num_layers': 2,
                'sequence_length': best_seq_length
            },
            classifier_path
        )
        
        comparison.add_model_result(
            'FallSeverityRegressor',
            {k: v for k, v in best_metrics.items() if 'reg' in k},
            {
                'input_dim': 63,
                'hidden_dim': 64,
                'num_layers': 2,
                'sequence_length': best_seq_length
            },
            regressor_path
        )
        
        # Compare with previous models
        comparison.compare_models('val_acc')
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()