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

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(val_loader), correct / total

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
            
            # Create sequences
            X_train, y_train = create_sliding_sequences(
                train_data[features].values, 
                train_data['label'].values,
                window_size=seq_length,
                stride=seq_length//2
            )
            X_val, y_val = create_sliding_sequences(
                val_data[features].values,
                val_data['label'].values,
                window_size=seq_length,
                stride=seq_length//2
            )
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64)
            
            # Initialize model
            model = AttentionBiLSTM(
                input_dim=X_train.shape[2],
                hidden_dim=64,
                num_layers=2
            ).to(device)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=3, verbose=True
            )
            early_stopping = EarlyStopping(patience=5)
            
            # Training loop
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            
            for epoch in range(15):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_acc = validate_epoch(
                    model, val_loader, criterion, device
                )
                
                scheduler.step(val_loss)
                early_stopping(val_loss)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break
            
            fold_metrics.append({
                'val_loss': min(val_losses),
                'val_acc': max(val_accs),
                'train_loss': min(train_losses),
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
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model = AttentionBiLSTM(
            input_dim=63,
            hidden_dim=64,
            num_layers=2
        ).to(device)
        
        # Save model and results
        model_path = f'models/attention_bilstm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save(model.state_dict(), model_path)
        
        # Add to model comparison
        comparison = ModelComparison()
        comparison.add_model_result(
            'AttentionBiLSTM',
            best_metrics,
            {
                'input_dim': 63,
                'hidden_dim': 64,
                'num_layers': 2,
                'sequence_length': best_seq_length
            },
            model_path
        )
        
        # Compare with previous models
        comparison.compare_models('val_acc')
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()