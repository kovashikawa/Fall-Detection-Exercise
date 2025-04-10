import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_data
from logger_config import setup_logger

# Setup logger
logger = setup_logger('train', 'training')

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(LSTMClassifier, self).__init__()
        logger.debug(f"Initializing LSTM with input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    logger.info("Starting model training")
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        for x_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    logger.info("Training completed successfully")
    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    logger.info("Plotting training and validation losses")
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()
    logger.info("Loss plot saved as 'loss_plot.png'")

def main():
    try:
        # Hyperparameters
        input_dim = 63  # Number of features
        hidden_dim = 64
        num_layers = 2
        lr = 0.001
        epochs = 15
        batch_size = 64
        seq_length = 100
        
        logger.info("Setting up training configuration")
        logger.info(f"Hyperparameters: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        logger.info(f"Training parameters: lr={lr}, epochs={epochs}, batch_size={batch_size}, seq_length={seq_length}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data_dir = 'data'
        processed_data = preprocess_data(data_dir, seq_length)
        
        # Create data loaders
        logger.info("Creating data loaders")
        train_dataset = TensorDataset(processed_data['train'][0], processed_data['train'][1])
        val_dataset = TensorDataset(processed_data['val'][0], processed_data['val'][1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Initialize model
        logger.info("Initializing model")
        model = LSTMClassifier(input_dim, hidden_dim, num_layers).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train model
        logger.info("Starting model training")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs, device
        )
        
        # Plot losses
        plot_losses(train_losses, val_losses)
        
        # Save model
        logger.info("Saving model")
        torch.save(model.state_dict(), 'models/lstm_model.pth')
        logger.info("Model saved successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 