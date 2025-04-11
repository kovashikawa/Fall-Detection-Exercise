import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from logger_config import setup_logger
from datetime import datetime

# Setup logger with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger('train', f'training_{timestamp}')

class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in tqdm(loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1).float())
            total_loss += loss.item()
    return total_loss / len(loader)

def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}_{timestamp}.png')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data'
    seq_length = 50
    processed_data = preprocess_data(data_dir, seq_length)
    input_dim = processed_data['train'][0].shape[2]

    train_loader = DataLoader(TensorDataset(*processed_data['train']), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(*processed_data['val']), batch_size=32)

    classifier = ImprovedLSTMClassifier(input_dim, 64, 2).to(device)
    regressor = LSTMRegressor(input_dim, 64, 2).to(device)

    clf_criterion = nn.BCELoss()
    reg_criterion = nn.MSELoss()

    clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    reg_optimizer = optim.Adam(regressor.parameters(), lr=0.001)

    epochs = 15
    clf_train_losses, clf_val_losses = [], []
    reg_train_losses, reg_val_losses = [], []

    for epoch in range(epochs):
        logger.info(f'Classifier Epoch {epoch+1}')
        clf_train_loss = train(classifier, train_loader, clf_criterion, clf_optimizer, device)
        clf_val_loss = evaluate(classifier, val_loader, clf_criterion, device)
        clf_train_losses.append(clf_train_loss)
        clf_val_losses.append(clf_val_loss)
        logger.info(f'Classifier - Epoch {epoch+1}, Train: {clf_train_loss:.4f}, Val: {clf_val_loss:.4f}')

        logger.info(f'Regressor Epoch {epoch+1}')
        reg_train_loss = train(regressor, train_loader, reg_criterion, reg_optimizer, device)
        reg_val_loss = evaluate(regressor, val_loader, reg_criterion, device)
        reg_train_losses.append(reg_train_loss)
        reg_val_losses.append(reg_val_loss)
        logger.info(f'Regressor - Epoch {epoch+1}, Train: {reg_train_loss:.4f}, Val: {reg_val_loss:.4f}')

    plot_losses(clf_train_losses, clf_val_losses, 'Classifier Losses')
    plot_losses(reg_train_losses, reg_val_losses, 'Regressor Losses')

    torch.save(classifier.state_dict(), f'models/classifier_{timestamp}.pth')
    torch.save(regressor.state_dict(), f'models/regressor_{timestamp}.pth')

if __name__ == '__main__':
    main()