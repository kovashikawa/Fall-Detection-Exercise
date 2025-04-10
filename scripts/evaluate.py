import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import LSTMClassifier
from preprocess import preprocess_data

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    data_dir = 'data'
    processed_data = preprocess_data(data_dir)
    
    # Create test loader
    test_dataset = TensorDataset(processed_data['test'][0], processed_data['test'][1])
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Load model
    model = LSTMClassifier(63, 64, 2).to(device)
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    
    # Evaluate model
    y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    main() 