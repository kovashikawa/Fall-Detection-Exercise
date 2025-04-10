import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import LSTMClassifier
from preprocess import preprocess_data
from logger_config import setup_logger

# Setup logger
logger = setup_logger('evaluate', 'evaluation')

def evaluate_model(model, test_loader, device):
    logger.info("Starting model evaluation")
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
    
    logger.info("Evaluation completed")
    return all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred):
    logger.info("Plotting confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    logger.info("Confusion matrix saved as 'confusion_matrix.png'")

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data_dir = 'data'
        processed_data = preprocess_data(data_dir)
        
        # Create test loader
        logger.info("Creating test data loader")
        test_dataset = TensorDataset(processed_data['test'][0], processed_data['test'][1])
        test_loader = DataLoader(test_dataset, batch_size=64)
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Load model
        logger.info("Loading trained model")
        model = LSTMClassifier(63, 64, 2).to(device)
        model.load_state_dict(torch.load('models/lstm_model.pth'))
        logger.info("Model loaded successfully")
        
        # Evaluate model
        logger.info("Evaluating model performance")
        y_true, y_pred = evaluate_model(model, test_loader, device)
        
        # Print classification report
        logger.info("\nClassification Report:")
        report = classification_report(y_true, y_pred, digits=4)
        logger.info(report)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 