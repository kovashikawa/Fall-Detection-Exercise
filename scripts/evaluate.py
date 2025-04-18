import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import AttentionBiLSTM
from preprocess import preprocess_data
from logger_config import setup_logger
import os
from datetime import datetime
from model_comparison import ModelComparison
import json

# Setup logger
logger = setup_logger('evaluate', 'evaluation')

def evaluate_model(model_path, test_data):
    """Evaluate model on test data"""
    logger.info(f"Evaluating model from {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionBiLSTM(
        input_size=test_data['input_size'],
        hidden_size=64
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get test data
    X_test, y_test = test_data['test']
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Evaluate
    with torch.no_grad():
        pred_cls, pred_severity = model(X_test)
        pred_labels = (pred_cls > 0.5).float()
        
        # Calculate metrics
        accuracy = (pred_labels == y_test.unsqueeze(1)).float().mean()
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        # Calculate severity error
        severity_error = torch.mean((pred_severity - y_test.unsqueeze(1))**2)
        logger.info(f"Severity MSE: {severity_error:.4f}")
    
    return {
        'accuracy': accuracy.item(),
        'severity_mse': severity_error.item()
    }

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
    # Get latest model
    comparison = ModelComparison()
    best_model = comparison.get_best_model('val_acc')
    
    if best_model is None:
        logger.error("No trained model found")
        return
    
    # Load and preprocess test data
    test_data = preprocess_data('data')
    
    # Evaluate model
    metrics = evaluate_model(best_model['model_path'], test_data)
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/evaluation_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'model_path': best_model['model_path'],
            'metrics': metrics,
            'timestamp': timestamp
        }, f, indent=4)
    
    logger.info(f"Evaluation results saved to {results_file}")

if __name__ == '__main__':
    main() 