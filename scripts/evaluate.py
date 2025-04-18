import torch
import numpy as np
from train import AttentionBiLSTM, FallSeverityRegressor
from preprocess import preprocess_data
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from logger_config import setup_logger

# Setup logger
logger = setup_logger('evaluate', 'evaluation')

def load_latest_model(model_type='classifier'):
    """Load the most recent model of the specified type"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find the most recent model file
    model_files = [f for f in os.listdir(models_dir) if f.startswith(model_type) and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No {model_type} model files found in {models_dir}")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    logger.info(f"Loading model from: {model_path}")
    
    return model_path

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_severity = []
    
    with torch.no_grad():
        for x_batch, y_batch, severity_batch in test_loader:
            x_batch, y_batch, severity_batch = x_batch.to(device), y_batch.to(device), severity_batch.to(device)
            outputs = model(x_batch)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_severity.extend(severity_batch.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_severity)

def plot_results(predictions, labels, severity_scores):
    plt.figure(figsize=(15, 5))
    
    # Plot classification results
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(labels, predictions), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot severity scores
    plt.subplot(1, 2, 2)
    plt.scatter(severity_scores, predictions, alpha=0.5)
    plt.title('Severity vs Prediction')
    plt.xlabel('Severity Score')
    plt.ylabel('Prediction')
    
    plt.tight_layout()
    plt.savefig('results/evaluation_results.png')
    plt.close()

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load data
        logger.info("Loading and preprocessing data")
        X_test, y_test, severity_test = preprocess_data(split='test')
        
        # Load models
        logger.info("Loading models")
        classifier_path = load_latest_model('classifier')
        regressor_path = load_latest_model('regressor')
        
        # Initialize models
        classifier = AttentionBiLSTM(
            input_dim=X_test.shape[1],
            hidden_dim=64,
            num_layers=2
        ).to(device)
        
        regressor = FallSeverityRegressor(
            input_dim=X_test.shape[1],
            hidden_dim=64,
            num_layers=2
        ).to(device)
        
        # Load state dicts
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        regressor.load_state_dict(torch.load(regressor_path, map_location=device))
        
        # Convert data to tensors
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
        severity_test_tensor = torch.FloatTensor(severity_test).unsqueeze(1).to(device)
        
        # Create test dataset and loader
        test_dataset = torch.utils.data.TensorDataset(
            X_test_tensor, y_test_tensor, severity_test_tensor
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
        
        # Evaluate models
        logger.info("Evaluating classifier")
        clf_preds, clf_labels, clf_severity = evaluate_model(classifier, test_loader, device)
        
        logger.info("Evaluating regressor")
        reg_preds, reg_labels, reg_severity = evaluate_model(regressor, test_loader, device)
        
        # Calculate metrics
        accuracy = accuracy_score(clf_labels, clf_preds)
        precision = precision_score(clf_labels, clf_preds)
        recall = recall_score(clf_labels, clf_preds)
        f1 = f1_score(clf_labels, clf_preds)
        mse = mean_squared_error(reg_severity, reg_preds)
        
        logger.info(f"Classification Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Regression MSE: {mse:.4f}")
        
        # Plot results
        plot_results(clf_preds, clf_labels, clf_severity)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 