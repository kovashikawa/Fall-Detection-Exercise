import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from train import main
from logger_config import setup_logger
from model_comparison import ModelComparison

# Setup logger
logger = setup_logger('run_training', 'training')

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'logs', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def load_metrics_history():
    """Load or create metrics history file"""
    metrics_file = 'results/metrics_history.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {
        'clf_train_loss': [],
        'reg_train_loss': [],
        'train_acc': [],
        'train_severity': [],
        'clf_val_loss': [],
        'reg_val_loss': [],
        'val_acc': [],
        'val_severity': [],
        'timestamps': []
    }

def save_metrics_history(history, timestamp):
    """Save metrics history to JSON file"""
    metrics_file = f'results/metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved metrics history to {metrics_file}")

def plot_metrics(history, timestamp):
    """Plot and save training metrics"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot classification metrics
    axes[0, 1].plot(history['val_accuracy'], label='Accuracy')
    axes[0, 1].plot(history['val_precision'], label='Precision')
    axes[0, 1].plot(history['val_recall'], label='Recall')
    axes[0, 1].plot(history['val_f1'], label='F1 Score')
    axes[0, 1].set_title('Classification Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    
    # Plot severity MSE
    axes[1, 0].plot(history['val_severity_mse'], label='Severity MSE')
    axes[1, 0].set_title('Severity Prediction MSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    
    # Plot learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'results/training_metrics_{timestamp}.png')
    plt.close()
    logger.info(f"Saved training metrics plot to results/training_metrics_{timestamp}.png")

def run_training():
    try:
        # Setup directories
        setup_directories()
        
        # Get timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting training run at {timestamp}")
        
        # Load metrics history
        metrics_history = load_metrics_history()
        metrics_history['timestamps'].append(timestamp)
        
        # Run training
        history = main()
        
        # Save metrics
        save_metrics_history(history, timestamp)
        
        # Plot metrics
        plot_metrics(history, timestamp)
        
        # Compare with previous models
        comparison = ModelComparison()
        comparison.compare_models('val_acc')
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    run_training() 