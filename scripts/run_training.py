import os
import json
from datetime import datetime
from train import main as train_main
from logger_config import setup_logger
from model_comparison import ModelComparison
from preprocess import set_seeds

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

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

def save_metrics_history(metrics_history):
    """Save metrics history to file"""
    with open('results/metrics_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=4)

def plot_metrics(metrics_history):
    """Plot training metrics"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Plot classification metrics
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['clf_train_loss'], label='Train Loss')
    plt.plot(metrics_history['clf_val_loss'], label='Validation Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_acc'], label='Train Accuracy')
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot regression metrics
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['reg_train_loss'], label='Train Loss')
    plt.plot(metrics_history['reg_val_loss'], label='Validation Loss')
    plt.title('Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['train_severity'], label='Train Severity Error')
    plt.plot(metrics_history['val_severity'], label='Validation Severity Error')
    plt.title('Severity Prediction Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def run_training():
    # Set random seeds for reproducibility
    set_seeds()
    
    # Setup directories and logging
    setup_directories()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger('run_training', f'run_training_{timestamp}')
    
    logger.info("Starting training process")
    logger.info(f"Timestamp: {timestamp}")
    
    # Load metrics history
    metrics_history = load_metrics_history()
    metrics_history['timestamps'].append(timestamp)
    
    try:
        # Run training
        train_main(metrics_history)
        
        # Save metrics and plot
        save_metrics_history(metrics_history)
        plot_metrics(metrics_history)
        
        # Compare with previous models
        comparison = ModelComparison()
        comparison.compare_models('val_acc')
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    run_training() 