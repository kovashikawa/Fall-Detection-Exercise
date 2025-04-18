import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from logger_config import setup_logger

logger = setup_logger('model_comparison', 'model_comparison')

class ModelComparison:
    def __init__(self, results_dir='model_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results_file = self.results_dir / 'model_comparison.json'
        self.load_existing_results()

    def load_existing_results(self):
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = {}

    def save_results(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=4)

    def add_model_result(self, model_name, metrics, hyperparameters, model_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = f"{model_name}_{timestamp}"
        
        self.results[result_id] = {
            'model_name': model_name,
            'timestamp': timestamp,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'model_path': str(model_path)
        }
        self.save_results()
        logger.info(f"Added results for model: {model_name}")

    def compare_models(self, metric='f1'):
        if not self.results:
            logger.warning("No model results available for comparison")
            return

        comparison_data = []
        for result_id, result in self.results.items():
            comparison_data.append({
                'model': result['model_name'],
                'timestamp': result['timestamp'],
                metric: result['metrics'].get(metric, None)
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values(by=metric, ascending=False)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y=metric, data=df)
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f'model_comparison_{metric}.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Comparison plot saved to {plot_path}")
        return df

    def get_best_model(self, metric='f1'):
        if not self.results:
            return None

        best_result = max(
            self.results.items(),
            key=lambda x: x[1]['metrics'].get(metric, 0)
        )
        return best_result[1]

    def plot_training_history(self, model_id):
        if model_id not in self.results:
            logger.error(f"Model {model_id} not found in results")
            return

        result = self.results[model_id]
        if 'training_history' not in result:
            logger.warning(f"No training history available for {model_id}")
            return

        history = result['training_history']
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy History')
        plt.legend()
        
        plt.tight_layout()
        plot_path = self.results_dir / f'training_history_{model_id}.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {plot_path}")

if __name__ == '__main__':
    # Example usage
    comparison = ModelComparison()
    
    # Example metrics
    metrics = {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.93,
        'f1': 0.935
    }
    
    # Example hyperparameters
    hyperparameters = {
        'input_dim': 63,
        'hidden_dim': 64,
        'num_layers': 2,
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    # Add a model result
    comparison.add_model_result(
        'LSTM_Classifier',
        metrics,
        hyperparameters,
        'models/lstm_model.pth'
    )
    
    # Compare models
    comparison.compare_models('f1') 