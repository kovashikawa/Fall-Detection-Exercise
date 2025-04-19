import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os
from logger_config import setup_logger

# Setup logger
logger = setup_logger('model_comparison', 'model_comparison')

class ModelComparison:
    def __init__(self, results_dir='results'):
        """Initialize ModelComparison with results directory"""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results_file = os.path.join(results_dir, 'model_results.json')
        self.results = self._load_results()
    
    def _load_results(self):
        """Load existing results from JSON file"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_results(self):
        """Save results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def add_model_result(self, model_name, metrics, hyperparameters, model_path):
        """Add a new model's results to the comparison."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create result entry
        result = {
            'timestamp': timestamp,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'model_path': model_path
        }
        
        # Add to results
        if model_name not in self.results:
            self.results[model_name] = []
        self.results[model_name].append(result)
        
        # Save updated results
        self._save_results()
        logger.info(f"Added results for {model_name}")
    
    def compare_models(self, metric='val_acc', top_k=5):
        """Compare models based on specified metric"""
        logger.info(f"Comparing models based on {metric}")
        
        # Collect all model results
        all_results = []
        for model_name, results in self.results.items():
            for result in results:
                if metric in result['metrics']:
                    all_results.append({
                        'model_name': model_name,
                        'timestamp': result['timestamp'],
                        'metric_value': result['metrics'][metric],
                        'hyperparameters': result['hyperparameters']
                    })
        
        if not all_results:
            logger.warning(f"No results found for metric {metric}")
            return
        
        # Sort by metric value
        all_results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Get top k results
        top_results = all_results[:top_k]
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(top_results)
        
        # Save comparison to CSV
        comparison_file = os.path.join(self.results_dir, f'model_comparison_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Saved comparison to {comparison_file}")
        
        # Plot comparison
        self._plot_comparison(comparison_df, metric)
    
    def _plot_comparison(self, comparison_df, metric):
        """Plot model comparison"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model_name', y='metric_value', data=comparison_df)
        plt.title(f'Model Comparison - {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.results_dir, f'model_comparison_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Saved comparison plot to {plot_file}")
    
    def get_best_model(self, metric='val_acc'):
        """Get the best model based on specified metric"""
        best_result = None
        best_value = float('-inf')
        
        for model_name, results in self.results.items():
            for result in results:
                if metric in result['metrics'] and result['metrics'][metric] > best_value:
                    best_value = result['metrics'][metric]
                    best_result = {
                        'model_name': model_name,
                        'timestamp': result['timestamp'],
                        'metrics': result['metrics'],
                        'hyperparameters': result['hyperparameters'],
                        'model_path': result['model_path']
                    }
        
        if best_result:
            logger.info(f"Best model: {best_result['model_name']} with {metric}={best_value:.4f}")
            return best_result
        else:
            logger.warning(f"No results found for metric {metric}")
            return None
    
    def plot_training_history(self, model_name, metric='loss'):
        """Plot training history for a specific model"""
        if model_name not in self.results:
            logger.warning(f"No results found for model {model_name}")
            return
        
        # Get the most recent result for the model
        result = self.results[model_name][-1]
        
        if metric not in result['metrics']:
            logger.warning(f"Metric {metric} not found in results")
            return
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(result['metrics'][metric])
        plt.title(f'Training History - {model_name} - {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(self.results_dir, f'training_history_{model_name}_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Saved training history plot to {plot_file}")

if __name__ == '__main__':
    # Example usage
    comparison = ModelComparison()
    
    # Add some example results
    example_metrics = {
        'val_acc': 0.95,
        'val_loss': 0.1,
        'train_acc': 0.97,
        'train_loss': 0.08
    }
    
    example_hyperparameters = {
        'input_size': 63,
        'hidden_size': 64,
        'num_layers': 2
    }
    
    comparison.add_model_result(
        'ExampleModel',
        example_metrics,
        example_hyperparameters,
        'models/example_model.pth'
    )
    
    # Compare models
    comparison.compare_models('val_acc')
    
    # Get best model
    best_model = comparison.get_best_model('val_acc')
    if best_model:
        print(f"Best model: {best_model['model_name']}")
        print(f"Metrics: {best_model['metrics']}")
    
    # Plot training history
    comparison.plot_training_history('ExampleModel', 'loss') 