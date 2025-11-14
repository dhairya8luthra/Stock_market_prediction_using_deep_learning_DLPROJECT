"""
Comprehensive Experiment Runner for Stock Market Prediction
Runs all experiments and generates comparison reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import custom modules
from data_preprocessing import StockDataPreprocessor, DataAugmentation
from improved_models import get_model


class ExperimentRunner:
    """
    Run comprehensive experiments on stock market prediction
    """
    
    def __init__(self, results_dir='./results/dl_assignment'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {}
        
    def load_nifty_stocks(self, stock_names, nifty_folder='./NIFTY'):
        """Load selected NIFTY stocks"""
        stock_data = {}
        for stock in stock_names:
            file_path = os.path.join(nifty_folder, f'{stock}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                stock_data[stock] = df
                print(f"Loaded {stock}: {len(df)} records")
            else:
                print(f"Warning: {file_path} not found")
        return stock_data
    
    def run_single_model_experiment(self, model_type, X_train, y_train, X_val, y_val,
                                   X_test, y_test, epochs=50, batch_size=32):
        """
        Run experiment for a single model
        """
        print(f"\\n{'='*60}")
        print(f"Training {model_type} Model")
        print(f"{'='*60}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = get_model(model_type, input_shape)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            ModelCheckpoint(
                os.path.join(self.results_dir, f'{model_type}_best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_val = model.predict(X_val, verbose=0).flatten()
        y_pred_test = model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        results = {
            'model_type': model_type,
            'train_metrics': {
                'mse': float(mean_squared_error(y_train, y_pred_train)),
                'mae': float(mean_absolute_error(y_train, y_pred_train)),
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'r2': float(r2_score(y_train, y_pred_train))
            },
            'val_metrics': {
                'mse': float(mean_squared_error(y_val, y_pred_val)),
                'mae': float(mean_absolute_error(y_val, y_pred_val)),
                'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
                'r2': float(r2_score(y_val, y_pred_val))
            },
            'test_metrics': {
                'mse': float(test_mse),
                'mae': float(test_mae),
                'rmse': float(np.sqrt(test_mse)),
                'r2': float(r2_score(y_test, y_pred_test))
            },
            'history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']]
            },
            'predictions': {
                'y_test': y_test.tolist(),
                'y_pred_test': y_pred_test.tolist()
            },
            'model_params': model.count_params()
        }
        
        # Save results
        with open(os.path.join(self.results_dir, f'{model_type}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save model
        model.save(os.path.join(self.results_dir, f'{model_type}_final_model.h5'))
        
        return results
    
    def plot_training_history(self, results_dict):
        """Plot training history for all models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for model_name, results in results_dict.items():
            history = results['history']
            epochs = range(1, len(history['loss']) + 1)
            
            # Loss plot
            axes[0].plot(epochs, history['loss'], label=f'{model_name} (train)', alpha=0.7)
            axes[0].plot(epochs, history['val_loss'], label=f'{model_name} (val)', linestyle='--', alpha=0.7)
            
            # MAE plot
            axes[1].plot(epochs, history['mae'], label=f'{model_name} (train)', alpha=0.7)
            axes[1].plot(epochs, history['val_mae'], label=f'{model_name} (val)', linestyle='--', alpha=0.7)
        
        axes[0].set_title('Training and Validation Loss', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Training and Validation MAE', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_comparison(self, results_dict):
        """Plot prediction comparisons"""
        fig, axes = plt.subplots(len(results_dict), 1, figsize=(15, 4 * len(results_dict)))
        
        if len(results_dict) == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            y_test = np.array(results['predictions']['y_test'])
            y_pred = np.array(results['predictions']['y_pred_test'])
            
            # Plot first 200 predictions
            sample_size = min(200, len(y_test))
            axes[idx].plot(y_test[:sample_size], label='Actual', alpha=0.7, linewidth=2)
            axes[idx].plot(y_pred[:sample_size], label='Predicted', alpha=0.7, linewidth=2)
            axes[idx].set_title(f'{model_name} - Predictions vs Actual', fontweight='bold')
            axes[idx].set_xlabel('Sample Index')
            axes[idx].set_ylabel('Price Change (%)')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self, results_dict):
        """Create comparison table of all models"""
        data = []
        for model_name, results in results_dict.items():
            row = {
                'Model': model_name,
                'Parameters': f"{results['model_params']:,}",
                'Train MAE': f"{results['train_metrics']['mae']:.6f}",
                'Val MAE': f"{results['val_metrics']['mae']:.6f}",
                'Test MAE': f"{results['test_metrics']['mae']:.6f}",
                'Train R²': f"{results['train_metrics']['r2']:.4f}",
                'Val R²': f"{results['val_metrics']['r2']:.4f}",
                'Test R²': f"{results['test_metrics']['r2']:.4f}",
                'Test RMSE': f"{results['test_metrics']['rmse']:.6f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)
        
        # Create styled table
        print("\\n" + "="*120)
        print("MODEL COMPARISON TABLE")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        return df
    
    def generate_report(self, results_dict):
        """Generate comprehensive report"""
        report_path = os.path.join(self.results_dir, 'experiment_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\\n")
            f.write("DEEP LEARNING ASSIGNMENT - STOCK MARKET PREDICTION\\n")
            f.write("Improved Models Experiment Report\\n")
            f.write("="*80 + "\\n\\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Number of Models Tested: {len(results_dict)}\\n\\n")
            
            f.write("IMPROVEMENTS IMPLEMENTED:\\n")
            f.write("-"*80 + "\\n")
            f.write("1. Advanced Feature Engineering (45+ technical indicators)\\n")
            f.write("2. Multi-head Attention Mechanisms\\n")
            f.write("3. Residual Connections for Better Gradient Flow\\n")
            f.write("4. Transformer-based Architecture\\n")
            f.write("5. Batch Normalization and Layer Normalization\\n")
            f.write("6. Multi-Stock Learning\\n")
            f.write("7. Data Augmentation\\n")
            f.write("8. Advanced Regularization (L2, Dropout)\\n\\n")
            
            for model_name, results in results_dict.items():
                f.write("="*80 + "\\n")
                f.write(f"MODEL: {model_name}\\n")
                f.write("="*80 + "\\n")
                f.write(f"Parameters: {results['model_params']:,}\\n\\n")
                
                f.write("TEST SET PERFORMANCE:\\n")
                f.write("-"*80 + "\\n")
                for metric, value in results['test_metrics'].items():
                    f.write(f"{metric.upper():15s}: {value:.6f}\\n")
                f.write("\\n")
            
            f.write("="*80 + "\\n")
            f.write("END OF REPORT\\n")
            f.write("="*80 + "\\n")
        
        print(f"\\nReport saved to: {report_path}")


def main():
    """Main experiment execution"""
    print("="*80)
    print("DEEP LEARNING ASSIGNMENT - IMPROVED STOCK MARKET PREDICTION")
    print("="*80)
    
    # Configuration
    SELECTED_STOCKS = ['TCS', 'RELIANCE', 'INFY', 'HDFCBANK', 'SBIN']
    WINDOW_SIZE = 60
    MODELS_TO_TEST = ['attention', 'residual', 'transformer']
    
    # Initialize
    runner = ExperimentRunner()
    
    # Load data
    print("\\nLoading NIFTY stock data...")
    stock_data = runner.load_nifty_stocks(SELECTED_STOCKS)
    
    # Prepare data
    print("\\nPreparing data...")
    preprocessor = StockDataPreprocessor(window_size=WINDOW_SIZE)
    X, y, stock_indices = preprocessor.create_multi_stock_dataset(stock_data)
    
    print(f"\\nTotal sequences created: {len(X)}")
    print(f"Feature dimension: {X.shape}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, shuffle=False)
    
    print(f"\\nData splits:")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    
    # Run experiments
    all_results = {}
    for model_type in MODELS_TO_TEST:
        try:
            results = runner.run_single_model_experiment(
                model_type, X_train, y_train, X_val, y_val, X_test, y_test,
                epochs=40, batch_size=32
            )
            all_results[model_type] = results
        except Exception as e:
            print(f"\\nError training {model_type}: {e}")
    
    # Generate visualizations and reports
    if all_results:
        print("\\n" + "="*80)
        print("GENERATING COMPARISON REPORTS")
        print("="*80)
        
        runner.plot_training_history(all_results)
        runner.plot_predictions_comparison(all_results)
        runner.create_comparison_table(all_results)
        runner.generate_report(all_results)
        
        print("\\n✓ All experiments completed successfully!")
        print(f"✓ Results saved to: {runner.results_dir}")


if __name__ == "__main__":
    main()
