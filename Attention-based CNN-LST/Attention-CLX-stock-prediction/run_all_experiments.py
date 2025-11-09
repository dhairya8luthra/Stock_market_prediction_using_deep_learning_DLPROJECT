"""
Script to run all experiments from the research paper and save results
"""
import os
import sys
import json
import time
from datetime import datetime

# Create results directory
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# Create subdirectories for different experiments
os.makedirs(f"{results_dir}/arima", exist_ok=True)
os.makedirs(f"{results_dir}/lstm", exist_ok=True)
os.makedirs(f"{results_dir}/xgboost", exist_ok=True)
os.makedirs(f"{results_dir}/hybrid_model", exist_ok=True)
os.makedirs(f"{results_dir}/plots", exist_ok=True)

def log_experiment(experiment_name, message):
    """Log experiment progress"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{experiment_name}] {message}"
    print(log_msg)
    with open(f"{results_dir}/experiment_log.txt", "a") as f:
        f.write(log_msg + "\n")

def run_arima():
    """Run ARIMA preprocessing experiment"""
    log_experiment("ARIMA", "Starting ARIMA preprocessing...")
    try:
        import ARIMA
        log_experiment("ARIMA", "ARIMA preprocessing completed successfully")
        return True
    except Exception as e:
        log_experiment("ARIMA", f"Error: {str(e)}")
        return False

def run_lstm():
    """Run LSTM experiments"""
    log_experiment("LSTM", "Starting LSTM experiments...")
    try:
        import LSTM
        log_experiment("LSTM", "LSTM experiments completed successfully")
        return True
    except Exception as e:
        log_experiment("LSTM", f"Error: {str(e)}")
        return False

def run_xgboost():
    """Run XGBoost experiment"""
    log_experiment("XGBoost", "Starting XGBoost experiment...")
    try:
        import XGBoost
        log_experiment("XGBoost", "XGBoost experiment completed successfully")
        return True
    except Exception as e:
        log_experiment("XGBoost", f"Error: {str(e)}")
        return False

def run_hybrid_model():
    """Run Hybrid Attention-based CNN-LSTM model"""
    log_experiment("Hybrid", "Starting Hybrid Model experiment...")
    try:
        import Main
        log_experiment("Hybrid", "Hybrid Model experiment completed successfully")
        return True
    except Exception as e:
        log_experiment("Hybrid", f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("STOCK PREDICTION EXPERIMENTS - RESEARCH PAPER IMPLEMENTATION")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: ARIMA Preprocessing
    print("\n[STEP 1/4] Running ARIMA Preprocessing...")
    arima_success = run_arima()
    
    if arima_success:
        # Step 2: LSTM Models
        print("\n[STEP 2/4] Running LSTM Models...")
        lstm_success = run_lstm()
        
        # Step 3: XGBoost Model
        print("\n[STEP 3/4] Running XGBoost Model...")
        xgboost_success = run_xgboost()
        
        # Step 4: Hybrid Attention-based CNN-LSTM Model
        print("\n[STEP 4/4] Running Hybrid Attention-based CNN-LSTM Model...")
        hybrid_success = run_hybrid_model()
        
        # Summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"ARIMA Preprocessing: {'✓ SUCCESS' if arima_success else '✗ FAILED'}")
        print(f"LSTM Models: {'✓ SUCCESS' if lstm_success else '✗ FAILED'}")
        print(f"XGBoost Model: {'✓ SUCCESS' if xgboost_success else '✗ FAILED'}")
        print(f"Hybrid Model: {'✓ SUCCESS' if hybrid_success else '✗ FAILED'}")
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
        print(f"\nResults saved in: {os.path.abspath(results_dir)}")
        print("="*80)
    else:
        print("\n[ERROR] ARIMA preprocessing failed. Cannot proceed with other experiments.")
        print("Please check the error log in results/experiment_log.txt")
