# Stock Prediction Experiments - Execution Guide

## Overview
This repository contains the implementation of a research paper on stock prediction using various machine learning approaches including ARIMA, LSTM, XGBoost, and a novel Hybrid Attention-based CNN-LSTM model.

## Environment Setup

### Virtual Environment Path
```
D:\CodingPlayground\Python\Deep_learning_proj\env_dl
```

### Activation Commands

**PowerShell:**
```powershell
D:\CodingPlayground\Python\Deep_learning_proj\env_dl\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
D:\CodingPlayground\Python\Deep_learning_proj\env_dl\Scripts\activate.bat
```

## Running Experiments

### Method 1: Using the Batch File (Recommended for Windows)
```cmd
run_experiments.bat
```

### Method 2: Using Python Directly
```powershell
# Activate environment first
D:\CodingPlayground\Python\Deep_learning_proj\env_dl\Scripts\Activate.ps1

# Run all experiments
python run_all_experiments.py
```

### Method 3: Quick Run
```powershell
python quick_run.py
```

### Method 4: Individual Experiments
Run experiments in this specific order:

```powershell
# 1. ARIMA preprocessing (MUST run first)
python ARIMA.py

# 2. LSTM models (requires ARIMA residuals)
python LSTM.py

# 3. XGBoost model (requires ARIMA predictions)
python XGBoost.py

# 4. Hybrid Attention-based CNN-LSTM model (requires ARIMA residuals)
python Main.py
```

## Experiment Details

### 1. ARIMA Preprocessing (`ARIMA.py`)
- **Purpose**: Time series analysis and residual extraction
- **Duration**: ~5-10 minutes
- **Outputs**:
  - `ARIMA.csv` - ARIMA predictions
  - `ARIMA_residuals1.csv` - Residuals for hybrid models
  - Various plots in `results/arima/`
  
### 2. LSTM Models (`LSTM.py`)
- **Models**: Single-layer, Multi-layer, and Bidirectional LSTM
- **Duration**: ~10-15 minutes
- **Epochs**: 50
- **Batch Size**: 32
- **Outputs**: Training history, predictions, and plots in `results/lstm/`

### 3. XGBoost Model (`XGBoost.py`)
- **Purpose**: Hybrid ARIMA + XGBoost approach
- **Duration**: ~5 minutes
- **Outputs**: Predictions and plots in `results/xgboost/`

### 4. Hybrid Attention-based CNN-LSTM (`Main.py`)
- **Purpose**: Proposed hybrid model combining CNN, LSTM, and Attention
- **Duration**: ~15-20 minutes
- **Epochs**: 50
- **Batch Size**: 32
- **Outputs**: 
  - Trained model (`stock_model.h5`)
  - Predictions and plots in `results/hybrid_model/`

## Results Structure

```
results/
├── arima/              # ARIMA preprocessing results
├── lstm/               # LSTM model results
├── xgboost/            # XGBoost model results
├── hybrid_model/       # Hybrid model results (main contribution)
├── README.md           # Detailed results documentation
└── experiment_log.txt  # Execution log
```

## Expected Total Runtime
- **Full experiment suite**: 35-50 minutes (depending on hardware)
- GPU acceleration recommended but not required

## Troubleshooting

### Common Issues

**1. NumPy compatibility error**
```
AttributeError: module 'numpy' has no attribute 'float'
```
**Solution**: Already fixed - using `float()` instead of `np.float()`

**2. Environment not activated**
```
ModuleNotFoundError: No module named 'keras'
```
**Solution**: Activate the virtual environment first

**3. Missing data file**
```
FileNotFoundError: 601988.SH.csv
```
**Solution**: Ensure you're in the correct directory

**4. ARIMA not run first**
```
FileNotFoundError: ARIMA_residuals1.csv
```
**Solution**: Run ARIMA.py before other experiments

## Model Performance Metrics

All models are evaluated using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared Score)

See `results/*/metrics.txt` for detailed performance metrics.

## Hardware Requirements

- **Minimum RAM**: 8GB
- **Recommended RAM**: 16GB
- **GPU**: Optional (CUDA-compatible recommended for faster training)
- **Disk Space**: ~500MB for results and models

## Dependencies

```
numpy>=1.16.5
sklearn>=0.21.3
statsmodels>=0.10.1
pandas>=0.25.1
tensorflow>=2.1.0
keras>=2.3.1
xgboost>=1.5.0
matplotlib>=3.1.0
```

## Notes

- All plots are saved automatically (non-interactive mode)
- Models are saved for later inference
- Training history is preserved in CSV format
- Experiment logs are appended to `results/experiment_log.txt`

## Citation

If you use this code or results in your research, please cite the original research paper.

## Contact

For issues or questions, please refer to the original repository or research paper.
