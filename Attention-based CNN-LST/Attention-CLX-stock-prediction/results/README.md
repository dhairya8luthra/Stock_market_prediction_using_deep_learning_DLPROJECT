# Experiment Results

This directory contains all the results from the stock prediction experiments based on the research paper implementation.

## Directory Structure

```
results/
├── arima/                  # ARIMA preprocessing results
│   ├── close_price.png    # Training and test set visualization
│   ├── first_order_diff.png
│   ├── second_order_diff.png
│   ├── ARIMA_prediction.png
│   ├── ARIMA_predictions.csv
│   ├── ARIMA_residuals.csv
│   ├── residuals_analysis.png
│   ├── diff_fit.png
│   └── metrics.txt        # Evaluation metrics
│
├── lstm/                   # LSTM model results
│   ├── residuals_training_validation_loss.png
│   ├── lstm_training_validation_loss.png
│   ├── residuals_training_history.csv
│   ├── lstm_training_history.csv
│   ├── bilstm_stock_price_prediction.png
│   ├── lstm_stock_price_prediction.png
│   ├── bilstm_predictions.csv
│   ├── lstm_predictions.csv
│   └── metrics.txt        # Evaluation metrics
│
├── xgboost/               # XGBoost model results
│   ├── residuals_prediction.png
│   ├── residuals_predictions.csv
│   ├── stock_price_prediction.png
│   ├── stock_price_predictions.csv
│   └── metrics.txt        # Evaluation metrics
│
├── hybrid_model/          # Hybrid Attention-based CNN-LSTM model results
│   ├── training_validation_loss.png
│   ├── training_history.csv
│   ├── prediction_plot.png
│   ├── predictions.csv
│   ├── stock_model.h5    # Saved model
│   ├── stock_normalize.npy
│   └── metrics.txt        # Evaluation metrics
│
└── experiment_log.txt     # Complete experiment execution log
```

## Experiment Workflow

1. **ARIMA Preprocessing**: Time series analysis and residual extraction
2. **LSTM Models**: Single-layer, multi-layer, and bidirectional LSTM experiments
3. **XGBoost Model**: ARIMA + XGBoost hybrid approach
4. **Hybrid Model**: Attention-based CNN-LSTM model (proposed approach)

## Evaluation Metrics

All experiments are evaluated using:

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared Score)

## Files Description

### ARIMA Results

- `ARIMA_predictions.csv`: ARIMA model predictions on test set
- `ARIMA_residuals.csv`: Extracted residuals for hybrid models
- `metrics.txt`: Model performance metrics

### LSTM Results

- `bilstm_predictions.csv`: BiLSTM model predictions
- `lstm_predictions.csv`: LSTM model predictions
- `training_history.csv`: Training and validation loss per epoch
- `metrics.txt`: Model performance metrics

### XGBoost Results

- `stock_price_predictions.csv`: Final stock price predictions
- `residuals_predictions.csv`: Residual predictions
- `metrics.txt`: Model performance metrics

### Hybrid Model Results

- `predictions.csv`: Model predictions vs actual values
- `stock_model.h5`: Trained Keras model (can be loaded for inference)
- `training_history.csv`: Training and validation loss per epoch
- `metrics.txt`: Model performance metrics

## How to Use These Results

### View Plots

All plots are saved as high-resolution PNG images (300 DPI) and can be opened with any image viewer.

### Analyze Predictions

Prediction CSVs can be loaded in Python/Excel for further analysis:

```python
import pandas as pd

# Load predictions
predictions = pd.read_csv('results/hybrid_model/predictions.csv')

# Analyze
print(predictions.describe())
```

### Load Trained Model

The hybrid model can be reloaded for inference:

```python
from keras.models import load_model
import numpy as np

# Load model
model = load_model('results/hybrid_model/stock_model.h5')
normalize = np.load('results/hybrid_model/stock_normalize.npy')

# Make predictions
# predictions = model.predict(new_data)
```

## Notes

- All plots are saved in non-interactive mode to prevent blocking
- Training logs are preserved in experiment_log.txt
- Intermediate files (ARIMA.csv, ARIMA_residuals1.csv) are also saved in the root directory for compatibility with the original code

## Citation

If you use these results, please cite the original research paper and acknowledge the implementation.
