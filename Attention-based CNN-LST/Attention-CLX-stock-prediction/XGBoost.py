import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from model import walk_forward_validation

# Create results directory
os.makedirs("./results/xgboost", exist_ok=True)
plt.ioff()  # Turn off interactive mode

data = pd.read_csv('./601988.SH.csv')
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]
# data = pd.DataFrame(data, dtype=np.float64)
close = data.pop('close')
data.insert(5, 'close', close)

residuals = pd.read_csv('./ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')
merge_data = pd.merge(data, residuals, on='trade_date')
#merge_data = merge_data.drop(labels='2007-01-04', axis=0)

Lt = pd.read_csv('./ARIMA.csv')
Lt = Lt.drop('trade_date', axis=1)
Lt = np.array(Lt)
Lt = Lt.flatten().tolist()

train, test = prepare_data(merge_data, n_test=180, n_in=6, n_out=1)

y, yhat = walk_forward_validation(train, test)

# Ensure we have exactly n_test predictions (180)
# walk_forward returns 181 values, trim to match n_test
if len(y) == 181:
    y = y[:180]
    yhat = yhat[:180]

# Get time index for the test period (last 180 data points)
time = pd.Series(data.index[3501:])
time = time[-len(y):]  # Match length with predictions

# Get actual stock prices for the test period
data1 = data.iloc[3501:, 5]
data1 = data1[-len(y):]  # Match length with predictions

# Get ARIMA predictions for the test period
Lt = Lt[-len(y):]  # Match length with predictions

# Debug: print lengths
print(f"\nDebug - Array lengths:")
print(f"time: {len(time)}, y: {len(y)}, yhat: {len(yhat)}")
print(f"data1: {len(data1)}, Lt: {len(Lt)}")

# Save residuals predictions
residuals_pred_df = pd.DataFrame({
    'time': time.values,  # Convert Series to numpy array
    'y_residuals': y,
    'yhat_residuals': yhat
})
residuals_pred_df.to_csv('./results/xgboost/residuals_predictions.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(time, y, label='Residuals')
plt.plot(time, yhat, label='Predicted Residuals')
plt.title('ARIMA+XGBoost: Residuals Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Residuals', fontsize=14, horizontalalignment='center')
plt.legend()
plt.savefig('./results/xgboost/residuals_prediction.png', dpi=300, bbox_inches='tight')
plt.close()

finalpredicted_stock_price = [i + j for i, j in zip(Lt, yhat)]
#print('final', finalpredicted_stock_price)

# Save final predictions
final_pred_df = pd.DataFrame({
    'time': time.values,  # Convert Series to numpy array
    'actual': data1.values,  # Convert Series to numpy array
    'predicted': finalpredicted_stock_price
})
final_pred_df.to_csv('./results/xgboost/stock_price_predictions.csv', index=False)

print("\n=== ARIMA+XGBoost Model Evaluation Metrics ===")
evaluation_metric(data1, finalpredicted_stock_price)

# Save metrics
with open('./results/xgboost/metrics.txt', 'w') as f:
    f.write("ARIMA+XGBoost Model Evaluation Metrics\n")
    f.write("="*50 + "\n")
    from sklearn import metrics
    MSE = metrics.mean_squared_error(data1, finalpredicted_stock_price)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(data1, finalpredicted_stock_price)
    R2 = metrics.r2_score(data1, finalpredicted_stock_price)
    f.write(f'MSE: {MSE:.5f}\n')
    f.write(f'RMSE: {RMSE:.5f}\n')
    f.write(f'MAE: {MAE:.5f}\n')
    f.write(f'R2: {R2:.5f}\n')

plt.figure(figsize=(10, 6))
plt.plot(time, data1, label='Stock Price')
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
plt.title('ARIMA+XGBoost: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.savefig('./results/xgboost/stock_price_prediction.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== XGBoost Experiment Completed ===")
print("Results saved in ./results/xgboost/")
