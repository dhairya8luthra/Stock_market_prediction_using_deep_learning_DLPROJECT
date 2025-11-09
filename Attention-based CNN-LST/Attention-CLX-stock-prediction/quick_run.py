"""
Quick experiment runner - runs experiments sequentially with progress updates
"""
import sys
import os

os.makedirs("./results", exist_ok=True)

print("="*80)
print("Starting Experiment 1/4: ARIMA Preprocessing")
print("="*80)
try:
    exec(open('ARIMA.py').read())
    print("\n✓ ARIMA Preprocessing completed successfully!\n")
except Exception as e:
    print(f"\n✗ ARIMA failed: {e}\n")
    sys.exit(1)

print("="*80)
print("Starting Experiment 2/4: LSTM Models")
print("="*80)
try:
    exec(open('LSTM.py').read())
    print("\n✓ LSTM Models completed successfully!\n")
except Exception as e:
    print(f"\n✗ LSTM failed: {e}\n")

print("="*80)
print("Starting Experiment 3/4: XGBoost Model")
print("="*80)
try:
    exec(open('XGBoost.py').read())
    print("\n✓ XGBoost Model completed successfully!\n")
except Exception as e:
    print(f"\n✗ XGBoost failed: {e}\n")

print("="*80)
print("Starting Experiment 4/4: Hybrid Attention CNN-LSTM Model")
print("="*80)
try:
    exec(open('Main.py').read())
    print("\n✓ Hybrid Model completed successfully!\n")
except Exception as e:
    print(f"\n✗ Hybrid Model failed: {e}\n")

print("="*80)
print("ALL EXPERIMENTS COMPLETED!")
print("Results saved in ./results/ directory")
print("="*80)
