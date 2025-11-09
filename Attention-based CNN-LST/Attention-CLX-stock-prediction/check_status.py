"""
Quick status checker for experiment progress
"""
import os
from pathlib import Path

print("="*80)
print("EXPERIMENT STATUS CHECKER")
print("="*80)

results_dir = Path("./results")

# Check log file
log_file = results_dir / "experiment_log.txt"
if log_file.exists():
    print("\n📋 Latest Log Entries:")
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[-5:]:  # Last 5 lines
            print(f"  {line.strip()}")
else:
    print("\n⚠ No log file found yet")

# Check ARIMA results
print("\n" + "="*80)
print("STEP 1/4: ARIMA Preprocessing")
print("="*80)
arima_dir = results_dir / "arima"
if arima_dir.exists():
    files = list(arima_dir.glob("*"))
    if files:
        print(f"✅ {len(files)} files generated:")
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name} ({size:.1f} KB)")
    else:
        print("🔄 In progress... (no files yet)")
else:
    print("⏳ Not started")

# Check LSTM results
print("\n" + "="*80)
print("STEP 2/4: LSTM Models")
print("="*80)
lstm_dir = results_dir / "lstm"
if lstm_dir.exists():
    files = list(lstm_dir.glob("*"))
    if files:
        print(f"✅ {len(files)} files generated:")
        for f in files:
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")
    else:
        print("🔄 In progress... (no files yet)")
else:
    print("⏳ Not started")

# Check XGBoost results
print("\n" + "="*80)
print("STEP 3/4: XGBoost Model")
print("="*80)
xgboost_dir = results_dir / "xgboost"
if xgboost_dir.exists():
    files = list(xgboost_dir.glob("*"))
    if files:
        print(f"✅ {len(files)} files generated:")
        for f in files:
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")
    else:
        print("🔄 In progress... (no files yet)")
else:
    print("⏳ Not started")

# Check Hybrid Model results
print("\n" + "="*80)
print("STEP 4/4: Hybrid Attention CNN-LSTM Model")
print("="*80)
hybrid_dir = results_dir / "hybrid_model"
if hybrid_dir.exists():
    files = list(hybrid_dir.glob("*"))
    if files:
        print(f"✅ {len(files)} files generated:")
        for f in files:
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")
    else:
        print("🔄 In progress... (no files yet)")
else:
    print("⏳ Not started")

# Overall summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

completed = 0
total = 4

if (arima_dir / "metrics.txt").exists():
    completed += 1
    print("✅ ARIMA: Complete")
else:
    print("🔄 ARIMA: Running or pending")

if (lstm_dir / "metrics.txt").exists():
    completed += 1
    print("✅ LSTM: Complete")
else:
    print("⏳ LSTM: Pending")

if (xgboost_dir / "metrics.txt").exists():
    completed += 1
    print("✅ XGBoost: Complete")
else:
    print("⏳ XGBoost: Pending")

if (hybrid_dir / "metrics.txt").exists():
    completed += 1
    print("✅ Hybrid Model: Complete")
else:
    print("⏳ Hybrid Model: Pending")

print(f"\nProgress: {completed}/{total} experiments completed ({completed/total*100:.0f}%)")

if completed == total:
    print("\n🎉 ALL EXPERIMENTS COMPLETED! 🎉")
    print("Results are available in the ./results/ directory")
else:
    print(f"\n⏳ Experiments in progress... ({total-completed} remaining)")

print("="*80)
