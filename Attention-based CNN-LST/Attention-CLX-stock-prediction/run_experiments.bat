@echo off
echo ================================================================================
echo STOCK PREDICTION EXPERIMENTS - ACTIVATING ENVIRONMENT
echo ================================================================================
echo.

REM Activate the virtual environment
echo Activating environment: D:\CodingPlayground\Python\Deep_learning_proj\env_dl
call D:\CodingPlayground\Python\Deep_learning_proj\env_dl\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate environment
    pause
    exit /b 1
)

echo Environment activated successfully!
echo.

REM Change to project directory
cd /d "d:\CodingPlayground\Python\Deep_learning_proj\Attention-based CNN-LSTM\Attention-CLX-stock-prediction"

echo ================================================================================
echo RUNNING ALL EXPERIMENTS
echo ================================================================================
echo.

REM Run the main experiment script
python run_all_experiments.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Experiments failed with error code %errorlevel%
) else (
    echo.
    echo ================================================================================
    echo ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
    echo ================================================================================
)

echo.
pause
