@echo off
echo.
echo ================================================================================
echo EXPERIMENT PROGRESS MONITOR
echo ================================================================================
echo.
echo Checking experiment progress every 30 seconds...
echo Press Ctrl+C to stop monitoring
echo.

:loop
python check_status.py
echo.
echo [%time%] Waiting 30 seconds before next check...
echo.
timeout /t 30 /nobreak > nul
goto loop
