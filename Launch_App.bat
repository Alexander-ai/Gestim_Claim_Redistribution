@echo off
REM Mining Claim Redistribution Dashboard - Enhanced Version
REM Double-click this file to launch the application

echo.
echo =============================================
echo  Mining Claim Redistribution Dashboard v2.0
echo =============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from python.org
    echo.
    pause
    exit /b 1
)

echo Checking dependencies...
echo.

REM Check if required libraries are installed
python -c "import geopandas, pandas, numpy, matplotlib, folium, tqdm, reportlab, PIL" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo One or more required libraries are missing.
    echo.
    echo Installing required libraries...
    echo.
    pip install -r requirements.txt
    echo.
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install dependencies.
        echo.
        echo Please manually run: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo Starting application...
echo.

python Claim_Redistribution_App.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================
    echo ERROR: Application failed to start
    echo ==========================================
    echo.
    echo Please ensure:
    echo 1. Python 3.8+ is installed
    echo 2. All files are in this folder:
    echo    - Claim_Redistribution_App.py
    echo    - Gestim_Claim_Redistribution.py
    echo    - requirements.txt
    echo 3. Dependencies are installed: pip install -r requirements.txt
    echo.
    pause
)
