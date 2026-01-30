@echo off
echo ================================================
echo  Mining Claim Redistribution Dashboard
echo  Starting Streamlit Application...
echo ================================================
echo.

cd /d "%~dp0"

echo Checking for Streamlit installation...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting application...
echo.
echo The app will open in your default browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo ================================================
echo.

streamlit run LVA_Analysis_Streamlit.py

pause
