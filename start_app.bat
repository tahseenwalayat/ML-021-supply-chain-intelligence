@echo off
setlocal EnableExtensions

rem Launch the local FastAPI service and Streamlit dashboard.
rem Run this file from Explorer or a Command Prompt in the project folder.

cd /d "%~dp0"
set "VENV_DIR=%CD%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "API_BASE_URL=http://127.0.0.1:8000"
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do if /I "%%A"=="API_KEY" set "API_KEY=%%B"
)
if "%API_KEY%"=="" set "API_KEY=sc-key-secret-2026"

if not exist "%PYTHON%" (
    echo Creating a local Python virtual environment...
    py -3 -m venv "%VENV_DIR%" 2>nul
    if errorlevel 1 (
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo.
        echo Python 3 could not be found. Install Python 3.11 or newer, then run this file again.
        pause
        exit /b 1
    )
)

rem A virtual environment can remain after its base Python installation is
rem removed or upgraded. Detect that case before attempting dependency setup.
"%PYTHON%" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo The local virtual environment cannot start because its base Python installation is unavailable.
    echo Install Python 3.11 or 3.12, then delete the .venv folder and run this file again.
    echo The existing .venv is left unchanged so no project files are removed automatically.
    pause
    exit /b 1
)

"%PYTHON%" -c "import fastapi, streamlit, plotly, requests, optuna, pyarrow, yaml" >nul 2>nul
if errorlevel 1 (
    echo Installing project dependencies. This may take a few minutes on the first run...
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo Starting the Supply Chain Intelligence Platform...
start "Supply Chain API" /D "%CD%" cmd /k ""%PYTHON%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo Waiting for the API to become ready...
set /a API_RETRIES=0
:wait_for_api
"%PYTHON%" -c "from urllib.request import urlopen; response = urlopen('http://127.0.0.1:8000/health', timeout=2); assert response.status == 200" >nul 2>nul
if not errorlevel 1 goto api_ready

set /a API_RETRIES+=1
if %API_RETRIES% GEQ 20 goto api_unavailable
timeout /t 1 /nobreak >nul
goto wait_for_api

:api_ready
start "Supply Chain Dashboard" /D "%CD%" cmd /k ""%PYTHON%" -m streamlit run dashboard\app.py --server.address 127.0.0.1 --server.port 8501"

echo.
echo API documentation: http://127.0.0.1:8000/docs
echo Dashboard:         http://127.0.0.1:8501
echo.
echo Two service windows have opened. Close those windows to stop the app.
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8501"
goto end

:api_unavailable
echo.
echo The backend did not respond within 20 seconds.
echo Review the "Supply Chain API" window for the startup error, then run this file again.
pause

:end
endlocal
