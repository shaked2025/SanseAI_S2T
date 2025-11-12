@echo off
REM Windows batch script to run the Speech-to-Text application

echo ================================================
echo   Real-Time Speech-to-Text System
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import whisper" >nul 2>&1
if errorlevel 1 (
    echo.
    echo It looks like dependencies are not installed.
    echo Would you like to run the installer? (Y/N)
    set /p INSTALL_CHOICE=
    if /i "%INSTALL_CHOICE%"=="Y" (
        python install.py
        echo.
        echo Press any key to start the application...
        pause >nul
    ) else (
        echo.
        echo Please run: python install.py
        echo Then run this script again.
        pause
        exit /b 1
    )
)

REM Run the application
echo Starting application...
echo.
python main.py

REM If the application exits, pause to show any error messages
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)

