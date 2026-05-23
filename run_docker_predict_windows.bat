@echo off
if "%~1"=="" (
    echo Usage: run_docker_predict_windows.bat data\filtered\5\5_A_0.wav
    exit /b 1
)

docker run --rm -v "%cd%\data:/app/data" -v "%cd%\outputs\predictions:/app/outputs/predictions" spoken-digit-recognition:windows-inference --audio "%~1"
