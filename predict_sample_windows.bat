@echo off
call .venv\Scripts\activate.bat
python scripts\predict_audio.py --audio data\filtered\5\5_A_0.wav
