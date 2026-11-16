@echo off
title Fake news demo (Streamlit)
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [ERROR] venv not found. See run_train.bat first.
    pause
    exit /b 1
)

if not exist lstm_fake_news.keras (
    echo [ERROR] lstm_fake_news.keras not found.
    echo Run run_train.bat first to train the model.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
streamlit run app.py
pause
