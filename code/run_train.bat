@echo off
title Train fake news model
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [ERROR] venv not found. Run these commands first:
    echo    py -3.12 -m venv venv
    echo    venv\Scripts\pip install -r requirements.txt tensorflow
    pause
    exit /b 1
)

if not exist data\Fake.csv (
    echo [ERROR] data\Fake.csv not found.
    echo Download ISOT dataset from Kaggle and put Fake.csv + True.csv into data\
    echo https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    pause
    exit /b 1
)

echo Activating venv and starting training...
call venv\Scripts\activate.bat
python train_and_evaluate.py
pause
