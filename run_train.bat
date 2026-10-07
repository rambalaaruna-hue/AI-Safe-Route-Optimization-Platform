@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment missing. Run run_setup.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
if not exist "real_accident_data.csv" (
  echo Dataset not found. Generating real_accident_data.csv ...
  python generate_dataset.py
)
echo Training XGBoost risk model...
python train_model.py
endlocal
