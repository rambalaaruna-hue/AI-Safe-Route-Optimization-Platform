@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment missing. Run run_setup.bat first.
  exit /b 1
)
if not exist "models\accident_model.pkl" (
  echo Model missing. Run run_train.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Starting Streamlit on http://localhost:8501
streamlit run app.py --server.headless true
endlocal
