@echo off
setlocal
cd /d "%~dp0"
echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Next: run_train.bat then run_app.bat
endlocal
