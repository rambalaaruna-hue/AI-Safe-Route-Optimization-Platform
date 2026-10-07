Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment missing. Run .\run_setup.ps1 first."
}
if (-not (Test-Path ".\models\accident_model.pkl")) {
    throw "Model missing. Run .\run_train.ps1 first."
}
Write-Host "Starting Streamlit on http://localhost:8501"
& .\.venv\Scripts\streamlit.exe run app.py --server.headless true
