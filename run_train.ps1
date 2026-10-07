Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment missing. Run .\run_setup.ps1 first."
}
if (-not (Test-Path ".\real_accident_data.csv")) {
    Write-Host "Dataset not found. Generating real_accident_data.csv ..."
    & .\.venv\Scripts\python.exe generate_dataset.py
}
Write-Host "Training XGBoost risk model..."
& .\.venv\Scripts\python.exe train_model.py
