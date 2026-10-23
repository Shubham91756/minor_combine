<#
Setup script for Windows PowerShell. Creates a .venv, activates it for the session, upgrades pip and installs requirements-all.txt

Usage (run from repository root):
.\setup_env.ps1

If you get execution policy issues, run PowerShell as administrator once and allow RemoteSigned or use the -Scope Process workaround documented in ENV_SETUP.md
#>

Write-Host 'Creating virtual environment .venv (Python must be on PATH)'
python -m venv .venv

Write-Host 'Activating virtual environment for this session'
. .\.venv\Scripts\Activate.ps1

Write-Host 'Upgrading pip, setuptools and wheel'
python -m pip install --upgrade pip setuptools wheel

Write-Host 'Installing consolidated requirements (requirements-all.txt)'
Write-Host 'Note: requirements-all.txt includes optional heavy packages (TensorFlow, Keras, shap).'
Write-Host 'If you do not need them, edit requirements-all.txt to remove heavy deps before installing.'
python -m pip install -r requirements-all.txt

Write-Host 'Done. To run backend with the new model (optional):'
Write-Host "$env:PYTHONPATH='.'; $env:GAIT_MODEL_PATH='models/trained_models/gait_signals/model_gait_enhanced_custom.joblib'; .venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
