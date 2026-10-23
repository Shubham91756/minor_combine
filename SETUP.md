# Environment Setup Guide

This guide will help you set up the Python environment for the Parkinsons ML Detection project.

## Prerequisites

- **Python 3.11+** (tested on Python 3.13)
- **pip** (comes with Python)
- **Git** (for version control)
- **4GB+ free disk space** (for dependencies, especially TensorFlow)

## Quick Setup (Automated)

### macOS / Linux
Run the setup script to automatically create and configure your environment:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This script will:
1. Create a new virtual environment (`env/`)
2. Upgrade pip
3. Install all dependencies from `requirements-all.txt`
4. Display quick start instructions

### Windows (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_env.ps1
```

## Manual Setup

If you prefer to set up manually or the script fails:

### Step 1: Create Virtual Environment
```bash
python3 -m venv env
```

### Step 2: Activate Environment

**macOS / Linux:**
```bash
source env/bin/activate
```

**Windows (PowerShell):**
```powershell
.\env\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
env\Scripts\activate.bat
```

### Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
pip install -r requirements-all.txt
```

**⏱️ This step will take 5-15 minutes depending on your internet speed and CPU.**

## Verify Installation

After installation completes, verify everything is working:

```bash
# Test imports
python -c "import tensorflow; import fastapi; import sklearn; print('✓ All imports successful')"
```

You should see: `✓ All imports successful`

## Running the Project

### 1. Backend API Server

**Terminal 1:**
```bash
source env/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Frontend Development Server

**Terminal 2:**
```bash
cd frontend
npm install  # (only needed first time)
npm start
```

The frontend will open at `http://localhost:3000`

### 3. Training Script

**Terminal 3:**
```bash
source env/bin/activate
cd models
python trainers/train_fixed.py
```

## Requirements Overview

### Core Dependencies
- **FastAPI 0.95.2** - Backend API framework
- **Uvicorn 0.22.0** - ASGI server
- **NumPy ≥1.24.0** - Numerical computing
- **Pandas ≥2.2.0** - Data manipulation (Python 3.13+ compatible)
- **Scikit-learn ≥1.3.0** - Machine learning models

### Heavy Dependencies (Optional but Required for Training)
- **TensorFlow 2.14.1** - Deep learning framework
- **Keras 2.14.0** - Neural network API
- **Pillow 10.0.0** - Image processing

### Development Dependencies
- **Matplotlib 3.10.7** - Data visualization
- **tqdm 4.67.1** - Progress bars
- **imbalanced-learn 0.11.0** - Imbalanced dataset handling

## Troubleshooting

### Issue: "Command not found: python3"
**Solution:** Install Python from https://www.python.org/downloads/

### Issue: "pip: command not found"
**Solution:** Use `python3 -m pip` instead of `pip`

### Issue: Permission denied when running setup_env.sh
**Solution:**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### Issue: TensorFlow installation fails
**Solution:** TensorFlow compilation can take time. Be patient (5-15 minutes) or install a pre-built wheel:
```bash
pip install tensorflow --only-binary=:all:
```

### Issue: "ModuleNotFoundError" when running backend
**Solution:** Make sure you've activated the virtual environment:
```bash
source env/bin/activate  # macOS/Linux
# or
.\env\Scripts\Activate.ps1  # Windows PowerShell
```

### Issue: Port 8000 or 3000 already in use
**Solution:** Use different ports:
```bash
# Backend on port 8001
python -m uvicorn backend.main:app --port 8001 --reload

# Frontend on port 3001
REACT_APP_API_URL=http://localhost:8001 npm start
```

## Project Structure

```
ML_mini_project/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── services/
│   │   ├── handwriting_service.py  # ← Updated to use new model
│   │   ├── motor_service.py
│   │   └── gait_service.py
│   └── requirements.txt
├── frontend/                   # React application
│   ├── src/
│   │   └── App.js             # Main frontend (makes API calls)
│   └── package.json
├── models/
│   ├── trainers/
│   │   ├── train_fixed.py     # ← New training script
│   │   └── ...other trainers
│   ├── trained_models/
│   │   └── handwriting_efficient_best_long/
│   │       └── new_trained_models/
│   │           ├── parkinson_model_best.h5    # ← New model
│   │           └── handwriting_model.h5
│   └── requirements.txt
└── requirements-all.txt        # ← Consolidated requirements

```

## Environment Variables

### Frontend
```bash
# In frontend/ or when running npm start:
REACT_APP_API_URL=http://localhost:8000
```

## Next Steps

1. ✅ Run setup script
2. ✅ Start backend server
3. ✅ Start frontend
4. ✅ Open http://localhost:3000 in browser
5. ✅ Upload handwriting images for predictions

## Getting Help

- Check logs in `backend/error_debug.log`
- Verify model path: `models/trained_models/handwriting_efficient_best_long/new_trained_models/parkinson_model_best.h5`
- Ensure frontend can reach backend: Open browser DevTools → Network tab → submit a prediction
