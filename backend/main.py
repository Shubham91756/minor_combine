from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil

from backend.services.handwriting_service import predict_image, predict_image_xai, ensure_loaded as ensure_handwriting_loaded
from backend.services.motor_service import predict_from_features_dict, ensure_loaded
from backend.services.gait_service import predict_from_timeseries_file
import traceback
import logging
import os
from datetime import datetime
import time
import uuid

# configure simple file logging for debug
LOG_PATH = Path('backend/error_debug.log')
logging.basicConfig(filename=str(LOG_PATH), level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(title='Parkinsons Models API')

# record startup info for traceability
SERVER_PID = os.getpid()
STARTUP_TIME = datetime.utcnow().isoformat()


# ------------------------------------------------------------------
# Motor timeseries session aggregation (in-memory)
# ------------------------------------------------------------------
_MOTOR_SESSIONS = {}  # session_id -> {'created_at': float, 'files': [dict, ...]}
_MOTOR_SESSION_TTL_SEC = 60 * 60  # 1 hour


def _prune_motor_sessions():
    now = time.time()
    dead = [sid for sid, s in _MOTOR_SESSIONS.items() if (now - float(s.get('created_at', now))) > _MOTOR_SESSION_TTL_SEC]
    for sid in dead:
        _MOTOR_SESSIONS.pop(sid, None)


def _motor_session_summary(session_id: str):
    s = _MOTOR_SESSIONS.get(session_id)
    if not s:
        return None
    files = list(s.get('files') or [])
    probs = [f.get('prob_patient') for f in files if isinstance(f.get('prob_patient'), (int, float))]
    prob_mean = (sum(probs) / len(probs)) if probs else None
    pred = 1 if (prob_mean is not None and prob_mean >= 0.5) else 0
    label = 'patient' if pred == 1 else 'healthy'
    return {
        'session_id': session_id,
        'count': len(files),
        'prob_patient': prob_mean,
        'pred': pred,
        'label': label,
        'files': files,
    }
# ------------------------------------------------------------------
# XAI heatmap cache (in-memory, short TTL)
# ------------------------------------------------------------------
_XAI_HEATMAPS = {}  # heatmap_id -> {'created_at': float, 'png': bytes}
_XAI_HEATMAP_TTL_SEC = 10 * 60  # 10 minutes
_XAI_HEATMAP_MAX_ITEMS = 64


def _prune_xai_heatmaps():
    now = time.time()
    dead = [hid for hid, h in _XAI_HEATMAPS.items() if (now - float(h.get('created_at', now))) > _XAI_HEATMAP_TTL_SEC]
    for hid in dead:
        _XAI_HEATMAPS.pop(hid, None)
    # cap size
    if len(_XAI_HEATMAPS) > _XAI_HEATMAP_MAX_ITEMS:
        # drop oldest
        items = sorted(_XAI_HEATMAPS.items(), key=lambda kv: float(kv[1].get('created_at', 0)))
        for hid, _ in items[: max(0, len(_XAI_HEATMAPS) - _XAI_HEATMAP_MAX_ITEMS)]:
            _XAI_HEATMAPS.pop(hid, None)



@app.on_event('startup')
def _preload_models():
    # preload models so first API call is fast and model path errors show early
    try:
        ensure_loaded()
        logging.info("✓ Motor model preloaded")
    except Exception as e:
        logging.error('Failed to preload motor model: %s', e)
    
    try:
        ensure_handwriting_loaded()
        logging.info("✓ Handwriting model preloaded")
    except Exception as e:
        logging.error('Failed to preload handwriting model: %s', e)


# Allow CORS for the frontend during development
# Accept multiple localhost ports since frontend may run on different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:60181",      # Current frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:60181",      # Current frontend port
        "http://localhost:3001",       # Fallback ports
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MotorFeatures(BaseModel):
    features: dict


@app.post('/predict/handwriting')
async def predict_handwriting(file: UploadFile = File(...), explain: bool = False):
    content = await file.read()
    try:
        if explain:
            res, heat_png = predict_image_xai(content)
            if heat_png:
                _prune_xai_heatmaps()
                hid = uuid.uuid4().hex
                _XAI_HEATMAPS[hid] = {'created_at': time.time(), 'png': heat_png}
                # attach relative URL; frontend prefixes the backend base
                try:
                    if isinstance(res, dict) and isinstance(res.get('reasoning'), dict):
                        res['reasoning']['heatmap_url'] = f'/xai/heatmap/{hid}'
                    res['heatmap_url'] = f'/xai/heatmap/{hid}'
                except Exception:
                    pass
            return JSONResponse(res)
        res = predict_image(content)
        return JSONResponse(res)
    except RuntimeError as e:
        # known runtime errors (e.g., missing heavy deps) -> return concise message
        logging.error("RuntimeError in predict_handwriting: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        tb = traceback.format_exc()
        # write traceback to log for inspection
        logging.error("Exception in predict_handwriting: %s", tb)
        # return traceback in JSON for debugging (development only)
        return JSONResponse(status_code=500, content={"detail": tb})


@app.post('/predict/motor')
async def predict_motor_json(body: MotorFeatures):
    try:
        # defensive validation: detect accidental double-wrapped payloads
        features = body.features
        # common mistake: {"features": {"features": { ... }}}
        if isinstance(features, dict) and 'features' in features and isinstance(features['features'], dict):
            msg = (
                "Invalid payload: found nested 'features.features'.\n"
                "Send either the inner features dict (e.g. {\"c1_mean\": 0.01, ...}) or the wrapper {\"features\": { ... }}.\n"
                "Do not wrap twice. Example valid body: {\"features\": {\"c1_mean\": 0.01, \"c1_std\": 0.1, ...}}"
            )
            return JSONResponse(status_code=400, content={"detail": msg})
        res = predict_from_features_dict(features)
        return JSONResponse(res)
    except RuntimeError as e:
        logging.error("RuntimeError in predict_motor_json: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Exception in predict_motor_json: %s", tb)
        return JSONResponse(status_code=500, content={"detail": tb})


@app.post('/predict/motor_csv')
async def predict_motor_csv(file: UploadFile = File(...)):
    tmp = Path('tmp')
    tmp.mkdir(exist_ok=True)
    dst = tmp / file.filename
    with dst.open('wb') as f:
        shutil.copyfileobj(file.file, f)
    from backend.services.motor_service import predict_from_csv
    out = predict_from_csv(str(dst))
    try:
        return JSONResponse({'predictions': out.to_dict(orient='records')})
    except RuntimeError as e:
        logging.error("RuntimeError in predict_motor_csv: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception:
        tb = traceback.format_exc()
        logging.error("Exception in predict_motor_csv: %s", tb)
        return JSONResponse(status_code=500, content={"detail": tb})




@app.post('/predict/motor_timeseries')
async def predict_motor_timeseries(file: UploadFile = File(...)):
    """Upload a raw motor timeseries (.txt) file and run the motor classifier.

    This endpoint extracts the same per-file features used in training and then
    runs the trained RandomForest. It returns both the original multi-class
    prediction and a binary (patient vs healthy) projection for demos.
    """
    tmp = Path('tmp')
    tmp.mkdir(exist_ok=True)
    dst = tmp / file.filename
    with dst.open('wb') as f:
        shutil.copyfileobj(file.file, f)
    try:
        from backend.services.motor_service import predict_from_timeseries_file
        out = predict_from_timeseries_file(str(dst))
        return JSONResponse(out)
    except RuntimeError as e:
        logging.error("RuntimeError in predict_motor_timeseries: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception:
        tb = traceback.format_exc()
        logging.error("Exception in predict_motor_timeseries: %s", tb)
        return JSONResponse(status_code=500, content={"detail": tb})




@app.get('/session/motor/new')
async def motor_session_new():
    """Create a new motor session id for aggregating multiple timeseries files."""
    _prune_motor_sessions()
    session_id = uuid.uuid4().hex
    _MOTOR_SESSIONS[session_id] = {'created_at': time.time(), 'files': []}
    return JSONResponse({'session_id': session_id})


@app.post('/session/motor/{session_id}/upload')
async def motor_session_upload(session_id: str, file: UploadFile = File(...)):
    """Upload one motor timeseries file into an existing session."""
    _prune_motor_sessions()
    if session_id not in _MOTOR_SESSIONS:
        return JSONResponse(status_code=404, content={'detail': 'Unknown session_id. Create one at /session/motor/new.'})

    tmp = Path('tmp')
    tmp.mkdir(exist_ok=True)
    dst = tmp / file.filename
    with dst.open('wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        from backend.services.motor_service import predict_from_timeseries_file
        out = predict_from_timeseries_file(str(dst))
        out['filename'] = file.filename
        _MOTOR_SESSIONS[session_id]['files'].append(out)
        return JSONResponse(out)
    except RuntimeError as e:
        logging.error('RuntimeError in motor_session_upload: %s', str(e))
        return JSONResponse(status_code=500, content={'detail': str(e)})
    except Exception:
        tb = traceback.format_exc()
        logging.error('Exception in motor_session_upload: %s', tb)
        return JSONResponse(status_code=500, content={'detail': tb})


@app.get('/session/motor/{session_id}/summary')
async def motor_session_summary(session_id: str):
    """Get aggregated motor prediction for a session (patient vs healthy)."""
    _prune_motor_sessions()
    summary = _motor_session_summary(session_id)
    if summary is None:
        return JSONResponse(status_code=404, content={'detail': 'Unknown session_id.'})
    return JSONResponse(summary)


@app.post('/predict/gait')
async def predict_gait(file: UploadFile = File(...), explain: bool = False):
    tmp = Path('tmp')
    tmp.mkdir(exist_ok=True)
    dst = tmp / file.filename
    with dst.open('wb') as f:
        shutil.copyfileobj(file.file, f)
    res = predict_from_timeseries_file(str(dst))
    try:
        # attach server trace fields for easier debugging from frontend
        res['server_pid'] = SERVER_PID
        res['server_time'] = datetime.utcnow().isoformat()
        return JSONResponse(res)
    except RuntimeError as e:
        logging.error("RuntimeError in predict_gait: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception:
        tb = traceback.format_exc()
        logging.error("Exception in predict_gait: %s", tb)
        return JSONResponse(status_code=500, content={"detail": tb})


@app.get('/xai/heatmap/{heatmap_id}')
async def xai_heatmap(heatmap_id: str):
    """Return a Grad-CAM heatmap PNG overlay by id (short-lived)."""
    _prune_xai_heatmaps()
    item = _XAI_HEATMAPS.get(heatmap_id)
    if not item:
        return JSONResponse(status_code=404, content={'detail': 'Unknown or expired heatmap_id.'})
    return Response(content=item.get('png') or b'', media_type='image/png')



@app.get('/info')
async def info():
    """Return backend model info so clients can verify which models are loaded."""
    try:
        from backend.services import motor_service, handwriting_service
        
        # Get motor model info
        motor_model, motor_feats, motor_labels = motor_service.ensure_loaded()
        motor_load_error = None
        try:
            motor_load_error = motor_service.get_load_error()
        except Exception:
            motor_load_error = None
        
        # Get handwriting model info
        hw_model_a, hw_model_b, hw_threshold, hw_weight = handwriting_service.ensure_loaded()
        
        return JSONResponse({
            'motor_model': {
                'path': str(motor_service.MODEL_PATH),
                'feature_count': len(motor_feats) if motor_feats is not None else None,
                'labels': motor_labels,
                'load_error': motor_load_error,
            },
            'handwriting_model': {
                'paths': [str(handwriting_service.MODEL_A_PATH), str(handwriting_service.MODEL_B_PATH)],
                'image_size': handwriting_service.DEFAULT_IMG_SIZE,
                'classes': handwriting_service.CLASS_NAMES,
                'loaded': (hw_model_a is not None and hw_model_b is not None),
                'threshold': hw_threshold,
                'weight_resnet': hw_weight,
            },
            'server': {
                'pid': SERVER_PID,
                'startup_time': STARTUP_TIME,
            }
        })
    except Exception as e:
        tb = traceback.format_exc()
        logging.error('Failed to return info: %s', tb)
        return JSONResponse(status_code=500, content={'detail': tb})


if __name__ == '__main__':
    uvicorn.run('backend.main:app', host='0.0.0.0', port=8000, reload=True)
