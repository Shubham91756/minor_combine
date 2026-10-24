from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import tempfile

from backend.services.handwriting_service import predict_image, predict_image_xai, ensure_loaded as ensure_handwriting_loaded
from backend.services.motor_service import predict_from_features_dict, ensure_loaded
from backend.services.gait_service import predict_from_timeseries_file
import traceback
import logging
import os
from datetime import datetime
import time
import uuid
import hmac

# configure simple file logging for debug
LOG_PATH = Path('backend/error_debug.log')
logging.basicConfig(filename=str(LOG_PATH), level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(title='Parkinsons Models API')

# record startup info for traceability
SERVER_PID = os.getpid()
STARTUP_TIME = datetime.utcnow().isoformat()
try:
    MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_BYTES', str(10 * 1024 * 1024)))
except ValueError as exc:
    raise RuntimeError('MAX_UPLOAD_BYTES must be an integer number of bytes.') from exc
if MAX_UPLOAD_BYTES <= 0:
    raise RuntimeError('MAX_UPLOAD_BYTES must be greater than zero.')
MAX_SESSION_FILES = 20
API_KEY = os.environ.get('PARKINSONS_API_KEY')
APP_ENV = os.environ.get('APP_ENV', 'development').lower()
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        'FRONTEND_ORIGINS',
        'http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001',
    ).split(',')
    if origin.strip()
]


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


@app.middleware('http')
async def protect_api(request: Request, call_next):
    """Require an API key outside local development."""
    content_length = request.headers.get('content-length')
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={'detail': f'Request exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.'},
                )
        except ValueError:
            return JSONResponse(status_code=400, content={'detail': 'Invalid Content-Length header.'})

    if request.url.path == '/healthz':
        return await call_next(request)

    if not API_KEY:
        if APP_ENV == 'production':
            return JSONResponse(status_code=503, content={'detail': 'API authentication is not configured.'})
        if request.client and request.client.host not in {'127.0.0.1', '::1', 'localhost'}:
            return JSONResponse(status_code=401, content={'detail': 'API authentication is required.'})
        return await call_next(request)

    supplied = request.headers.get('X-API-Key', '')
    if not hmac.compare_digest(supplied, API_KEY):
        return JSONResponse(status_code=401, content={'detail': 'Invalid or missing API key.'})
    return await call_next(request)
# ------------------------------------------------------------------
# XAI heatmap cache (in-memory, short TTL)
# ------------------------------------------------------------------
_XAI_HEATMAPS = {}  # heatmap_id -> {'created_at': float, 'png': bytes}
_XAI_HEATMAP_TTL_SEC = 10 * 60  # 10 minutes
_XAI_HEATMAP_MAX_ITEMS = 64


async def _store_upload(file: UploadFile, allowed_suffixes: set[str]) -> tuple[Path, str]:
    """Store an upload under a generated name and enforce basic request limits."""
    original_name = Path(file.filename or '').name
    suffix = Path(original_name).suffix.lower()
    if suffix not in allowed_suffixes:
        allowed = ', '.join(sorted(allowed_suffixes))
        raise HTTPException(status_code=415, detail=f'Unsupported file type. Expected: {allowed}.')

    upload_root = Path('tmp')
    upload_root.mkdir(exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix='parkinsons-upload-', dir=upload_root))
    destination = tmp_dir / f'{uuid.uuid4().hex}{suffix}'
    total = 0
    try:
        with destination.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f'Uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.',
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        tmp_dir.rmdir()
        raise
    finally:
        await file.close()
    return destination, original_name


def _remove_upload(path: Path) -> None:
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        logging.warning('Unable to remove temporary upload directory %s', path.parent)


def _server_error(message: str, exc: Exception) -> JSONResponse:
    error_id = uuid.uuid4().hex
    logging.error('%s [%s]: %s', message, error_id, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={'detail': 'Prediction failed. Contact the server administrator with error ID ' + error_id},
    )


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
    if APP_ENV == 'production' and (not API_KEY or len(API_KEY) < 32):
        raise RuntimeError(
            'PARKINSONS_API_KEY must be configured with at least 32 characters in production.'
        )
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


@app.get('/healthz')
async def healthz():
    """Return a lightweight liveness response without model internals."""
    return JSONResponse({'status': 'ok'})


# Allow CORS for the frontend during development
# Accept multiple localhost ports since frontend may run on different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class MotorFeatures(BaseModel):
    features: dict


@app.post('/predict/handwriting')
async def predict_handwriting(file: UploadFile = File(...), explain: bool = False):
    try:
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            raise HTTPException(
                status_code=415,
                detail='Unsupported file type. Expected a JPEG, PNG, BMP, or WebP image.',
            )
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f'Uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.',
            )
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
    except HTTPException:
        raise
    except RuntimeError as e:
        # known runtime errors (e.g., missing heavy deps) -> return concise message
        logging.error("RuntimeError in predict_handwriting: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return _server_error('Exception in predict_handwriting', e)


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
        return _server_error('Exception in predict_motor_json', e)


@app.post('/predict/motor_csv')
async def predict_motor_csv(file: UploadFile = File(...)):
    dst = None
    try:
        dst, _ = await _store_upload(file, {'.csv'})
        from backend.services.motor_service import predict_from_csv
        out = predict_from_csv(str(dst))
        return JSONResponse({'predictions': out.to_dict(orient='records')})
    except HTTPException:
        raise
    except RuntimeError as e:
        logging.error("RuntimeError in predict_motor_csv: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return _server_error('Exception in predict_motor_csv', e)
    finally:
        if dst is not None:
            _remove_upload(dst)




@app.post('/predict/motor_timeseries')
async def predict_motor_timeseries(file: UploadFile = File(...)):
    """Upload a raw motor timeseries (.txt) file and run the motor classifier.

    This endpoint extracts the same per-file features used in training and then
    runs the trained RandomForest. It returns both the original multi-class
    prediction and a binary (patient vs healthy) projection for demos.
    """
    dst = None
    try:
        dst, _ = await _store_upload(file, {'.txt'})
        from backend.services.motor_service import predict_from_timeseries_file
        out = predict_from_timeseries_file(str(dst))
        return JSONResponse(out)
    except HTTPException:
        raise
    except RuntimeError as e:
        logging.error("RuntimeError in predict_motor_timeseries: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return _server_error('Exception in predict_motor_timeseries', e)
    finally:
        if dst is not None:
            _remove_upload(dst)




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
    if len(_MOTOR_SESSIONS[session_id]['files']) >= MAX_SESSION_FILES:
        return JSONResponse(
            status_code=413,
            content={'detail': f'A motor session may contain at most {MAX_SESSION_FILES} files.'},
        )

    dst = None
    try:
        dst, _ = await _store_upload(file, {'.txt'})
        from backend.services.motor_service import predict_from_timeseries_file
        out = predict_from_timeseries_file(str(dst))
        out['filename'] = f'file_{len(_MOTOR_SESSIONS[session_id]["files"]) + 1}'
        _MOTOR_SESSIONS[session_id]['files'].append(out)
        return JSONResponse(out)
    except HTTPException:
        raise
    except RuntimeError as e:
        logging.error('RuntimeError in motor_session_upload: %s', str(e))
        return JSONResponse(status_code=500, content={'detail': str(e)})
    except Exception as e:
        return _server_error('Exception in motor_session_upload', e)
    finally:
        if dst is not None:
            _remove_upload(dst)


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
    dst = None
    try:
        dst, _ = await _store_upload(file, {'.txt'})
        res = predict_from_timeseries_file(str(dst))
        return JSONResponse(res)
    except HTTPException:
        raise
    except RuntimeError as e:
        logging.error("RuntimeError in predict_gait: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return _server_error('Exception in predict_gait', e)
    finally:
        if dst is not None:
            _remove_upload(dst)


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
    """Return safe backend model status for authorized clients."""
    try:
        from backend.services import motor_service, handwriting_service
        
        # Get motor model info
        motor_model, motor_feats, motor_labels = motor_service.ensure_loaded()
        
        # Get handwriting model info
        hw_model_a, hw_model_b, hw_threshold, hw_weight = handwriting_service.ensure_loaded()
        
        return JSONResponse({
            'motor_model': {
                'feature_count': len(motor_feats) if motor_feats is not None else None,
                'labels': motor_labels,
                'loaded': motor_model is not None,
            },
            'handwriting_model': {
                'image_size': handwriting_service.DEFAULT_IMG_SIZE,
                'classes': handwriting_service.CLASS_NAMES,
                'loaded': (hw_model_a is not None and hw_model_b is not None),
                'threshold': hw_threshold,
                'weight_resnet': hw_weight,
            },
            'server': {
                'startup_time': STARTUP_TIME,
            }
        })
    except Exception as e:
        return _server_error('Failed to return info', e)


if __name__ == '__main__':
    uvicorn.run('backend.main:app', host='0.0.0.0', port=8000, reload=True)
