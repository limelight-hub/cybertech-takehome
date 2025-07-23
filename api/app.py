from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from violence_detector import ViolenceDetector
from datetime import datetime
import os
import uuid
import time
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize detector
detector = ViolenceDetector()

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
def health_check():
    model_info = detector.get_model_info()
    return {
        'status': 'healthy',
        'message': 'Violence Detection API - FastAPI Version',
        'model_loaded': detector.is_model_loaded(),
        'model_info': model_info,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'timestamp': datetime.now().isoformat()
    }

@app.post("/api/detect")
async def detect_violence(video: UploadFile = File(...), analysis_type: str = Form('summary')):
    start_time = time.time()
    try:
        if not allowed_file(video.filename):
            raise HTTPException(status_code=400, detail="Invalid file format")

        if not detector.is_model_loaded():
            raise HTTPException(status_code=500, detail="Model not loaded")

        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{video.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as buffer:
            buffer.write(await video.read())

        logger.info(f"File saved: {filepath}")

        if analysis_type == 'frame_by_frame':
            output_path = os.path.join(OUTPUT_FOLDER, f"analyzed_{filename}")
            result = detector.analyze_frame_by_frame(filepath, output_path)
            result['download_id'] = file_id
        else:
            result = detector.analyze_summary(filepath)

        os.remove(filepath)
        processing_time = time.time() - start_time

        return {
            'success': True,
            'file_id': file_id,
            'analysis_type': analysis_type,
            'result': result,
            'processing_time_seconds': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/download/{file_id}")
def download_result(file_id: str):
    try:
        output_files = [f for f in os.listdir(OUTPUT_FOLDER) if file_id in f]
        if not output_files:
            raise HTTPException(status_code=404, detail="File not found")

        output_path = os.path.join(OUTPUT_FOLDER, output_files[0])
        return FileResponse(output_path, filename=f"violence_analysis_{file_id}.mp4")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/api/model/status")
def model_status():
    model_info = detector.get_model_info()
    available_models = []
    for model_file in ['MoBiLSTM_model.h5', 'MoBiLSTM_model.keras', '../MoBiLSTM_model.h5', '../MoBiLSTM_model.keras']:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024*1024)
            available_models.append({
                'path': model_file,
                'size_mb': round(file_size, 2),
                'format': 'keras' if model_file.endswith('.keras') else 'h5 (legacy)'
            })

    return {
        'model_loaded': detector.is_model_loaded(),
        'model_info': model_info,
        'available_models': available_models,
        'supported_formats': list(ALLOWED_EXTENSIONS)
    }

@app.post("/api/model/load")
def load_model(model_path: str = Form('MoBiLSTM_model.h5')):
    try:
        success = detector.load_model(model_path)
        if success:
            return {
                'success': True,
                'message': f"Model loaded from {model_path}",
                'model_info': detector.get_model_info(),
                'format': 'keras' if model_path.endswith('.keras') else 'h5 (legacy)'
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
