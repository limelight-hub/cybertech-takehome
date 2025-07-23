# Violence Detection API 

**AI-powered video analysis API for real-time violence detection**

Uses MobileNet + BiLSTM neural network to analyze videos and detect violent content with confidence scores.

## API Documentation
Interactive API docs are available at:
- Swagger UI: https://cybertech-takehome.onrender.com/docs
- ReDoc: https://cybertech-takehome.onrender.com/redoc

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
cd api
uvicorn main:app --reload
```


### 4. Test the API

```bash
# Check if API is running
curl http://localhost:8000/

# Upload a video for analysis (default: summary)
curl -X POST -F "video=@your_video.mp4" http://localhost:8000/api/detect 

# or with analysis_type to specify
curl -X POST -F "video=@your_video.mp4" -F "analysis_type=summary" http://localhost:8000/api/detect
curl -X POST -F "video=@your_video.mp4" -F "analysis_type=" http:frame_by_frame//localhost:8000/api/detect


```

## Key Features

- **Dual Format Support**: Works with both `.h5` and `.keras` model formats
- **Two Analysis Modes**: Summary analysis or frame-by-frame with video output
- **CORS Enabled**: Ready for web frontend integration
- **Auto Model Detection**: Automatically finds and loads your model
- **Comprehensive Logging**: Detailed error messages and processing info

## Main Endpoints

| Endpoint             | Method | Description                 |
| -------------------- | ------ | --------------------------- |
| `/`                  | GET    | Health check & model status |
| `/api/detect`        | POST   | Analyze video for violence  |
| `/api/download/{id}` | GET    | Download processed video    |
| `/api/model/status`  | GET    | Model information           |

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV
- FastAPI

## Model Training: 
- Everything is in the /models folder
- Checkout the original work: [abduulrahmankhalid](https://github.com/abduulrahmankhalid/Real-Time-Violence-Detection/blob/main/Violence_Detection_MoBiLSTM.ipynb)

---
