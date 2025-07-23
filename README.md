# Violence Detection API 

**AI-powered video analysis API for real-time violence detection**

Uses MobileNet + BiLSTM neural network to analyze videos and detect violent content with confidence scores.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Model

Place your trained model in this directory:

- `MoBiLSTM_model.h5` (legacy format)
- `MoBiLSTM_model.keras` (modern format)

### 3. Run the API

```bash
cd api
uvicorn main:app --reload
```
visit the docs at


### 4. Test the API

```bash
# Check if API is running
curl http://localhost:8000/

# Upload a video for analysis
curl -X POST -F "video=@your_video.mp4" http://localhost:8000/api/detect
```

## Key Features

- **Dual Format Support**: Works with both `.h5` and `.keras` model formats
- **Two Analysis Modes**: Summary analysis or frame-by-frame with video output
- **CORS Enabled**: Ready for web frontend integration
- **Auto Model Detection**: Automatically finds and loads your model
- **Comprehensive Logging**: Detailed error messages and processing info

## 📡 Main Endpoints

| Endpoint             | Method | Description                 |
| -------------------- | ------ | --------------------------- |
| `/`                  | GET    | Health check & model status |
| `/api/detect`        | POST   | Analyze video for violence  |
| `/api/download/{id}` | GET    | Download processed video    |
| `/api/model/status`  | GET    | Model information           |

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV
- Flask

## Model Training

If you don't have a trained model yet:

1. **Download from Kaggle:** Use the original notebook's pre-trained model
2. **Train yourself:** Run the `Violence_Detection_MoBiLSTM.ipynb` notebook
3. **Save the model:** Add `MoBiLSTM_model.save('MoBiLSTM_model.h5')` after training

---
