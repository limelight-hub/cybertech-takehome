# Violence Detection API Documentation

## 📋 Overview

The Violence Detection API uses a MobileNet + BiLSTM neural network to analyze videos and detect violent content in real-time. The API processes video files and returns predictions with confidence scores.

**Base URL:** `http://localhost:5000`  
**API Version:** v1  
**Model Support:** Both `.h5` (legacy) and `.keras` (modern) formats  
**Video Formats:** MP4, AVI, MOV, MKV, WMV, FLV

---

## 🚀 Quick Start

### Installation

```bash
cd api
pip install -r requirements.txt
python app.py
```

### Basic Usage

```bash
# Upload a video for analysis
curl -X POST -F "video=@your_video.mp4" http://localhost:5000/api/detect
```

---

## 📡 API Endpoints

### 1. Health Check

**GET** `/`

Check API status and model information.

**Response:**

```json
{
  "status": "healthy",
  "message": "Violence Detection API - Supports both .h5 and .keras formats",
  "model_loaded": true,
  "model_info": {
    "input_shape": "(None, 16, 64, 64, 3)",
    "output_shape": "(None, 2)",
    "classes": ["NonViolence", "Violence"],
    "sequence_length": 16,
    "image_size": "64x64",
    "total_params": 2847234
  },
  "supported_formats": ["mp4", "avi", "mov", "mkv", "wmv", "flv"],
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

---

### 2. Detect Violence

**POST** `/api/detect`

Analyze a video file for violent content.

**Request:**

- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `video` (file, required): Video file to analyze
  - `analysis_type` (string, optional): "summary" or "frame_by_frame" (default: "summary")

**cURL Example:**

```bash
# Summary analysis (entire video → single result)
curl -X POST \
  -F "video=@test_video.mp4" \
  -F "analysis_type=summary" \
  http://localhost:5000/api/detect

# Frame-by-frame analysis (creates annotated output video)
curl -X POST \
  -F "video=@test_video.mp4" \
  -F "analysis_type=frame_by_frame" \
  http://localhost:5000/api/detect
```

**Response (Summary Analysis):**

```json
{
  "success": true,
  "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "analysis_type": "summary",
  "result": {
    "prediction": "Violence",
    "confidence": 0.8734,
    "probabilities": {
      "NonViolence": 0.1266,
      "Violence": 0.8734
    },
    "analysis_type": "summary"
  },
  "processing_time_seconds": 3.45,
  "timestamp": "2024-01-15T10:35:00.123Z"
}
```

**Response (Frame-by-Frame Analysis):**

```json
{
  "success": true,
  "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "analysis_type": "frame_by_frame",
  "result": {
    "analysis_type": "frame_by_frame",
    "overall_prediction": "Violence",
    "violence_percentage": 65.5,
    "total_frames": 240,
    "violence_frames": 157,
    "output_video_created": true,
    "predictions_sample": [
      {
        "frame": 231,
        "prediction": "Violence",
        "confidence": 0.8923
      },
      {
        "frame": 232,
        "prediction": "NonViolence",
        "confidence": 0.7234
      }
    ],
    "download_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  },
  "processing_time_seconds": 12.34,
  "timestamp": "2024-01-15T10:35:00.123Z"
}
```

---

### 3. Download Processed Video

**GET** `/api/download/{file_id}`

Download the annotated video from frame-by-frame analysis.

**Parameters:**

- `file_id` (string, required): File ID from the detection response

**cURL Example:**

```bash
curl -o analyzed_video.mp4 \
  http://localhost:5000/api/download/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response:**

- **Content-Type:** `video/mp4`
- **File:** Annotated video with violence predictions overlaid on each frame

---

### 4. Model Status

**GET** `/api/model/status`

Get detailed information about the loaded model and available model files.

**Response:**

```json
{
  "model_loaded": true,
  "model_info": {
    "input_shape": "(None, 16, 64, 64, 3)",
    "output_shape": "(None, 2)",
    "classes": ["NonViolence", "Violence"],
    "sequence_length": 16,
    "image_size": "64x64",
    "total_params": 2847234
  },
  "available_models": [
    {
      "path": "MoBiLSTM_model.h5",
      "size_mb": 234.5,
      "format": "h5 (legacy)"
    }
  ],
  "supported_formats": ["mp4", "avi", "mov", "mkv", "wmv", "flv"],
  "format_info": {
    "h5": "Legacy format (still works fine)",
    "keras": "New recommended format"
  }
}
```

---

### 5. Load Model

**POST** `/api/model/load`

Load or reload a model file.

**Request:**

```json
{
  "model_path": "MoBiLSTM_model.h5"
}
```

**cURL Example:**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"model_path": "MoBiLSTM_model.keras"}' \
  http://localhost:5000/api/model/load
```

**Response:**

```json
{
  "success": true,
  "message": "Model loaded successfully from MoBiLSTM_model.keras",
  "model_info": {
    "input_shape": "(None, 16, 64, 64, 3)",
    "output_shape": "(None, 2)",
    "classes": ["NonViolence", "Violence"],
    "sequence_length": 16,
    "image_size": "64x64",
    "total_params": 2847234
  },
  "format": "keras"
}
```

---

### 6. Format Information

**GET** `/api/formats`

Get information about supported model formats (.h5 vs .keras).

**Response:**

```json
{
  "current_tensorflow_recommendation": ".keras format",
  "formats": {
    "h5": {
      "description": "HDF5 format - legacy but widely supported",
      "pros": [
        "Compatible with older code",
        "Widely documented",
        "Works with your current API"
      ],
      "cons": ["Considered legacy", "May not support future features"],
      "usage": "model.save(\"model.h5\")"
    },
    "keras": {
      "description": "Native Keras format - modern and recommended",
      "pros": ["Future-proof", "Better compression", "Faster loading"],
      "cons": ["Newer format", "Less tutorial coverage"],
      "usage": "model.save(\"model.keras\")"
    }
  },
  "recommendation": "Both formats work fine. Use .h5 for compatibility or .keras for future-proofing.",
  "api_support": "This API automatically detects and loads both formats"
}
```

---

## ⚠️ Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error description",
  "suggestion": "Helpful suggestion (when applicable)"
}
```

### Common Error Codes:

| Status Code | Description           | Common Causes                           |
| ----------- | --------------------- | --------------------------------------- |
| 400         | Bad Request           | Missing video file, invalid file format |
| 404         | Not Found             | File ID not found for download          |
| 500         | Internal Server Error | Model not loaded, processing failure    |

### Example Error Responses:

**400 - No video file:**

```json
{
  "error": "No video file provided"
}
```

**500 - Model not loaded:**

```json
{
  "error": "Model not loaded. Please place MoBiLSTM_model.h5 or MoBiLSTM_model.keras in the project directory.",
  "suggestion": "Save your model after training with: MoBiLSTM_model.save(\"MoBiLSTM_model.h5\") or MoBiLSTM_model.save(\"MoBiLSTM_model.keras\")"
}
```

---

## 🔧 Technical Specifications

### Model Requirements:

- **Architecture:** MobileNetV2 + Bidirectional LSTM
- **Input:** 16 frames of 64x64 RGB images
- **Output:** 2 classes (NonViolence, Violence)
- **Supported Formats:** `.h5` (HDF5) or `.keras` (Native Keras)

### Video Processing:

- **Frame Extraction:** 16 evenly distributed frames per video
- **Frame Resize:** All frames resized to 64x64 pixels
- **Normalization:** Pixel values normalized to 0-1 range
- **Sequence Analysis:** Sliding window approach for frame-by-frame analysis

### Performance:

- **Summary Analysis:** ~2-5 seconds per video
- **Frame-by-Frame:** ~10-30 seconds per video (depending on length)
- **Max File Size:** 500MB (configurable)

---

## 💡 Usage Examples

### Python Example:

```python
import requests

# Summary analysis
with open('test_video.mp4', 'rb') as video_file:
    response = requests.post(
        'http://localhost:5000/api/detect',
        files={'video': video_file},
        data={'analysis_type': 'summary'}
    )
    result = response.json()
    print(f"Prediction: {result['result']['prediction']}")
    print(f"Confidence: {result['result']['confidence']:.2%}")
```

### JavaScript Example:

```javascript
// Using fetch API
const formData = new FormData();
formData.append("video", videoFile);
formData.append("analysis_type", "summary");

fetch("http://localhost:5000/api/detect", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => {
    console.log("Prediction:", data.result.prediction);
    console.log("Confidence:", data.result.confidence);
  });
```

### Postman Example:

1. **Method:** POST
2. **URL:** `http://localhost:5000/api/detect`
3. **Body:** form-data
   - Key: `video`, Type: File, Value: [select video file]
   - Key: `analysis_type`, Type: Text, Value: `summary`

---

## 🛠️ Development Notes

### File Structure:

```
api/
├── app.py                 # Main Flask application
├── violence_detector.py   # Core detection logic
├── requirements.txt       # Python dependencies
├── uploads/              # Temporary upload storage
├── outputs/              # Processed video storage
└── MoBiLSTM_model.h5     # Your trained model (place here)
```

### Environment Variables:

```bash
export FLASK_ENV=development  # For debug mode
export FLASK_DEBUG=1          # Enable auto-reload
```

### CORS Support:

The API includes CORS headers, making it accessible from web browsers and frontend applications.

---

## 📞 Support

For issues or questions:

1. Check the model file exists and is in the correct format
2. Verify video file format is supported
3. Check logs for detailed error messages
4. Ensure all dependencies are installed correctly

**Logs location:** Console output when running `python app.py`

---

## 📄 License

This API is part of the Real-Time Violence Detection project using MobileNet and Bi-directional LSTM.
