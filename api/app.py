from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import time
from datetime import datetime
import logging
from violence_detector import ViolenceDetector

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize detector
detector = ViolenceDetector()

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_info = detector.get_model_info()
    return jsonify({
        'status': 'healthy',
        'message': 'Violence Detection API - Supports both .h5 and .keras formats',
        'model_loaded': detector.is_model_loaded(),
        'model_info': model_info,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect_violence():
    """Main endpoint for violence detection"""
    start_time = time.time()
    
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Supported: mp4, avi, mov, mkv, wmv, flv'}), 400
        
        if not detector.is_model_loaded():
            return jsonify({
                'error': 'Model not loaded. Please place MoBiLSTM_model.h5 or MoBiLSTM_model.keras in the project directory.',
                'suggestion': 'Save your model after training with: MoBiLSTM_model.save("MoBiLSTM_model.h5") or MoBiLSTM_model.save("MoBiLSTM_model.keras")'
            }), 500
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save uploaded file
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
        
        # Get analysis type from request
        analysis_type = request.form.get('analysis_type', 'summary')  # 'summary' or 'frame_by_frame'
        
        # Perform detection
        if analysis_type == 'frame_by_frame':
            # Frame-by-frame analysis with output video
            output_path = os.path.join(OUTPUT_FOLDER, f"analyzed_{filename}")
            result = detector.analyze_frame_by_frame(filepath, output_path)
            result['download_id'] = file_id
        else:
            # Summary analysis
            result = detector.analyze_summary(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'analysis_type': analysis_type,
            'result': result,
            'processing_time_seconds': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in detect_violence: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/download/<file_id>', methods=['GET'])
def download_result(file_id):
    """Download processed video result"""
    try:
        # Find the output file
        output_files = [f for f in os.listdir(OUTPUT_FOLDER) if file_id in f]
        
        if not output_files:
            return jsonify({'error': 'File not found'}), 404
        
        output_path = os.path.join(OUTPUT_FOLDER, output_files[0])
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"violence_analysis_{file_id}.mp4"
        )
        
    except Exception as e:
        logger.error(f"Error in download_result: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get detailed model status"""
    model_info = detector.get_model_info()
    
    # Check for available model files
    available_models = []
    for model_file in ['MoBiLSTM_model.h5', 'MoBiLSTM_model.keras', '../MoBiLSTM_model.h5', '../MoBiLSTM_model.keras']:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024*1024)  # Size in MB
            available_models.append({
                'path': model_file,
                'size_mb': round(file_size, 2),
                'format': 'keras' if model_file.endswith('.keras') else 'h5 (legacy)'
            })
    
    return jsonify({
        'model_loaded': detector.is_model_loaded(),
        'model_info': model_info,
        'available_models': available_models,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'format_info': {
            'h5': {
                'description': 'HDF5 format - legacy but widely supported',
                'pros': ['Compatible with older code', 'Widely documented', 'Works with your current API'],
                'cons': ['Considered legacy', 'May not support future features'],
                'usage': 'model.save("model.h5")'
            },
            'keras': {
                'description': 'Native Keras format - modern and recommended',
                'pros': ['Future-proof', 'Better compression', 'Faster loading'],
                'cons': ['Newer format', 'Less tutorial coverage'],
                'usage': 'model.save("model.keras")'
            }
        },
        'recommendation': 'Both formats work fine. Use .h5 for compatibility or .keras for future-proofing.',
        'api_support': 'This API automatically detects and loads both formats'
    })

@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load or reload the model"""
    try:
        data = request.get_json() or {}
        model_path = data.get('model_path', 'MoBiLSTM_model.h5')
        
        success = detector.load_model(model_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Model loaded successfully from {model_path}',
                'model_info': detector.get_model_info(),
                'format': 'keras' if model_path.endswith('.keras') else 'h5 (legacy)'
            })
        else:
            return jsonify({
                'error': f'Failed to load model from {model_path}',
                'suggestion': 'Make sure the model file exists and is in the correct format'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/formats', methods=['GET'])
def format_info():
    """Information about .h5 vs .keras formats"""
    return jsonify({
        'current_tensorflow_recommendation': '.keras format',
        'formats': {
            'h5': {
                'description': 'HDF5 format - legacy but widely supported',
                'pros': ['Compatible with older code', 'Widely documented', 'Works with your current API'],
                'cons': ['Considered legacy', 'May not support future features'],
                'usage': 'model.save("model.h5")'
            },
            'keras': {
                'description': 'Native Keras format - modern and recommended',
                'pros': ['Future-proof', 'Better compression', 'Faster loading'],
                'cons': ['Newer format', 'Less tutorial coverage'],
                'usage': 'model.save("model.keras")'
            }
        },
        'recommendation': 'Both formats work fine. Use .h5 for compatibility or .keras for future-proofing.',
        'api_support': 'This API automatically detects and loads both formats'
    })

if __name__ == '__main__':
    print("🚀 Starting Violence Detection API...")
    print("📝 Supports both .h5 (legacy) and .keras (modern) model formats")
    print("💡 Place your model file in the project directory as:")
    print("   - MoBiLSTM_model.h5 (legacy format)")
    print("   - MoBiLSTM_model.keras (modern format)")
    app.run(debug=True, host='0.0.0.0', port=5000)
