import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ViolenceDetector:
    def __init__(self):
        self.model = None
        self.IMAGE_HEIGHT = 64
        self.IMAGE_WIDTH = 64
        self.SEQUENCE_LENGTH = 16
        self.CLASSES_LIST = ["NonViolence", "Violence"]
        
        # Try to auto-load model if it exists
        self.auto_load_model()
        
    def auto_load_model(self):
        """Try to automatically load model from common locations and formats"""
        possible_paths = [
            'MoBiLSTM_model.h5',           # Your current format
            'MoBiLSTM_model.keras',        # New format
            '../MoBiLSTM_model.h5',        # Parent directory
            '../MoBiLSTM_model.keras',     # Parent directory, new format
            'model.h5',
            'model.keras',
            'violence_detection_model.h5',
            'violence_detection_model.keras'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found model at: {path}")
                if self.load_model(path):
                    break
    
    def load_model(self, model_path):
        """Load model - supports both .h5 and .keras formats"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model (TensorFlow handles both formats automatically)
            self.model = tf.keras.models.load_model(model_path)
            
            # Check if it's the expected format
            expected_input_shape = (None, self.SEQUENCE_LENGTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
            if self.model.input_shape != expected_input_shape:
                logger.warning(f"Model input shape {self.model.input_shape} doesn't match expected {expected_input_shape}")
            
            logger.info(f"✅ Model loaded successfully from {model_path}")
            logger.info(f"📊 Model format: {'.keras' if model_path.endswith('.keras') else '.h5 (legacy)'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self):
        """Get model information"""
        if not self.is_model_loaded():
            return None
        return {
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape),
            'classes': self.CLASSES_LIST,
            'sequence_length': self.SEQUENCE_LENGTH,
            'image_size': f"{self.IMAGE_HEIGHT}x{self.IMAGE_WIDTH}",
            'total_params': self.model.count_params()
        }
    
    def extract_frames(self, video_path):
        """Extract frames from video (from notebook)"""
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)
        
        if not video_reader.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return frames_list
        
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_frames_count == 0:
            video_reader.release()
            return frames_list
        
        skip_frames_window = max(int(video_frames_count / self.SEQUENCE_LENGTH), 1)
        
        for frame_counter in range(self.SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            
            if not success:
                break
                
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_list.append(normalized_frame)
        
        video_reader.release()
        return frames_list
    
    def analyze_summary(self, video_path):
        """Analyze entire video (from notebook predict_video function)"""
        if not self.is_model_loaded():
            raise Exception("Model not loaded")
        
        frames = self.extract_frames(video_path)
        if len(frames) != self.SEQUENCE_LENGTH:
            raise Exception(f"Expected {self.SEQUENCE_LENGTH} frames, got {len(frames)}")
        
        # Predict using the model
        predicted_probabilities = self.model.predict(np.expand_dims(frames, axis=0), verbose=0)[0]
        predicted_label = np.argmax(predicted_probabilities)
        predicted_class = self.CLASSES_LIST[predicted_label]
        confidence = predicted_probabilities[predicted_label]
        
        return {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                'NonViolence': float(predicted_probabilities[0]),
                'Violence': float(predicted_probabilities[1])
            },
            'analysis_type': 'summary'
        }
    
    def analyze_frame_by_frame(self, input_path, output_path):
        """Frame-by-frame analysis (from notebook predict_frames function)"""
        if not self.is_model_loaded():
            raise Exception("Model not loaded")
        
        video_reader = cv2.VideoCapture(input_path)
        if not video_reader.isOpened():
            raise Exception(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = video_reader.get(cv2.CAP_PROP_FPS)
        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        frame_count = 0
        violence_frames = 0
        predictions_log = []
        
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Prepare frame
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_queue.append(normalized_frame)
            
            predicted_class_name = "Loading..."
            
            # Predict when we have enough frames
            if len(frames_queue) == self.SEQUENCE_LENGTH:
                try:
                    predictions = self.model.predict(np.expand_dims(list(frames_queue), axis=0), verbose=0)[0]
                    predicted_label = np.argmax(predictions)
                    predicted_class_name = self.CLASSES_LIST[predicted_label]
                    confidence = predictions[predicted_label]
                    
                    if predicted_class_name == "Violence":
                        violence_frames += 1
                    
                    predictions_log.append({
                        'frame': frame_count,
                        'prediction': predicted_class_name,
                        'confidence': float(confidence)
                    })
                    
                except Exception as e:
                    logger.error(f"Prediction error at frame {frame_count}: {str(e)}")
                    predicted_class_name = "Error"
            
            # Add text to frame
            color = (0, 0, 255) if predicted_class_name == "Violence" else (0, 255, 0)
            if predicted_class_name in ["Loading...", "Error"]:
                color = (255, 255, 0)
                
            cv2.putText(frame, predicted_class_name, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            video_writer.write(frame)
        
        video_reader.release()
        video_writer.release()
        
        # Calculate statistics
        total_prediction_frames = max(frame_count - self.SEQUENCE_LENGTH + 1, 1)
        violence_percentage = (violence_frames / total_prediction_frames) * 100
        overall_prediction = "Violence" if violence_percentage > 50 else "NonViolence"
        
        return {
            'analysis_type': 'frame_by_frame',
            'overall_prediction': overall_prediction,
            'violence_percentage': violence_percentage,
            'total_frames': frame_count,
            'violence_frames': violence_frames,
            'output_video_created': True,
            'predictions_sample': predictions_log[-10:]
        }
