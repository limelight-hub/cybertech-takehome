#!/usr/bin/env python3
"""
Violence Detection API Test Script

This script tests all API endpoints to ensure everything is working correctly.
Run this after starting your API with: python app.py
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000"
TEST_VIDEO_PATH = None  # Will be set by user input or auto-detection

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def print_result(test_name, success, details=""):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"   {details}")

def test_health_check():
    """Test the health check endpoint"""
    print_header("Testing Health Check Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        
        if response.status_code == 200:
            data = response.json()
            print_result("Health Check", True, f"Status: {data.get('status', 'unknown')}")
            print_result("Model Loaded", data.get('model_loaded', False), 
                        f"Model Info: {data.get('model_info', {}).get('input_shape', 'N/A')}")
            return True
        else:
            print_result("Health Check", False, f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Health Check", False, f"Error: {str(e)}")
        return False

def test_model_status():
    """Test the model status endpoint"""
    print_header("Testing Model Status Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/model/status")
        
        if response.status_code == 200:
            data = response.json()
            model_loaded = data.get('model_loaded', False)
            available_models = data.get('available_models', [])
            
            print_result("Model Status Request", True)
            print_result("Model Loaded", model_loaded)
            print_result("Available Models", len(available_models) > 0, 
                        f"Found {len(available_models)} model(s)")
            
            # Print available models
            for model in available_models:
                print(f"   📁 {model['path']} ({model['format']}, {model['size_mb']} MB)")
            
            return model_loaded
        else:
            print_result("Model Status Request", False, f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Model Status Request", False, f"Error: {str(e)}")
        return False

def test_format_info():
    """Test the format information endpoint"""
    print_header("Testing Format Information Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/formats")
        
        if response.status_code == 200:
            data = response.json()
            print_result("Format Info Request", True)
            print(f"   📝 TensorFlow Recommendation: {data.get('current_tensorflow_recommendation', 'N/A')}")
            print(f"   🔄 API Support: {data.get('api_support', 'N/A')}")
            return True
        else:
            print_result("Format Info Request", False, f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Format Info Request", False, f"Error: {str(e)}")
        return False

def find_test_video():
    """Try to find a test video file"""
    possible_paths = [
        "test_video.mp4",
        "../test_video.mp4", 
        "sample.mp4",
        "../sample.mp4"
    ]
    
    # Look for any video files in current or parent directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    for directory in ['.', '..']:
        try:
            for file_path in Path(directory).iterdir():
                if file_path.suffix.lower() in video_extensions:
                    return str(file_path)
        except:
            pass
    
    return None

def test_video_detection(video_path=None):
    """Test video detection endpoints"""
    print_header("Testing Video Detection")
    
    if not video_path:
        video_path = find_test_video()
    
    if not video_path or not os.path.exists(video_path):
        print_result("Video Detection", False, "No test video found")
        print("   💡 To test video detection:")
        print("   1. Place a video file (MP4, AVI, etc.) in this directory")
        print("   2. Name it 'test_video.mp4' or specify path when running script")
        print("   3. Re-run this test")
        return False
    
    print(f"   📹 Using video: {video_path}")
    
    # Test summary analysis
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {'analysis_type': 'summary'}
            
            print("   ⏳ Testing summary analysis...")
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/api/detect", files=files, data=data)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('result', {}).get('prediction', 'Unknown')
                confidence = result.get('result', {}).get('confidence', 0)
                
                print_result("Summary Analysis", True, 
                           f"Prediction: {prediction}, Confidence: {confidence:.2%}, Time: {processing_time:.1f}s")
                
                # Test frame-by-frame analysis (optional, takes longer)
                test_frame_by_frame = input("\n   🎬 Test frame-by-frame analysis? (creates output video, may take time) [y/N]: ")
                
                if test_frame_by_frame.lower() == 'y':
                    with open(video_path, 'rb') as video_file:
                        files = {'video': video_file}
                        data = {'analysis_type': 'frame_by_frame'}
                        
                        print("   ⏳ Testing frame-by-frame analysis (this may take a while)...")
                        start_time = time.time()
                        response = requests.post(f"{API_BASE_URL}/api/detect", files=files, data=data)
                        processing_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result = response.json()
                            overall_pred = result.get('result', {}).get('overall_prediction', 'Unknown')
                            violence_pct = result.get('result', {}).get('violence_percentage', 0)
                            download_id = result.get('result', {}).get('download_id')
                            
                            print_result("Frame-by-Frame Analysis", True,
                                       f"Overall: {overall_pred}, Violence: {violence_pct:.1f}%, Time: {processing_time:.1f}s")
                            
                            # Test download
                            if download_id:
                                test_download(download_id)
                        else:
                            print_result("Frame-by-Frame Analysis", False, 
                                       f"Status Code: {response.status_code}")
                
                return True
            else:
                print_result("Summary Analysis", False, f"Status Code: {response.status_code}")
                if response.status_code == 500:
                    error_msg = response.json().get('error', 'Unknown error')
                    print(f"   Error: {error_msg}")
                return False
                
    except Exception as e:
        print_result("Video Detection", False, f"Error: {str(e)}")
        return False

def test_download(file_id):
    """Test video download endpoint"""
    try:
        print("   ⏳ Testing video download...")
        response = requests.get(f"{API_BASE_URL}/api/download/{file_id}")
        
        if response.status_code == 200:
            # Save the downloaded video
            output_path = f"test_output_{file_id[:8]}.mp4"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            print_result("Video Download", True, f"Saved to {output_path} ({file_size:.1f} MB)")
            
        else:
            print_result("Video Download", False, f"Status Code: {response.status_code}")
            
    except Exception as e:
        print_result("Video Download", False, f"Error: {str(e)}")

def run_all_tests():
    """Run all API tests"""
    print("🚀 Violence Detection API Test Suite")
    print(f"🎯 Testing API at: {API_BASE_URL}")
    
    # Check if API is reachable
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
    except requests.exceptions.ConnectionError:
        print("\n❌ FAILED: Cannot connect to API")
        print("   💡 Make sure the API is running with: python app.py")
        return
    except Exception as e:
        print(f"\n❌ FAILED: Connection error: {str(e)}")
        return
    
    # Run all tests
    tests_passed = 0
    total_tests = 0
    
    # Basic endpoint tests
    if test_health_check():
        tests_passed += 1
    total_tests += 1
    
    if test_model_status():
        tests_passed += 1
    total_tests += 1
    
    if test_format_info():
        tests_passed += 1
    total_tests += 1
    
    # Video detection test (only if model is loaded)
    model_loaded = test_model_status()
    if model_loaded:
        if test_video_detection():
            tests_passed += 1
        total_tests += 1
    else:
        print("\n⚠️  Skipping video detection tests - no model loaded")
        print("   💡 Place your trained model in the api/ directory:")
        print("      - MoBiLSTM_model.h5 (legacy format)")
        print("      - MoBiLSTM_model.keras (modern format)")
    
    # Final summary
    print_header("Test Summary")
    print(f"📊 Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Next steps:")
    print("   1. Check the full API documentation: API_DOCUMENTATION.md")
    print("   2. Try the API with your own videos")
    print("   3. Integrate with your frontend application")

if __name__ == "__main__":
    # Allow custom video path
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if video_path:
        print(f"Using custom video: {video_path}")
    
    run_all_tests() 
