# import os
# import shutil
# import cv2
# import math
# import random
# import numpy as np
# import datetime as dt
# import warnings

# warnings.filterwarnings('ignore')

# # Updated imports for newer versions
# import tensorflow as tf
# # from tensorflow import keras
# from collections import deque
# import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8")  # Updated seaborn style

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Updated Keras imports
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# # Check TensorFlow version
# print(f"TensorFlow version: {tf.__version__}")
# print(f"Keras version: {keras.__version__}")

# # Video display functions
# from IPython.display import HTML
# from base64 import b64encode


# def Play_Video(filepath):
#     """Display video in notebook"""
#     if not os.path.exists(filepath):
#         print(f"Video file not found: {filepath}")
#         return None

#     try:
#         html = ''
#         with open(filepath, 'rb') as video_file:
#             video = video_file.read()
#         src = 'data:video/mp4;base64,' + b64encode(video).decode()
#         html += f'<video width=640 muted controls autoplay loop><source src="{src}" type="video/mp4"></video>'
#         return HTML(html)
#     except Exception as e:
#         print(f"Error loading video: {e}")
#         return None


# # Configuration
# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 16

# # Update these paths according to your dataset location
# DATASET_DIR = "content/real-life-violence-situations-dataset"
# CLASSES_LIST = ["NonViolence", "Violence"]

# # Add these paths for better organization
# PROCESSED_DATA_DIR = "content"
# MODEL_DIR = ""


# def setup_directories():
#     """Create necessary directories"""
#     os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
#     os.makedirs(MODEL_DIR, exist_ok=True)


# def frames_extraction(video_path):
#     """Extract frames from video with improved error handling"""
#     frames_list = []

#     if not os.path.exists(video_path):
#         print(f"Video not found: {video_path}")
#         return frames_list

#     try:
#         # Read the Video File
#         video_reader = cv2.VideoCapture(video_path)

#         if not video_reader.isOpened():
#             print(f"Cannot open video: {video_path}")
#             return frames_list

#         # Get the total number of frames in the video
#         video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

#         if video_frames_count == 0:
#             print(f"No frames found in video: {video_path}")
#             video_reader.release()
#             return frames_list

#         # Calculate the interval after which frames will be added to the list
#         skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

#         # Iterate through the Video Frames
#         for frame_counter in range(SEQUENCE_LENGTH):
#             # Set the current frame position of the video
#             frame_pos = frame_counter * skip_frames_window
#             video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

#             # Reading the frame from the video
#             success, frame = video_reader.read()

#             if not success:
#                 break

#             # Resize the Frame to fixed height and width
#             resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

#             # Normalize the resized frame
#             normalized_frame = resized_frame / 255.0

#             # Append the normalized frame into the frames list
#             frames_list.append(normalized_frame)

#         video_reader.release()

#     except Exception as e:
#         print(f"Error processing video {video_path}: {e}")
#         return []

#     return frames_list


# def create_dataset():
#     """Create dataset with improved error handling and progress tracking"""
#     features = []
#     labels = []
#     video_files_paths = []

#     # Check if dataset directory exists
#     if not os.path.exists(DATASET_DIR):
#         print(f"Dataset directory not found: {DATASET_DIR}")
#         print("Please update DATASET_DIR with the correct path to your dataset")
#         print("Expected structure:")
#         print(f"  {DATASET_DIR}/")
#         print("    ├── NonViolence/")
#         print("    │   ├── video1.mp4")
#         print("    │   └── video2.mp4")
#         print("    └── Violence/")
#         print("        ├── video1.mp4")
#         print("        └── video2.mp4")
#         return np.array([]), np.array([]), []

#     total_processed = 0
#     total_skipped = 0

#     # Iterating through all the classes
#     for class_index, class_name in enumerate(CLASSES_LIST):
#         print(f'\nExtracting Data of Class: {class_name}')
#         print("-" * 50)

#         class_path = os.path.join(DATASET_DIR, class_name)
#         if not os.path.exists(class_path):
#             print(f"Class directory not found: {class_path}")
#             continue

#         # Get the list of video files present in the specific class name directory
#         files_list = os.listdir(class_path)
#         video_files = [f for f in files_list if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'))]

#         print(f"Found {len(video_files)} video files in {class_name}")

#         class_processed = 0
#         class_skipped = 0

#         # Iterate through all the files present in the files list
#         for i, file_name in enumerate(video_files, 1):
#             print(f"Processing {i}/{len(video_files)}: {file_name[:30]}{'...' if len(file_name) > 30 else ''}", end=" ")

#             # Get the complete video path
#             video_file_path = os.path.join(class_path, file_name)

#             # Extract the frames of the video file
#             frames = frames_extraction(video_file_path)

#             # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified
#             if len(frames) == SEQUENCE_LENGTH:
#                 # Append the data to their respective lists
#                 features.append(frames)
#                 labels.append(class_index)
#                 video_files_paths.append(video_file_path)
#                 class_processed += 1
#                 total_processed += 1
#                 print("✓")
#             else:
#                 class_skipped += 1
#                 total_skipped += 1
#                 print(f"✗ (got {len(frames)} frames, expected {SEQUENCE_LENGTH})")

#         print(f"\n{class_name} Summary: {class_processed} processed, {class_skipped} skipped")

#     print(f"\nDataset Creation Summary:")
#     print(f"Total videos processed: {total_processed}")
#     print(f"Total videos skipped: {total_skipped}")
#     print(f"Success rate: {total_processed / (total_processed + total_skipped) * 100:.1f}%" if (
#                                                                                                            total_processed + total_skipped) > 0 else "No videos found")

#     if len(features) == 0:
#         print("\nNo valid videos found! Please check:")
#         print("1. Dataset path is correct")
#         print("2. Video files are in supported formats")
#         print("3. Video files are not corrupted")
#         return np.array([]), np.array([]), []

#     features = np.asarray(features)
#     labels = np.array(labels)

#     return features, labels, video_files_paths


# def create_model():
#     """Create the MobileNet + BiLSTM model"""
#     print("Building model architecture...")

#     # Load MobileNetV2 with updated parameters
#     mobilenet = MobileNetV2(
#         include_top=False,
#         weights="imagenet",
#         input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
#     )

#     # Fine-tuning: make the last 40 layers trainable
#     mobilenet.trainable = True
#     for layer in mobilenet.layers[:-40]:
#         layer.trainable = False

#     print(f"MobileNet layers: {len(mobilenet.layers)} (last 40 trainable)")

#     # Build the model
#     model = Sequential([
#         # Input layer
#         Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),

#         # Apply MobileNet to each frame
#         TimeDistributed(mobilenet, name='mobilenet_features'),
#         Dropout(0.25, name='dropout_1'),

#         # Flatten the features
#         TimeDistributed(Flatten(), name='flatten_features'),

#         # Bidirectional LSTM
#         Bidirectional(LSTM(32, return_sequences=False), name='bilstm'),
#         Dropout(0.25, name='dropout_2'),

#         # Dense layers
#         Dense(256, activation='relu', name='dense_1'),
#         Dropout(0.25, name='dropout_3'),
#         Dense(128, activation='relu', name='dense_2'),
#         Dropout(0.25, name='dropout_4'),
#         Dense(64, activation='relu', name='dense_3'),
#         Dropout(0.25, name='dropout_5'),
#         Dense(32, activation='relu', name='dense_4'),
#         Dropout(0.25, name='dropout_6'),

#         # Output layer
#         Dense(len(CLASSES_LIST), activation='softmax', name='output')
#     ])

#     return model


# def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
#     """Plot training metrics"""
#     metric_value_1 = model_training_history.history[metric_name_1]
#     metric_value_2 = model_training_history.history[metric_name_2]

#     epochs = range(len(metric_value_1))

#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1, linewidth=2)
#     plt.plot(epochs, metric_value_2, 'orange', label=metric_name_2, linewidth=2)
#     plt.title(plot_name, fontsize=14, fontweight='bold')
#     plt.xlabel('Epochs', fontsize=12)
#     plt.ylabel('Value', fontsize=12)
#     plt.legend(fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# def predict_frames(video_file_path, output_file_path, model, sequence_length):
#     """Predict violence in video frames"""
#     if not os.path.exists(video_file_path):
#         print(f"Input video not found: {video_file_path}")
#         return

#     # Read from the video file
#     video_reader = cv2.VideoCapture(video_file_path)

#     if not video_reader.isOpened():
#         print(f"Cannot open video: {video_file_path}")
#         return

#     # Get video properties
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = video_reader.get(cv2.CAP_PROP_FPS)

#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

#     # VideoWriter to store the output video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_file_path, fourcc, fps,
#                                    (original_video_width, original_video_height))

#     # Queue to store video frames
#     frames_queue = deque(maxlen=sequence_length)
#     predicted_class_name = ''

#     frame_count = 0

#     print(f"Processing video: {video_file_path}")
#     print(f"Output will be saved to: {output_file_path}")

#     # Process video frames
#     while video_reader.isOpened():
#         ok, frame = video_reader.read()

#         if not ok:
#             break

#         frame_count += 1

#         # Resize and normalize frame
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#         normalized_frame = resized_frame / 255.0
#         frames_queue.append(normalized_frame)

#         # Make prediction when we have enough frames
#         if len(frames_queue) == sequence_length:
#             try:
#                 # Predict
#                 frames_array = np.expand_dims(list(frames_queue), axis=0)
#                 predicted_probabilities = model.predict(frames_array, verbose=0)[0]
#                 predicted_label = np.argmax(predicted_probabilities)
#                 predicted_class_name = CLASSES_LIST[predicted_label]
#                 confidence = predicted_probabilities[predicted_label]

#                 # Add confidence to display
#                 display_text = f"{predicted_class_name} ({confidence:.2f})"
#             except Exception as e:
#                 print(f"Prediction error at frame {frame_count}: {e}")
#                 display_text = "Error"
#         else:
#             display_text = "Loading..."

#         # Add text to frame
#         color = (0, 0, 255) if predicted_class_name == "Violence" else (0, 255, 0)
#         cv2.putText(frame, display_text, (10, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

#         # Write frame to output video
#         video_writer.write(frame)

#         # Print progress every 100 frames
#         if frame_count % 100 == 0:
#             print(f"Processed {frame_count} frames...")

#     # Clean up
#     video_reader.release()
#     video_writer.release()
#     print(f"Completed! Processed {frame_count} frames. Output saved to: {output_file_path}")


# def main():
#     """Main execution function with menu"""
#     print("=" * 60)
#     print("    VIOLENCE DETECTION SYSTEM")
#     print("=" * 60)

#     setup_directories()

#     # Check what files exist
#     features_file = os.path.join(PROCESSED_DATA_DIR, "features.npy")
#     labels_file = os.path.join(PROCESSED_DATA_DIR, "labels.npy")
#     paths_file = os.path.join(PROCESSED_DATA_DIR, "video_files_paths.npy")
#     model_file = os.path.join(MODEL_DIR, "violence_detection_model.h5")

#     data_exists = all(os.path.exists(f) for f in [features_file, labels_file, paths_file])
#     model_exists = os.path.exists(model_file)

#     print(f"Dataset processed: {'✓' if data_exists else '✗'}")
#     print(f"Model trained: {'✓' if model_exists else '✗'}")
#     print()

#     while True:
#         print("Choose an option:")
#         print("1. Create dataset from videos")
#         print("2. Train model")
#         print("3. Evaluate model")
#         print("4. Test on new video")
#         print("5. Show dataset info")
#         print("0. Exit")

#         choice = input("\nEnter your choice (0-5): ").strip()

#         if choice == "0":
#             print("Goodbye!")
#             break

#         elif choice == "1":
#             print("\nCreating dataset...")
#             features, labels, video_files_paths = create_dataset()

#             if len(features) > 0:
#                 # Save the extracted data
#                 np.save(features_file, features)
#                 np.save(labels_file, labels)
#                 np.save(paths_file, video_files_paths)
#                 print(f"\nDataset saved successfully!")
#                 print(f"Features shape: {features.shape}")
#                 print(f"Labels shape: {labels.shape}")
#                 data_exists = True
#             else:
#                 print("Dataset creation failed!")

#         elif choice == "2":
#             if not data_exists:
#                 print("Dataset not found! Please create dataset first (option 1).")
#                 continue

#             print("\nLoading dataset...")
#             features = np.load(features_file)
#             labels = np.load(labels_file)
#             video_files_paths = np.load(paths_file)
#             print(f"Loaded dataset: {features.shape}")

#             # Encode labels
#             one_hot_encoded_labels = to_categorical(labels)

#             # Split the data
#             features_train, features_test, labels_train, labels_test = train_test_split(
#                 features, one_hot_encoded_labels, test_size=0.1, shuffle=True, random_state=42
#             )

#             print(f"Training set: {features_train.shape}")
#             print(f"Test set: {features_test.shape}")

#             # Create and compile model
#             print("\nCreating model...")
#             model = create_model()

#             # Compile model
#             model.compile(
#                 optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy']
#             )

#             model.summary()

#             # Set up callbacks
#             early_stopping = EarlyStopping(
#                 monitor='val_accuracy',
#                 patience=10,
#                 restore_best_weights=True,
#                 verbose=1
#             )

#             reduce_lr = ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.6,
#                 patience=5,
#                 min_lr=0.00005,
#                 verbose=1
#             )

#             # Train model
#             print("\nStarting training...")
#             history = model.fit(
#                 features_train, labels_train,
#                 epochs=50,
#                 batch_size=8,
#                 validation_split=0.2,
#                 callbacks=[early_stopping, reduce_lr],
#                 verbose=1
#             )

#             # Evaluate model
#             print("\nEvaluating model...")
#             test_loss, test_accuracy = model.evaluate(features_test, labels_test, verbose=0)
#             print(f"Test Accuracy: {test_accuracy:.4f}")

#             # Plot training history
#             plot_metric(history, 'loss', 'val_loss', 'Training vs Validation Loss')
#             plot_metric(history, 'accuracy', 'val_accuracy', 'Training vs Validation Accuracy')

#             # Save model
#             model.save(model_file)
#             print(f"\nModel saved as '{model_file}'")
#             model_exists = True

#         elif choice == "3":
#             if not model_exists:
#                 print("Model not found! Please train model first (option 2).")
#                 continue

#             if not data_exists:
#                 print("Dataset not found! Please create dataset first (option 1).")
#                 continue

#             print("\nLoading model and dataset...")
#             model = tf.keras.models.load_model(model_file)
#             features = np.load(features_file)
#             labels = np.load(labels_file)

#             # Prepare test data
#             one_hot_encoded_labels = to_categorical(labels)
#             _, features_test, _, labels_test = train_test_split(
#                 features, one_hot_encoded_labels, test_size=0.1, shuffle=True, random_state=42
#             )

#             # Detailed evaluation
#             predictions = model.predict(features_test, verbose=0)
#             predicted_labels = np.argmax(predictions, axis=1)
#             true_labels = np.argmax(labels_test, axis=1)

#             # Accuracy
#             accuracy = accuracy_score(true_labels, predicted_labels)
#             print(f"Accuracy Score: {accuracy:.4f}")

#             # Confusion Matrix
#             import seaborn as sns
#             plt.figure(figsize=(8, 6))
#             cm = confusion_matrix(true_labels, predicted_labels)
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                         xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
#             plt.title('Confusion Matrix')
#             plt.xlabel('Predicted')
#             plt.ylabel('Actual')
#             plt.tight_layout()
#             plt.show()

#             # Classification Report
#             report = classification_report(true_labels, predicted_labels,
#                                            target_names=CLASSES_LIST)
#             print("Classification Report:")
#             print(report)

#         elif choice == "4":
#             if not model_exists:
#                 print("Model not found! Please train model first (option 2).")
#                 continue

#             video_path = input("Enter path to video file: ").strip()
#             if not os.path.exists(video_path):
#                 print("Video file not found!")
#                 continue

#             output_path = input("Enter output path (press Enter for default): ").strip()
#             if not output_path:
#                 output_path = f"./output_video_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

#             print("\nLoading model...")
#             model = tf.keras.models.load_model(model_file)

#             predict_frames(video_path, output_path, model, SEQUENCE_LENGTH)

#         elif choice == "5":
#             if not data_exists:
#                 print("Dataset not found! Please create dataset first (option 1).")
#                 continue

#             features = np.load(features_file)
#             labels = np.load(labels_file)
#             video_files_paths = np.load(paths_file)

#             print(f"\nDataset Information:")
#             print(f"Total samples: {len(features)}")
#             print(f"Features shape: {features.shape}")
#             print(f"Sequence length: {SEQUENCE_LENGTH}")
#             print(f"Image size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")

#             unique, counts = np.unique(labels, return_counts=True)
#             print(f"\nClass distribution:")
#             for i, count in zip(unique, counts):
#                 print(f"  {CLASSES_LIST[i]}: {count} videos ({count / len(labels) * 100:.1f}%)")

#         else:
#             print("Invalid choice! Please try again.")

#         print("\n" + "=" * 60 + "\n")


# # Main execution
# if __name__ == "__main__":
#     main()
