# src/face_engine.py
import os
import json
import face_recognition
import numpy as np
from config import Config  # Dynamic integration with your exact class setup

class FacultyFaceEngine:
    """
    Multimodal processing engine generating 128-dimensional geometric 
    signatures to identify UMT-SST faculty members via Few-Shot Learning.
    Operates strictly offline using deep residual networks.
    """
    def __init__(self, images_dir=Config.FACULTY_IMAGES_DIR, json_path=Config.FACULTY_JSON_PATH):
        self.images_dir = images_dir
        self.json_path = json_path
                
        # In-memory storage for rapid Euclidean distance matrix operations
        self.known_encodings = []
        self.known_keys = []
        self.faculty_db = {}
                
        self.load_database()
        self.load_known_faculty()

    def load_database(self):
        """Loads the JSON administrative metadata registry into memory."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.faculty_db = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Metadata registry not found at {self.json_path}. Engine operating in blind mode.")
            self.faculty_db = {}
        except json.JSONDecodeError:
            print(f"Critical Error: Malformed JSON syntax in {self.json_path}.")
            self.faculty_db = {}

    def load_known_faculty(self):
        """
        Iterates over the reference directory and extracts 128-dimensional 
        mathematical signatures on system boot. Maps each encoding to its filename key.
        """
        if not os.path.exists(self.images_dir):
            return
        for filename in os.listdir(self.images_dir):
            # Explicitly handles your .png images alongside standard formats
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                key = os.path.splitext(filename)[0]
                                
                # Performance optimization: Only process compute-heavy vectors 
                # for faculty members explicitly registered in the JSON registry
                if key in self.faculty_db:
                    img_path = os.path.join(self.images_dir, filename)
                    try:
                        image = face_recognition.load_image_file(img_path)
                        # Generate bounding boxes and subsequent encodings
                        encodings = face_recognition.face_encodings(image)
                                                
                        if encodings:
                            # Append the primary detected face in the reference image
                            self.known_encodings.append(encodings[0])
                            self.known_keys.append(key)
                    except Exception as e:
                        print(f"System Error processing reference image {filename}: {str(e)}")

    def identify_face(self, unknown_image_stream, threshold=0.6):
        """
        Compares an uploaded vector against the pre-computed database array via Euclidean Distance.
        Handles direct byte streams (BytesIO) passed seamlessly from Streamlit's st.file_uploader.
        """
        try:
            # load_image_file natively decodes BytesIO streams without requiring disk writes
            unknown_image = face_recognition.load_image_file(unknown_image_stream)
            unknown_encodings = face_recognition.face_encodings(unknown_image)
                        
            # Edge Case Handling: Image contains no detectable facial landmarks
            if not unknown_encodings:
                return {
                    "status": "error", 
                    "message": "No clear face detected in the uploaded image. Please ensure lighting is adequate and the subject is facing the camera."
                }
                        
            # Compute distance metric arrays against all known loaded vectors using optimized C backends
            distances = face_recognition.face_distance(self.known_encodings, unknown_encodings[0])
                        
            if len(distances) == 0:
                return {
                    "status": "error", 
                    "message": "System routing error: No reference encodings were successfully loaded into the biometric database during startup."
                }
            
            # Locate the array index of the absolute minimum distance (the closest geometric match)
            best_match_idx = np.argmin(distances)
                        
            # Evaluate the nearest match against the mathematical threshold (default 0.6)
            if distances[best_match_idx] <= threshold:
                matched_key = self.known_keys[best_match_idx]
                profile = self.faculty_db.get(matched_key, {})
                                
                # Confidence metric scaling (inverse representation of Euclidean distance)
                confidence_score = round(float(1 - distances[best_match_idx]), 2)
                                
                return {
                    "status": "success", 
                    "data": profile, 
                    "confidence": confidence_score
                }
                        
            # Execution reaches here if the minimum distance exceeds the 0.6 threshold
            return {
                "status": "unknown", 
                "message": "Face detected, but the resulting biometric signature does not closely match any registered SST faculty member."
            }
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"An anomaly occurred during vector extraction and decoding: {str(e)}"
            }