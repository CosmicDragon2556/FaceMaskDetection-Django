import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class FaceMaskDetector:
    def __init__(self, model):
        self.model = model
        self.labels = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        
    def preprocess_frame(self, frame):
        """Prepare frame for model prediction"""
        face = cv2.resize(frame, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        return np.expand_dims(face, axis=0)
    
    def process_frame(self, frame):
        """Detect mask in a single frame"""
        processed = self.preprocess_frame(frame)
        pred = self.model.predict(processed)[0]
        label = self.labels[np.argmax(pred)]
        confidence = np.max(pred)
        
        # Visualization
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def real_time_detection(self):
        """Run webcam detection (for local testing)"""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow('Mask Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()