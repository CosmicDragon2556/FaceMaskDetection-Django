from detection.models.mask_model import build_mask_detector

class FaceMaskDetector:
    def __init__(self):
        self.model = build_mask_detector()  # Initialize your model
        self.labels = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    def process_frame(self, frame):
        # Preprocessing (match your notebook)
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        
        # Prediction
        pred = self.model.predict(frame)[0]
        label = self.labels[np.argmax(pred)]
        
        # Visualization
        cv2.putText(frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame