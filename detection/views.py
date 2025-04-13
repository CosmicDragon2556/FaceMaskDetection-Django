# from django.shortcuts import render

# # Create your views here.

from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators import gzip
import cv2
import numpy as np
from .utils.mask_detection import FaceMaskDetector

detector = FaceMaskDetector()  # Your detection class

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detector.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace;boundary=frame")

# views.py
from detection.models.mask_model import build_model
from detection.utils.data_loader import DataLoader
from detection.utils.detector import FaceMaskDetector
from detection.utils.trainer import ModelTrainer

def train_view(request):
    # 1. Load data
    df = DataLoader.load_dataset('images/', 'annotations/')
    DataLoader.prepare_cnn_dataset(df, 'images/')
    
    # 2. Train model
    model = build_model()
    class_weights = ModelTrainer.get_class_weights(
        {'with_mask':3232, 'without_mask':717, 'mask_weared_incorrect':123}
    )
    ModelTrainer.train_model(model, 'dataset_cnn', class_weights)
    
    # 3. Save trained model
    model.save('detection/models/trained_model.h5')