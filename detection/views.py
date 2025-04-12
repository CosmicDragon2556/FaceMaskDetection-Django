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