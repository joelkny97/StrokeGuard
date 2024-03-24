import cv2
import os, urllib, requests
import numpy as np
from django.conf import settings
from django.core.files.temp import NamedTemporaryFile
import retinaface
from datetime import datetime



class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        if not self.video.isOpened():
            return
        success, image = self.video.read()
        resize = cv2.resize(image, (640, 480), interpolation = cv2.INTER_LINEAR)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    def save_frames_from_video(self,delay=1,cycle=10):
        base_path = f"{settings.MEDIA_ROOT}/" + "images/"

        if not self.video.isOpened():
            return
        n=0
        while True:
            ret, frame = self.video.read()
            if (cv2.waitKey(delay=delay) & 0xFF == ord('q')):
                break
            if n == cycle:
                n = 0
                cv2.imwrite(f"{base_path}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg", frame)
            n+=1
        
        

def generate_frames(camera):
    while True:
        frame = camera.get_frame()
        camera.save_frames_from_video()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

