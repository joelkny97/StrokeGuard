
import os
import pathlib
import sys
import numpy as np
import cv2
import torch 
from glob import glob
from PIL import Image
from retinaface import RetinaFace

class StrokeDetectorClassifier():
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def preprocess(self, path_to_images_folder: str):
        
       
        image_files = glob(os.path.join(path_to_images_folder, '*.jpeg') )
        
        images_data = []
        for image_file in image_files:
            image = Image.open(image_file)
            images_data.append(image)
        
        return images_data
        
        

        



if __name__ == "__main__":
    sd = StrokeDetectorClassifier()

