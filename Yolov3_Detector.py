import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Detector:
    resize = (320, 320)
    def __init__(self):
        pass
    def load_model(self, config_path = './yolov3.cfg', weights_path = './yolov3.weights', verbose = True):
        # ok
        # Load YOLO model using provided configuration and weights paths
        net = cv2.dnn.readNet(config_path, weights_path)
        self.net = net
        if verbose:
            print(f'* Model config loaded from {config_path} \n* Model weights loaded from {weights_path}')

        #return net
    def load_COCO_labels(self, coco_names_path = './coco.names'):
        labels = []
        with open(coco_names_path, 'rt') as coco_file:
            labels = coco_file.read().rstrip('\n').rsplit('\n')
        self.labels = labels
        print(f'* COCO labels loaded from {coco_names_path}')


    def preprocess_img(self, img, size=(320, 320)):
        # ok
        self.resize = size
        blob = cv2.dnn.blobFromImage(img, 1 / 255, size, [0, 0, 0], 1, crop=False)
        return blob

    def set_detection_params(self, score_threshold=0.8, NMS_threshold=0.5):
        self.score_threshold = score_threshold
        self.NMS_threshold = NMS_threshold
    def run_detection_on_img(self, img, score_threshold=0.8, NMS_threshold=0.5):

        pass
