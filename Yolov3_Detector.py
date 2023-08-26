import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Detector:
    """
    Class for performing object detection using YOLOv3 model.
    """

    def __init__(self, config_path='./yolov3.cfg', weights_path='./yolov3.weights',
                 coco_names_path='./coco.names', score_threshold=0.5, NMS_threshold=0.5):
        """
        Initializes the Detector instance.

        Parameters:
            config_path (str): Path to YOLO model configuration file.
            weights_path (str): Path to YOLO model weights file.
            coco_names_path (str): Path to COCO class names file.
            score_threshold (float): Detection score threshold.
            NMS_threshold (float): Non-Maximum Suppression threshold.
        """
        self.load_model(config_path, weights_path)
        self.load_coco_labels(coco_names_path)
        self.set_detection_params(score_threshold, NMS_threshold)

    def load_model(self, config_path, weights_path):
        """
        Loads the YOLO model using provided configuration and weights paths.

        Parameters:
            config_path (str): Path to YOLO model configuration file.
            weights_path (str): Path to YOLO model weights file.
        """
        net = cv2.dnn.readNet(config_path, weights_path)
        self.net = net
        print(f'* Model config loaded from {config_path}\n* Model weights loaded from {weights_path}')

    def load_coco_labels(self, coco_names_path):
        """
        Loads COCO class labels from the provided file.

        Parameters:
            coco_names_path (str): Path to COCO class names file.
        """
        labels = []
        with open(coco_names_path, 'rt') as coco_file:
            labels = coco_file.read().rstrip('\n').rsplit('\n')
        self.labels = labels
        print(f'* COCO labels loaded from {coco_names_path}')

    def preprocess_img(self, img, size=(320, 320)):
        """
        Preprocesses the input image using OpenCV's blobFromImage.

        Parameters:
            img (np.ndarray): Input image.
            size (tuple): Desired image size after preprocessing.

        Returns:
            np.ndarray: Preprocessed image blob.
        """
        self.resize = size
        blob = cv2.dnn.blobFromImage(img, 1 / 255, size, [0, 0, 0], 1, crop=False)
        return blob

    def set_detection_params(self, score_threshold, NMS_threshold):
        """
        Sets detection parameters.

        Parameters:
            score_threshold (float): Detection score threshold.
            NMS_threshold (float): Non-Maximum Suppression threshold.
        """
        self.score_threshold = score_threshold
        self.NMS_threshold = NMS_threshold

    def run_detection_on_img(self, img):
        """
        Runs object detection on the provided image.

        Parameters:
            img (np.ndarray): Input image.

        Returns:
            list: List of dictionaries containing detection results.
        """
        blob = self.preprocess_img(img)
        self.net.setInput(blob)
        layers_names = self.net.getLayerNames()

        # Perform forward pass through the DNN model
        output_layers_idx = self.net.getUnconnectedOutLayers()[0] - 1
        output_names = [(layers_names[idx - 1]) for idx in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_names)

        # Get the shape of the input image
        hT, wT, cT = img.shape

        # Create empty lists to store the bounding boxes, class IDs, and confidence scores for detected objects
        bbox = []
        class_ids = []
        confs = []

        # Loop over each output of the DNN model after the forward pass
        for output in outputs:
            # Loop over each detection in the output
            for det in output:
                # Extract the class ID, confidence score, and bounding box coordinates from the detection
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.score_threshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        # Perform non-maximum suppression to eliminate overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.score_threshold, self.NMS_threshold)

        results_list = []
        # Loop over each index in the indices list
        for i in indices:
            result_dict = {}
            # Get the bounding box coordinates, class label, and confidence score for the current index
            box = bbox[i]
            class_id = class_ids[i]
            conf = int(confs[i] * 100)
            # Create a dictionary to store each detection of the detection results
            result_dict['label'] = self.labels[class_id]
            result_dict['confidence'] = str(conf) +'%'
            result_dict['bbox_xywh'] = box

            results_list.append(result_dict)

        # return the list of detections
        return results_list

