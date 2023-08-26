import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Detector:

    def __init__(self, ):

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

    def set_detection_params(self, score_threshold=0.5, NMS_threshold=0.5):
        self.score_threshold = score_threshold
        self.NMS_threshold = NMS_threshold


    def run_detection_on_img(self, img):
        blob = self.preprocess_img(img)
        self.net.setInput(blob)
        layersNames = self.net.getLayerNames()

        # Perform forward pass through the DNN model
        output_layers_idx = self.net.getUnconnectedOutLayers()[0] - 1
        outputNames = [(layersNames[idx - 1]) for idx in self.net.getUnconnectedOutLayers()]
        # print(outputNames)
        outputs = self.net.forward(outputNames)
        # ----------------------------------------------------------------------------------------
        # Get the shape of the input image
        hT, wT, cT = img.shape

        # Create empty lists to store the bounding boxes, class IDs and confidence scores for detected objects
        bbox = []
        classIds = []
        confs = []

        # Loop over each output of the DNN model after forward pass
        for output in outputs:
            # Loop over each detection in the output
            for det in output:
                # Extract the class ID, confidence score and bounding box coordinates from the detection
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.score_threshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        # Perform non-maximum suppression to eliminate overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.score_threshold, self.NMS_threshold)
    # ----------------------------------------------------------------------------------------

        """ results_dict = {
            'bbox_xywh': [],
            'confs': [],
            'labels': []
        }"""

        results_list = []
        # Loop over each index in the indices list
        for i in indices:

            result_dict = {}
            # Get the bounding box coordinates, class label and confidence score for the current index
            box = bbox[i]
            class_id = classIds[i]
            conf = int(confs[i] * 100)

            result_dict['label'] = self.labels[class_id]
            result_dict['confidence'] = conf
            result_dict['bbox_xywh'] = box

            results_list.append(result_dict)

        return results_list
