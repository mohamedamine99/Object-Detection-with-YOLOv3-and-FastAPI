import os
import shutil

import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_img_for_detection(net, img, size=(320, 320)):
    """
    This function preprocesses an input image for object detection using a specified YOLOv3 or YOLOv3-tiny
    DNN model. The image is resized to the specified size and converted into a blob. The blob is then set
    as the input for the DNN model. The function returns the output of the DNN model after forward pass.

    Parameters:
        net: cv2.dnn_Net object
        YOLOv3 or YOLOv3-tiny DNN model.

        img: numpy.ndarray
        Input image for object detection.

        size: tuple, optional
        Size to which the input image is resized. Default value is (320, 320).

    Returns:
        outputs: numpy.ndarray
        Output of the DNN model after forward pass.


    """
    # Convert the input image into a blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, size, [0, 0, 0], 1, crop=False)

    # Set the blob as the input for the DNN model
    net.setInput(blob)
    layersNames = net.getLayerNames()

    # Perform forward pass through the DNN model
    output_layers_idx = net.getUnconnectedOutLayers()[0] - 1
    outputNames = [(layersNames[idx - 1]) for idx in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)

    # Return the output of the DNN model after forward pass
    return outputs


def detectObjects(img, outputs, score_threshold=0.8, NMS_threshold=0.5):
    """
    This function takes an input image and the output of a YOLOv3 or YOLOv3-tiny DNN model after forward pass,
    detects objects in the image and draws bounding boxes around the objects. It also writes the class label and
    confidence score for each object inside the bounding box.

    Parameters:
        img: numpy.ndarray
        Input image for object detection.

        outputs: numpy.ndarray
        Output of the YOLOv3 or YOLOv3-tiny DNN model after forward pass.

        score_threshold: float, optional
            Minimum confidence score required for an object to be considered for detection. Default value is 0.8.

        NMS_threshold: float, optional
            Non-maximum suppression threshold for eliminating overlapping bounding boxes. Default value is 0.5.

        Returns:
            img: numpy.ndarray
            Input image with bounding boxes and class labels drawn around the detected objects.

    """
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
            if confidence > score_threshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Perform non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, score_threshold, NMS_threshold)

    # Loop over each index in the indices list
    for i in indices:
        # Get the bounding box coordinates, class label and confidence score for the current index
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{labels[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Return the input image with bounding boxes and class labels drawn around the detected objects
    return img


coco_names_file = './coco.names'
yolov3_cfg = './yolov3.cfg'
yolov3_weights = './yolov3.weights'

labels = []
with open(coco_names_file, 'rt') as coco_file:
    labels = coco_file.read().rstrip('\n').rsplit('\n')

print(labels)

# Creating YOLOv3 DNN model from configuration and pre-trained weights
net = cv2.dnn.readNetFromDarknet(yolov3_cfg, yolov3_weights)

img = cv2.imread('./test imgs/highway.PNG')
