import cv2
import pydantic
import numpy
import fastapi
import uvicorn
import matplotlib
from Yolov3_Detector import Detector
from collections import Counter

for module in [cv2, pydantic, numpy, fastapi, uvicorn, matplotlib]:
    print(f'{module.__name__}=={module.__version__}')

d = Detector()
d.load_COCO_labels()
d.load_model()
print(d.labels)

img = cv2.imread('images/people walking.PNG')
d.set_detection_params()
res = d.run_detection_on_img(img)
print(res)

