from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from Yolov3_Detector import Detector
import numpy as np
from io import BytesIO
import cv2

from collections import Counter


def read_image(file) -> np.ndarray:
    # Create a BytesIO stream from the uploaded file
    image_stream = BytesIO(file)

    # Move the stream's position to the beginning
    image_stream.seek(0)

    # Read the bytes from the stream and decode them using OpenCV
    # This will decode the image data into a NumPy array
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    return image


class DetectionResults(BaseModel):
    filename : str = None
    results_str : str = 'No detections'
    results_list : list = None


class DetectionParams(BaseModel):
    score_threshold: float = 0.5
    NMS_threshold: float = 0.5

app = FastAPI()

detector = Detector()
detector.load_COCO_labels()
detector.load_model()


@app.post("/detection/")
async def detect_on_img(file: UploadFile = File(...) ):

    results = DetectionResults()
    img = read_image(await file.read())
    detector.set_detection_params()
    results.results_list = detector.run_detection_on_img(img)
    results.filename = file.filename
    if len(results.results_list) :
        results.results_str = f'Found {len(results.results_list)} objects'
    return results

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)