
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel

from collections import Counter
import numpy as np
from io import BytesIO
import cv2

from Yolov3_Detector import Detector


def read_image(file) -> np.ndarray:
    """Reads and decodes an image from an uploaded file."""
    # Create a BytesIO stream from the uploaded file
    image_stream = BytesIO(file)

    # Move the stream's position to the beginning
    image_stream.seek(0)

    # Read the bytes from the stream and decode them using OpenCV
    # This will decode the image data into a NumPy array
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    # Return the decoded image as a NumPy array
    return image


class DetectionResults(BaseModel):
    """Model class to represent the results of object detection."""
    filename: str = None
    results_str: str = 'No detections'
    results_list: list = None


class DetectionParams(BaseModel):
    """Model class to represent detection parameters."""
    score_threshold: float = 0.5
    NMS_threshold: float = 0.5


# Create a FastAPI and Detector instances
app = FastAPI()
detector = Detector()


@app.post("/detection/")
async def detect_on_img(file: UploadFile = File(...)):
    """Endpoint to perform object detection on an uploaded image."""

    # Create an instance of DetectionResults to store the detection results
    results = DetectionResults()

    # Read the uploaded image and decode it using OpenCV
    img = read_image(await file.read())

    # Perform object detection using the detector instance
    results.results_list = detector.run_detection_on_img(img)

    # Store the filename of the uploaded image in the results
    results.filename = file.filename

    # If objects are detected in the image, update the results string
    if len(results.results_list):
        detected_labels_counter = Counter([detected_object['label'] for detected_object in results.results_list])
        str2 = ", ".join(f"{value} x {key}" for key, value in detected_labels_counter.items())
        results.results_str = f'Found {len(results.results_list)} objects : ' + str2

    # Return the DetectionResults instance containing the detection results
    return results


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

