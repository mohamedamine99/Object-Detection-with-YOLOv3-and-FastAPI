# Object-Detection-with-YOLOv3-and-FastAPI


Welcome to the **Object Detection with YOLOv3 and FastAPI** repository! This project showcases the integration of YOLOv3, a powerful object detection algorithm, with FastAPI, a modern web framework for building APIs with Python. The combination of these technologies allows you to create a user-friendly API for real-time object detection in images.

## Features

- Integration of YOLOv3 for accurate object detection.
- FastAPI-based API for easy deployment and interaction.
- Real-time object detection in static images.
- Customizable confidence and NMS threshold options.

## Usage

Follow the steps below to get started with using the Object Detection API:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/mohamedamine99/Object-Detection-with-YOLOv3-and-FastAPI.git
   ```
2. **Install Dependencies:**

   ```sh
    pip install -r requirements.txt
   ```
3. **Download YOLOv3 Weights:**
   - [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
   - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
  
4. **Run the FastAPI Server:**
   ```sh
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
