# Object-Detection-with-YOLOv3-and-FastAPI


Welcome to the **Object Detection with YOLOv3 and FastAPI** repository! This project showcases the integration of YOLOv3, a powerful object detection algorithm, with FastAPI, a modern web framework for building APIs with Python. The combination of these technologies allows you to create a user-friendly API for real-time object detection in images.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Contribution](#contribution)
  
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
3. **Download YOLOv3 Weights and COCO labels:**

   Place these files in your working directory.  
   - [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
   - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
   
  
4. **Run the FastAPI Server:**
   - **Method 1 :** Run with uvicorn CLI:
     Activate your environment then run the following command:
   ```sh
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   - **Method 2:** Execute the `main.py` script using either the command-line interface (CLI) or your preferred code editor:
   
   ```sh
   python main.py
   ```
5. **Make API Requests**
      - **Method 1:** Use tools like curl or API clients to make POST requests to the API endpoint for object detection.

        **Example :** 
      Here's an example of how to make a simple API request using curl:

   ```sh
   curl -X POST -F "file=@image.jpg" http://localhost:8000/detection
   ```
   - **Method 2:** Access Swagger UI:
Open your web browser and navigate to the following URL to interact with your API using Swagger UI:
    ```sh
   http://localhost:8000/docs
   ```
   Here, you'll find an interactive interface that presents a list of all available API endpoints. You can explore each endpoint's input parameters, send requests directly from the browser, and view the API's responses. This powerful tool simplifies the process of testing and interacting with your FastAPI application.

## Contribution:

Contributions to this repository are welcome! If you find any issues or want to add new features, feel free to open a pull request.
Let's make object detection with YOLOv3 and FastAPI even more accessible and powerful together!
