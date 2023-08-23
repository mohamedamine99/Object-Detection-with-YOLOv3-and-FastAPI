import cv2
import pydantic
import numpy
import fastapi
import uvicorn

for module in [cv2, pydantic, numpy, fastapi, uvicorn]:
    print(f'{module.__name__}=={module.__version__}')