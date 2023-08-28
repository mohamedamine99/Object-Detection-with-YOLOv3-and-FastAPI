import cv2
import pydantic
import numpy
import fastapi
import uvicorn
import matplotlib


for module in [cv2, pydantic, numpy, fastapi, uvicorn, matplotlib]:
    print(f'{module.__name__}=={module.__version__}')



