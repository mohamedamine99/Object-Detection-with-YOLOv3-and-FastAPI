from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.responses import JSONResponse
from fastapi import Form
from typing import List

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
