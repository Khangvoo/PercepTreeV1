import os
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from inference_pretrained.green_meter_core import GreenMeterAnalyzer

app = FastAPI(title="GreenMeter API", version="1.0")

analyzer = GreenMeterAnalyzer()
RESULT_DIR = os.path.join("output", "api_results")


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    person_height_cm: Optional[float] = Form(None),
):
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh tải lên.")

    try:
        result = analyzer.analyze(
            image_bgr=img,
            image_name=file.filename or "upload.jpg",
            save_visual=True,
            output_dir=RESULT_DIR,
            person_height_cm=person_height_cm,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = result.to_dict()
    return payload


if __name__ == "__main__":
    uvicorn.run("inference_pretrained.api_green_meter:app", host="0.0.0.0", port=8000, reload=False)
