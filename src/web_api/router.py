import base64
import os.path
from io import BytesIO

from PIL import Image
import numpy as np
import cv2

from src.prediction import predict_one_stage
from src.processing import get_cropped_image, get_tumor_contour
from fastapi import APIRouter, Request, UploadFile
from fastapi.templating import Jinja2Templates

router = APIRouter(
    prefix="/home",
    tags=["Pages"]
)
templates = Jinja2Templates(directory="src/web_api/templates")

@router.get("/")
async def analysis(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@router.get("/diagnosis/")
async def analysis(request: Request):
    return templates.TemplateResponse("add-diagnosis.html", {"request": request})

@router.get("/analysis/")
async def analysis(request: Request):
    pred_class = 'диагноз'
    return templates.TemplateResponse("analysis.html",
                                      {"request": request, "predClass": pred_class})

@router.post("/analysis/")
def analysis_post(file: UploadFile, request: Request):
    img = Image.open(BytesIO(file.file.read()))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    np_rgb = np.asarray(img)
    np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

    cropped_img = get_cropped_image(np_bgr)
    st_dict_path = os.path.join('src', 'web_api', 'weights.pt')
    pred_class = predict_one_stage(cropped_img, preprocess=False, st_dict_path=st_dict_path)

    selected = np_bgr.copy()

    raw_encoded = cv2.imencode('.jpeg', selected)[1].tostring()
    raw_encoded = base64.b64encode(raw_encoded)
    raw_encoded = str(raw_encoded)
    raw_encoded = raw_encoded[2:-1]

    res_encoded = cv2.imencode('.jpeg', cropped_img)[1].tostring()
    res_encoded = base64.b64encode(res_encoded)
    res_encoded = str(res_encoded)
    res_encoded = res_encoded[2:-1]

    return templates.TemplateResponse("analysis.html",
                                      {"request": request, "rawImage": raw_encoded, "resImage": res_encoded,
                                       "predClass": pred_class})

@router.get("/diagnosis/")
async def diagnosis(request: Request):
    return templates.TemplateResponse("add-diagnosis.html", {"request": request})