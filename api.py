from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np

app = FastAPI()

model = YOLO("models/best.pt")


@app.get("/")
def root():
    return {"status": "AI Server Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    image.thumbnail((320,320))

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name)

        results = model.predict(
            source=tmp.name,
            imgsz=320,
            conf=0.25,
            verbose=False
        )

    if results[0].masks is None:

        return {
            "moss_ratio":0,
            "score":100,
            "rank":"A",
            "comment":"コケは検出されませんでした。"
        }

    masks = results[0].masks.data.cpu().numpy()

    merged = np.any(masks, axis=0)

    moss_pixels = np.sum(merged)

    total_pixels = merged.shape[0] * merged.shape[1]

    moss_ratio = moss_pixels / total_pixels * 100

    score = max(
        0,
        100-moss_ratio
    )

    if score>=95:
        rank="A"

    elif score>=85:
        rank="B"

    elif score>=70:
        rank="C"

    elif score>=50:
        rank="D"

    else:
        rank="E"

    return {

        "moss_ratio":round(moss_ratio,1),

        "score":round(score),

        "rank":rank

    }