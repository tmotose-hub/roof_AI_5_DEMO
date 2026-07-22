from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

app = FastAPI()

# モデルは起動時に一度だけ読み込む
model = YOLO("models/best.pt")


@app.get("/")
def root():
    return {"status": "AI Server Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 画像読込
    image = Image.open(file.file).convert("RGB")

    # デモ版なので縮小
    image.thumbnail((256, 256))

    # numpyへ変換
    img = np.array(image)

    # 推論
    with torch.inference_mode():

        results = model.predict(
            source=img,
            imgsz=256,
            conf=0.25,
            classes=[0],
            max_det=1,
            half=False,
            verbose=False
        )

    # コケなし
    if results[0].masks is None:

        return {
            "moss_ratio": 0,
            "score": 100,
            "rank": "A",
            "comment": "コケは検出されませんでした。"
        }

    # マスク取得
    masks = results[0].masks.data.cpu().numpy()

    merged = np.any(masks, axis=0)

    moss_pixels = np.sum(merged)

    total_pixels = merged.shape[0] * merged.shape[1]

    moss_ratio = moss_pixels / total_pixels * 100

    # スコア
    score = max(0, 100 - moss_ratio)

    # ランク
    if score >= 95:
        rank = "A"
        comment = "コケはほとんど確認されませんでした。"

    elif score >= 85:
        rank = "B"
        comment = "軽度のコケが確認されました。"

    elif score >= 70:
        rank = "C"
        comment = "中程度のコケが確認されました。"

    elif score >= 50:
        rank = "D"
        comment = "広範囲にコケが確認されました。"

    else:
        rank = "E"
        comment = "重度のコケが確認されました。早めのメンテナンスを推奨します。"

    return {

        "moss_ratio": round(moss_ratio, 1),

        "score": round(score),

        "rank": rank,

        "comment": comment

    }