import numpy as np
import cv2
from blur.backend.config import CASCADE_XML


class Cascade:
    
    def __init__(self, predict_params: dict | None = None):
        self.model = cv2.CascadeClassifier(
            cv2.data.haarcascades + CASCADE_XML
        )
        
        self.predict_params = {
            "scaleFactor": 1.1,
            "minNeighbors": 5,
            "minSize": (30, 30),
        }

        if predict_params is not None:
            self.predict_params.update(predict_params)
    
    def __repr__(self):
        return f"Cascade model with predict params = {self.predict_params}"

    def predict(self, images: list[np.ndarray], idx: np.ndarray | None = None) -> list[dict]:
        assert (images[0].ndim == 3) and (images[0].shape[2] == 3)
        batch_size = len(images)
        predictions = []
        for img_id in range(batch_size):
            image = images[img_id]
            
            if idx is not None:
                assert (idx.ndim == 1)
                img_id = idx[img_id]
                
            faces = self.model.detectMultiScale(
                image, **self.predict_params,
            )

            for (x, y, w, h) in faces:
                predictions.append({
                    "img_id": img_id,
                    "label": "face",
                    "confidence": 1.0,
                    "xmin": x,
                    "xmax": x + w,
                    "ymin": y,
                    "ymax": y + h,
                    "width": w,
                    "height": h,
                })
                
        return predictions

