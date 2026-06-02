from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .models import ModelManager
from .types import Detection


class Detector:
    def __init__(self, model_manager: Optional[ModelManager] = None) -> None:
        self._model = model_manager or ModelManager.get_instance()

    @classmethod
    def from_config(
        cls,
        model_name: str = "buffalo_l",
        root: str = "~/.insightface",
        providers: Optional[list[str]] = None,
        det_thresh: float = 0.5,
        ctx_id: int = 0,
    ) -> "Detector":
        mm = ModelManager.get_instance(
            model_name=model_name,
            root=root,
            providers=providers,
            det_thresh=det_thresh,
            ctx_id=ctx_id,
        )
        return cls(model_manager=mm)

    def detect(
        self,
        img: np.ndarray,
        max_num: int = 0,
    ) -> list[Detection]:
        if img is None:
            return []
        faces = self._model.detect(img, max_num=max_num)
        results: list[Detection] = []
        for face in faces:
            bbox = face.bbox.astype(np.float32)
            det_score = float(face.det_score) if hasattr(face, "det_score") else 1.0
            kps = face.kps.astype(np.float32) if face.kps is not None else None
            results.append(Detection(bbox=bbox, det_score=det_score, landmarks=kps))
        return results

    def detect_from_path(
        self,
        image_path: str,
        max_num: int = 0,
    ) -> list[Detection]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(img, max_num=max_num)
