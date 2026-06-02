from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .models import ModelManager
from .types import Detection, FaceData


class Recognizer:
    def __init__(self, model_manager: Optional[ModelManager] = None) -> None:
        self._model = model_manager or ModelManager.get_instance()

    @classmethod
    def from_config(
        cls,
        model_name: str = "buffalo_l",
        root: str = "~/.insightface",
        providers: Optional[list[str]] = None,
        ctx_id: int = 0,
    ) -> "Recognizer":
        mm = ModelManager.get_instance(
            model_name=model_name,
            root=root,
            providers=providers,
            ctx_id=ctx_id,
        )
        return cls(model_manager=mm)

    def _to_face_obj(self, detection: Detection):
        from insightface.app.common import Face

        face = Face()
        face.bbox = detection.bbox
        face.kps = detection.landmarks
        face.det_score = detection.det_score
        return face

    def get_embedding(self, img: np.ndarray, detection: Detection) -> np.ndarray:
        face = self._to_face_obj(detection)
        emb = self._model.get_embedding(img, face)
        if emb is None:
            raise RuntimeError("Recognition model not loaded")
        return emb.ravel().astype(np.float32)

    def extract(self, img: np.ndarray, detection: Detection) -> FaceData:
        face = self._to_face_obj(detection)
        self._model.get_embedding(img, face)
        embedding = face.embedding.ravel().astype(np.float32) if face.embedding is not None else None
        normed = face.normed_embedding.ravel().astype(np.float32) if face.normed_embedding is not None else None
        age = int(face.age) if hasattr(face, "age") and face.age is not None else None
        gender = int(face.gender) if hasattr(face, "gender") and face.gender is not None else None
        return FaceData(
            detection=detection,
            embedding=embedding,
            normed_embedding=normed,
            age=age,
            gender=gender,
        )

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return self._model.compute_similarity(emb1, emb2)

    def compare_multiple(
        self, query_emb: np.ndarray, gallery_embs: list[np.ndarray]
    ) -> list[float]:
        return [self.compare(query_emb, g) for g in gallery_embs]
