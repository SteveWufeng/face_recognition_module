from __future__ import annotations

import os
import os.path as osp
from typing import Optional

import cv2
import numpy as np
import onnxruntime

from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import model_zoo as i_model_zoo
from insightface.utils import face_align


_DEFAULT_MP_NAME = "buffalo_l"


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(
        self,
        model_name: str = _DEFAULT_MP_NAME,
        root: str = "~/.insightface",
        providers: Optional[list[str]] = None,
        det_thresh: float = 0.5,
        ctx_id: int = 0,
    ) -> None:
        self.model_name = model_name
        self.root = root
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.det_thresh = det_thresh
        self.ctx_id = ctx_id

        self._app: Optional[FaceAnalysis] = None
        self._rec_model = None

    @classmethod
    def get_instance(
        cls,
        model_name: str = _DEFAULT_MP_NAME,
        root: str = "~/.insightface",
        providers: Optional[list[str]] = None,
        det_thresh: float = 0.5,
        ctx_id: int = 0,
    ) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(
                model_name=model_name,
                root=root,
                providers=providers,
                det_thresh=det_thresh,
                ctx_id=ctx_id,
            )
        return cls._instance

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            self._app = FaceAnalysis(
                name=self.model_name,
                root=self.root,
                providers=self.providers,
            )
            self._app.prepare(ctx_id=self.ctx_id, det_thresh=self.det_thresh)
        return self._app

    @property
    def rec_model(self):
        if self._rec_model is None:
            self._rec_model = self.app.models.get("recognition")
        return self._rec_model

    def detect(self, img: np.ndarray, max_num: int = 0) -> list[Face]:
        return self.app.get(img, max_num=max_num)

    def get_embedding(self, img: np.ndarray, face: Face) -> Optional[np.ndarray]:
        if self.rec_model is None:
            return None
        embedding = self.rec_model.get(img, face)
        return embedding

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        from numpy.linalg import norm

        emb1 = emb1.ravel()
        emb2 = emb2.ravel()
        sim = float(np.dot(emb1, emb2) / (norm(emb1) * norm(emb2)))
        return max(-1.0, min(1.0, sim))
