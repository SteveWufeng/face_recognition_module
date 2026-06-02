from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import Detector
from .models import ModelManager
from .recognizer import Recognizer
from .types import Detections, FaceData, SearchResult


def _to_detections(results: list) -> Detections:
    return Detections(results=results)


class Enroller:
    def __init__(
        self,
        detector: Optional[Detector] = None,
        recognizer: Optional[Recognizer] = None,
        similarity_threshold: float = 0.36,
    ) -> None:
        self._detector = detector or Detector()
        self._recognizer = recognizer or Recognizer()
        self.threshold = similarity_threshold
        self._gallery: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

    @classmethod
    def from_config(
        cls,
        model_name: str = "buffalo_l",
        root: str = "~/.insightface",
        providers: Optional[list[str]] = None,
        det_thresh: float = 0.5,
        ctx_id: int = 0,
        similarity_threshold: float = 0.36,
    ) -> "Enroller":
        mm = ModelManager.get_instance(
            model_name=model_name,
            root=root,
            providers=providers,
            det_thresh=det_thresh,
            ctx_id=ctx_id,
        )
        detector = Detector(model_manager=mm)
        recognizer = Recognizer(model_manager=mm)
        return cls(
            detector=detector,
            recognizer=recognizer,
            similarity_threshold=similarity_threshold,
        )

    def enroll(
        self,
        identity: str,
        img: np.ndarray,
        metadata: Optional[dict] = None,
        max_num: int = 1,
    ) -> FaceData:
        detections = self._detector.detect(img, max_num=max_num)
        if not detections:
            raise ValueError(f"No face detected for identity '{identity}'")
        detection = detections[0]
        face_data = self._recognizer.extract(img, detection)
        if face_data.normed_embedding is None:
            raise RuntimeError("Failed to extract embedding")
        self._gallery[identity] = face_data.normed_embedding
        self._metadata[identity] = metadata or {}
        return face_data

    def enroll_from_path(
        self,
        identity: str,
        image_path: str,
        metadata: Optional[dict] = None,
    ) -> FaceData:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.enroll(identity, img, metadata=metadata)

    def enroll_multiple(
        self,
        identity: str,
        imgs: list[np.ndarray],
        metadata: Optional[dict] = None,
    ) -> np.ndarray:
        if not imgs:
            raise ValueError("No images provided")
        embeddings: list[np.ndarray] = []
        for img in imgs:
            detections = self._detector.detect(img, max_num=1)
            if not detections:
                continue
            face_data = self._recognizer.extract(img, detections[0])
            if face_data.normed_embedding is not None:
                embeddings.append(face_data.normed_embedding)
        if not embeddings:
            raise ValueError(f"No valid face found in any image for identity '{identity}'")
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        self._gallery[identity] = avg_embedding
        self._metadata[identity] = metadata or {}
        return avg_embedding

    def search(
        self,
        img: np.ndarray,
        max_num: int = 0,
        top_k: int = 1,
    ) -> Detections:
        detections = self._detector.detect(img, max_num=max_num)
        if not detections:
            return Detections(results=[])
        results: list[SearchResult] = []
        for det in detections:
            face_data = self._recognizer.extract(img, det)
            if face_data.normed_embedding is None:
                continue
            best_id: Optional[str] = None
            best_conf: float = -1.0
            for identity, gallery_emb in self._gallery.items():
                sim = self._recognizer.compare(face_data.normed_embedding, gallery_emb)
                if sim > best_conf:
                    best_conf = sim
                    best_id = identity
            results.append(
                SearchResult(
                    identity=best_id if best_conf >= self.threshold else "unknown",
                    confidence=best_conf,
                    metadata=self._metadata.get(best_id, {}) if best_id else {},
                )
            )
        return Detections(results=results)

    def search_from_path(
        self,
        image_path: str,
        top_k: int = 1,
    ) -> Detections:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.search(img, top_k=top_k)

    def remove(self, identity: str) -> None:
        self._gallery.pop(identity, None)
        self._metadata.pop(identity, None)

    def clear(self) -> None:
        self._gallery.clear()
        self._metadata.clear()

    @property
    def identities(self) -> list[str]:
        return list(self._gallery.keys())

    @property
    def size(self) -> int:
        return len(self._gallery)

    def save(self, path: str | Path) -> None:
        data = {
            "gallery": self._gallery,
            "metadata": self._metadata,
            "threshold": self.threshold,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "Enroller":
        with open(path, "rb") as f:
            data = pickle.load(f)
        enroller = cls(**kwargs)
        enroller._gallery = data["gallery"]
        enroller._metadata = data["metadata"]
        enroller.threshold = data.get("threshold", enroller.threshold)
        return enroller
