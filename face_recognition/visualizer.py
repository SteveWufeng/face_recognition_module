from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


_LANDMARK_COLORS = [(0, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 0)]


class Visualizer:
    def draw_detections(
        self,
        img: np.ndarray,
        detections: list[Detection],
        labels: list[str] | None = None,
    ) -> np.ndarray:
        canvas = img.copy()
        labels = labels or [""] * len(detections)
        for det, label in zip(detections, labels):
            self._draw_bbox(canvas, det.bbox, label)
            if det.landmarks is not None:
                self._draw_landmarks(canvas, det.landmarks)
        return canvas

    def draw_search_results(
        self,
        img: np.ndarray,
        detections: list[Detection],
        identities: list[str],
        confidences: list[float],
    ) -> np.ndarray:
        canvas = img.copy()
        for det, identity, conf in zip(detections, identities, confidences):
            color = (0, 200, 0) if identity != "unknown" else (0, 0, 200)
            label = f"{identity} ({conf:.2f})"
            self._draw_bbox(canvas, det.bbox, label, color=color)
            if det.landmarks is not None:
                self._draw_landmarks(canvas, det.landmarks)
        return canvas

    def _draw_bbox(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        label: str = "",
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    def _draw_landmarks(self, img: np.ndarray, landmarks: np.ndarray) -> None:
        for i, (x, y) in enumerate(landmarks.astype(int)):
            cv2.circle(img, (x, y), 2, _LANDMARK_COLORS[i], -1)
