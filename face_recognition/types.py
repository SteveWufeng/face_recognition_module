from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray
    det_score: float
    landmarks: Optional[np.ndarray] = None


@dataclass
class FaceData:
    detection: Detection
    embedding: Optional[np.ndarray] = None
    normed_embedding: Optional[np.ndarray] = None
    age: Optional[int] = None
    gender: Optional[int] = None


@dataclass
class EnrolledFace:
    identity: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    identity: str
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Detections:
    results: list[SearchResult] = field(default_factory=list)

    def __bool__(self) -> bool:
        return len(self.results) > 0

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> SearchResult:
        return self.results[idx]

    @property
    def top(self) -> Optional[SearchResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.confidence)
