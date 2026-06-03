# face_recognition

A convenience wrapper around [InsightFace](https://github.com/deepinsight/insightface) that bundles detection, recognition, and enrollment into a single importable module with a camera-ready CLI.

InsightFace is the engine (model loading, ONNX inference, face alignment). This module is an opinionated toolkit that adds identity gallery management, unknown-face rejection, persistence, and labeled image export — the ~200 lines of boilerplate that users typically rewrite per project.

Requires the `buffalo_l` model pack (auto-downloaded on first use by InsightFace).

---

## Installation

### Option A — pip from GitHub (recommended)

```bash
pip install git+https://github.com/SteveWufeng/face_recognition_module.git
```

### Option B — pip from local path

```bash
pip install /path/to/face_recognition
```

### Option C — editable (develop) mode

```bash
pip install -e /path/to/face_recognition
```

Lets you edit source and see changes immediately without reinstalling.

### Option D — copy into your project

```bash
cp -r /path/to/face_recognition/face_recognition/ your_project/
```

Then `from face_recognition import Detector` works directly.

### Option E — PYTHONPATH

```bash
export PYTHONPATH="/path/to/face_recognition:$PYTHONPATH"
```

### Option F — pixi

```bash
pixi run python -m pip install /path/to/face_recognition
```

Or add to `pixi.toml`:

```toml
[dependencies]
face-recognition = { git = "https://github.com/SteveWufeng/face_recognition_module.git" }
```

### Option G — Docker

```bash
docker compose build
```

Models and gallery persist locally in `.insightface/` and `.face_recognition/`.
No rebuild needed for code changes — the project root is bind-mounted live.

---

## Package structure

```
face_recognition/
├── __init__.py        # Public API: Detector, Recognizer, Enroller, Visualizer, types
├── __main__.py        # python -m face_recognition entry point
├── types.py           # Detection, FaceData, SearchResult, Detections dataclasses
├── models.py          # ModelManager singleton (lazy-loads ONNX models once)
├── detector.py        # Detector — detect faces in images
├── recognizer.py      # Recognizer — extract embeddings, compare faces
├── enroller.py        # Enroller — identity gallery, search, save/load
├── visualizer.py      # Visualizer — draw bounding boxes, landmarks, labels
└── cli.py             # CLI entry point
```

---

## Python API

### Detector

```python
from face_recognition import Detector
import cv2

detector = Detector()

# From numpy array
img = cv2.imread("photo.jpg")
dets = detector.detect(img)                # -> list[Detection]

# From file path
dets = detector.detect_from_path("photo.jpg")

for d in dets:
    print(d.bbox, d.det_score, d.landmarks)
```

`from_config(...)` accepts `model_name`, `det_thresh`, `ctx_id`, `providers`.

### Recognizer

```python
from face_recognition import Recognizer

recognizer = Recognizer()

# Extract embedding + attributes
fd = recognizer.extract(img, dets[0])      # -> FaceData
fd.embedding       # raw embedding vector
fd.normed_embedding  # L2-normalized embedding
fd.age, fd.gender

# Get embedding only (lightweight)
emb = recognizer.get_embedding(img, dets[0])

# Compare two embeddings
sim = recognizer.compare(emb1, emb2)       # -> float (-1 to 1)
```

### Enroller

```python
from face_recognition import Enroller

enroller = Enroller(similarity_threshold=0.36)

# Enroll from image
enroller.enroll("Alice", img, metadata={"note": "passport photo"})
enroller.enroll_from_path("Bob", "bob.jpg")

# Enroll from multiple images (averages embeddings)
enroller.enroll_multiple("Alice", [img1, img2, img3])

# Search
results = enroller.search(img)                       # -> Detections
results.top                                          # -> SearchResult | None
results.top.identity                                 # "Alice" or "unknown"
results.top.confidence                               # 0.0 to 1.0

for r in results:
    print(r.identity, r.confidence)

# Manage gallery
enroller.identities       # -> ["Alice", "Bob"]
enroller.size             # -> 2
enroller.remove("Bob")
enroller.clear()

# Persist
enroller.save("gallery.pkl")
enroller = Enroller.load("gallery.pkl")
```

### Visualizer

```python
from face_recognition import Visualizer

viz = Visualizer()

# Bounding boxes + landmarks only
labeled = viz.draw_detections(img, dets)

# Bounding boxes + identity labels (green=known, red=unknown)
labeled = viz.draw_search_results(img, dets, identities, confidences)

cv2.imwrite("output.jpg", labeled)
```

---

## CLI

Run via `python -m face_recognition`, or via `face-recognition` when installed with pip.

### With Docker

```bash
# All CLI commands work identically inside Docker:
docker compose run --rm face-recognition list
docker compose run --rm face-recognition enroll-path "Alice" /workspace/alice.jpg
docker compose run --rm face-recognition search /workspace/group.jpg --out /workspace/result.jpg
```

Place images you want the container to access in `./workspace/` (bind-mounted to `/workspace`).

Camera commands (auto-captures in headless mode, no display needed):

```bash
docker compose run --rm face-recognition --camera 2 export-cam /workspace/snapshot.jpg
docker compose run --rm face-recognition --camera 1 enroll-cam "Bob"
```

The camera device index defaults to `0`. Use `--camera N` to select another device.
All `/dev/video0`–`/dev/video9` from the host are forwarded to the container.

### Without Docker

Enroll a face from an image:

```bash
python -m face_recognition enroll-path "Alice" alice.jpg
```

Enroll from camera (interactive — SPACE to capture, ESC to cancel):

```bash
python -m face_recognition enroll-cam "Bob"
```

In headless environments the frame is captured automatically without UI.

Search faces in an image — green labels for known, red for unknown:

```bash
python -m face_recognition search group_photo.jpg --out result.jpg
```

Live camera recognition (ESC to quit):

```bash
python -m face_recognition search-cam
```

In headless environments a single frame is captured, searched, and results printed.

Export an image with detection boxes/landmarks only:

```bash
python -m face_recognition export photo.jpg labeled.jpg
```

Capture camera frame with detections:

```bash
python -m face_recognition export-cam snapshot.jpg
```

In headless environments the first frame is captured and saved immediately.

List, remove, clear gallery:

```bash
python -m face_recognition list
python -m face_recognition remove "Alice"
python -m face_recognition clear
```

### Global options

Place these **before** the subcommand:

| Option | Default | Description |
|---|---|---|
| `--threshold F` | `0.36` | Cosine similarity threshold |
| `--model NAME` | `buffalo_l` | InsightFace model pack |
| `--gallery PATH` | `~/.face_recognition/gallery.pkl` | Gallery file |
| `--camera N` | `0` | Camera device index |

---

## Testing

```bash
cd /path/to/face_recognition
pixi run python test_demo.py
```

Tests detection, enrollment, search (known vs unknown), and pairwise comparison using the InsightFace test images.
