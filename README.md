# face_recognition

Python package for face detection, recognition, and enrollment built on [InsightFace](https://github.com/deepinsight/insightface).

Requires the `buffalo_l` model pack (auto-downloaded on first use by InsightFace).

---

## Installation

### Option A — pip install (recommended)

```bash
pip install /path/to/face_recognition
```

This installs `face_recognition` into the active Python environment. After this, `import face_recognition` works from anywhere, and `face-recognition` is available as a shell command.

### Option B — editable (develop) mode

```bash
pip install -e /path/to/face_recognition
```

Lets you edit the source files and see changes immediately without reinstalling.

### Option C — copy into your project

```bash
cp -r /path/to/face_recognition/face_recognition/ your_project/
```

Then import as usual: `from face_recognition import Detector`.

### Option D — PYTHONPATH

```bash
export PYTHONPATH="/path/to/face_recognition:$PYTHONPATH"
```

No install, no copy — just point Python at the directory.

### Option E — pixi (if your project uses pixi)

```bash
pixi run python -m pip install /path/to/face_recognition
```

Or add it as a local dependency in `pixi.toml`:

```toml
[tasks]
cmd = "python -m face_recognition search-cam"

[dependencies]
face-recognition = { path = "/path/to/face_recognition" }
```

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

Run via `python -m face_recognition`.

Enroll a face from an image:

```bash
python -m face_recognition enroll-path "Alice" alice.jpg
```

Enroll from camera (SPACE to capture, ESC to cancel):

```bash
python -m face_recognition enroll-cam "Bob"
```

Search faces in an image — green labels for known, red for unknown:

```bash
python -m face_recognition search group_photo.jpg --out result.jpg
```

Live camera recognition (ESC to quit):

```bash
python -m face_recognition search-cam
```

Export an image with detection boxes/landmarks only:

```bash
python -m face_recognition export photo.jpg labeled.jpg
```

Capture camera frame with detections:

```bash
python -m face_recognition export-cam snapshot.jpg
```

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

---

## Testing

```bash
cd /path/to/face_recognition
pixi run python test_demo.py
```

Tests detection, enrollment, search (known vs unknown), and pairwise comparison using the InsightFace test images.
