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
├── ros_node.py        # ROS2 node — camera subscriber, face recognition publisher
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

All API classes operate on **numpy arrays** (OpenCV images). You manage the
camera yourself — there is no hidden camera state inside the library.

```python
import cv2
from face_recognition import Detector, Enroller

# Open your camera externally (any index, any backend)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

detector = Detector()
enroller = Enroller()

# Detect faces, then search against the gallery
faces = detector.detect(frame)
results = enroller.search(frame)

for r in results:
    print(r.identity, r.confidence)   # "Alice" / "unknown", 0.0–1.0
```

The same pattern works with images, video files, network streams — any
`numpy.ndarray` with shape `(H, W, 3)` in BGR order.

### Detector

```python
from face_recognition import Detector
import cv2

detector = Detector()

# From numpy array (file, camera frame, video stream, …)
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

Camera commands discard the first 10 frames on open to let auto-exposure and
white balance settle, preventing green/frozen first-capture issues.

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

## ROS2 Integration

The `ros_node.py` module provides a ROS2 node that subscribes to camera frames and publishes face recognition results. Designed to run as part of the [NeuroCube](https://github.com/SteveWufeng/NeuroCube) robot system using `rmw_zenoh_cpp`.

### Topics

| Direction | Topic | Type | Description |
|---|---|---|---|
| Input | `/camera0/color/image_raw` | `sensor_msgs/Image` | RGB camera frame |
| Output | `/cube/face_recognition/results` | `std_msgs/String` | JSON with detected faces, identities, confidence scores |
| Output | `/cube/face_recognition/frame` | `sensor_msgs/CompressedImage` | Annotated frame with bounding boxes and labels |
| Input | `/cube/face_recognition/enroll` | `std_msgs/String` | Enroll request — `{"name": "Alice"}` or plain `"Alice"` |
| Output | `/cube/face_recognition/enroll_result` | `std_msgs/String` | Enroll result — success/failure, age, gender |

### Enroll flow

Publish a name to `/cube/face_recognition/enroll`, then the **next detected face** in a camera frame is enrolled and saved to the gallery:

```bash
# Request enroll for "Alice"
ros2 topic pub /cube/face_recognition/enroll std_msgs/String 'data: "Alice"' --once
# Next frame with a face → enrolled, result on /cube/face_recognition/enroll_result
```

### Run with Docker

```bash
# Requires a running Zenoh router (rmw_zenoh_cpp)
docker compose up -d
```

The node auto-loads the gallery from `/root/.face_recognition/gallery.pkl`. If no gallery exists, an empty enroller is created and can be populated via the enroll topic.

### Run standalone (for development)

```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
python -m face_recognition.ros_node
```

---

## Testing

### Standalone (no ROS2)

```bash
cd /path/to/face_recognition
pixi run python test_demo.py
```

Expected output (Tom Hanks enrolled, faces in `t1.jpg` searched):

```
[Enroll] Tom Hanks — age:65 gender:M
[Detect] t1.jpg — 4 face(s) found
[Enroll] Alice — age:31
───────────────────────────────────────────────────────
Identity         Confidence   Age    Gender
───────────────────────────────────────────────────────
Tom Hanks        0.4246       ?      ?   ✓
Alice            0.4619       ?      ?   ✓
unknown          0.0559       ?      ?
unknown          -0.0016      ?      ?
─ Known faces in gallery: ['Tom Hanks', 'Alice']
─ Search returned 4 result(s)
─ Recognized: 2  Unknown: 2
[Compare] face-0 vs face-1 similarity: 0.2542
```

### ROS2 integration test

`test_ros_node.py` publishes a static image to the camera topic and prints recognition results. Requires a running face recognition node and Zenoh router.

```bash
# Search mode — publish test_image.jpg, print recognition results
python test_ros_node.py search [--wait 5]

# Enroll mode — enroll a person from test_image.jpg, then observe result
python test_ros_node.py enroll "Alice" [--wait 5]
```

Expected output — search (empty gallery):

```
[INFO] Publishing /app/test_image.jpg (1350x827)...
[INFO] Done (result=received)

[result] 4 face(s) detected:
  Identity             Confidence   Det Score
  ──────────────────────────────────────────
  unknown              -1.0000      0.8311
  unknown              -1.0000      0.8280
  unknown              -1.0000      0.7820
  unknown              -1.0000      0.6270
```

Expected output — enroll:

```
[INFO] Requesting enroll for "Alice"...
[INFO] Publishing /app/test_image.jpg (1350x827)...
[INFO] Done (result=received)

[enroll] "Alice" — age:None, gender:? ✓
```

Expected output — search after enroll:

```
[INFO] Publishing /app/test_image.jpg (1350x827)...
[INFO] Done (result=received)

[result] 4 face(s) detected:
  Identity             Confidence   Det Score
  ──────────────────────────────────────────
  unknown              -0.0257      0.8311
  Alice                1.0000       0.8280     ✓
  unknown              -0.0294      0.7820
  unknown              0.0615       0.6270
```

Confidence of `-1.0` means no match found (empty gallery or below threshold). Known faces show `✓` and confidence `≥ similarity_threshold` (default `0.36`).

In Docker:

```bash
# Start the ROS2 node
docker compose up -d

# Run the test (separate container, same image)
docker run --rm --network host \
  --entrypoint bash \
  -v .:/app \
  -e PYTHONPATH=/app \
  -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  face-recognition:latest \
  -c 'source /opt/ros/jazzy/setup.bash && python3 /app/test_ros_node.py search --image /app/test_image.jpg --wait 5'
```
