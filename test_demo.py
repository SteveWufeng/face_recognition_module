"""Manual test: detection → enrollment → recognition + unknown rejection."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_recognition import Detector, Enroller

IMG_DIR = Path(__file__).resolve().parent.parent / "python-package" / "insightface" / "data" / "images"

def test():
    detector = Detector()
    enroller = Enroller(similarity_threshold=0.36)

    # 1. ENROLL Tom Hanks
    tom_path = str(IMG_DIR / "Tom_Hanks_54745.png")
    fd = enroller.enroll_from_path("Tom Hanks", tom_path,
                                   metadata={"name": "Tom Hanks", "source": "Tom_Hanks_54745.png"})
    print(f"[Enroll] Tom Hanks — age:{fd.age} gender:{'M' if fd.gender==1 else 'F'}")

    # 2. ENROLL from group photo as "Alice" (first face)
    t1_path = str(IMG_DIR / "t1.jpg")
    img = cv2.imread(t1_path)
    dets = detector.detect(img)
    print(f"\n[Detect] t1.jpg — {len(dets)} face(s) found")

    if len(dets) >= 2:
        fd2 = enroller.enroll("Alice", img, metadata={"note": "first face in t1.jpg"})
        print(f"[Enroll] Alice — age:{fd2.age}")

    # 3. SEARCH all faces in t1.jpg against gallery
    print(f"\n{'─'*55}")
    print(f"{'Identity':<16} {'Confidence':<12} {'Age':<6} {'Gender':<6}")
    print(f"{'─'*55}")

    results = enroller.search(img)
    for r in results:
        age = "?"
        gender = "?"
        if r.metadata:
            for fd_tmp in [r.metadata]:
                pass
        flag = " ✓" if r.identity != "unknown" else ""
        print(f"{r.identity:<16} {r.confidence:<12.4f} {age:<6} {gender:<6}{flag}")

    # 4. VERIFY known-vs-unknown
    print(f"\n— Known faces in gallery: {enroller.identities}")
    print(f"— Search returned {len(results)} result(s)")
    known = [r for r in results if r.identity != "unknown"]
    unknown = [r for r in results if r.identity == "unknown"]
    print(f"— Recognized: {len(known)}  Unknown: {len(unknown)}")

    # 5. DIRECT COMPARE
    if len(dets) >= 2:
        from face_recognition import Recognizer
        rec = Recognizer()
        fd_a = rec.extract(img, dets[0])
        fd_b = rec.extract(img, dets[1])
        sim = rec.compare(fd_a.normed_embedding, fd_b.normed_embedding)
        print(f"\n[Compare] face-0 vs face-1 similarity: {sim:.4f}")

if __name__ == "__main__":
    test()
