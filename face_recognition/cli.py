#!/usr/bin/env python3
"""face_recognition — face enrollment & recognition CLI.

Usage:
  face_recognition enroll-path <identity> <image>         Enroll face from image
  face_recognition enroll-cam <identity>                  Enroll face from camera
  face_recognition search <image> [--out <file>]          Search faces in image
  face_recognition search-cam [--out <dir>]               Live camera recognition
  face_recognition export <image> <out>                   Label detections & export
  face_recognition export-cam <out>                       Capture camera, label, export
  face_recognition list                                   List enrolled identities
  face_recognition remove <identity>                      Remove an identity
  face_recognition clear                                  Clear all identities

Options:
  --threshold F     Similarity threshold (default: 0.36)
  --model NAME      Model pack name (default: buffalo_l)
  --gallery FILE    Gallery file path (default: ~/.face_recognition/gallery.pkl)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_recognition import Detector, Enroller, Recognizer
from face_recognition.visualizer import Visualizer


_GALLERY_DIR = Path.home() / ".face_recognition"
_GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def _get_gallery_path(path: str | None) -> Path:
    return Path(path) if path else _GALLERY_DIR / "gallery.pkl"


def _load_enroller(gallery: Path, threshold: float, model: str) -> Enroller:
    if gallery.exists():
        print(f"[load] {gallery}")
        return Enroller.load(str(gallery), similarity_threshold=threshold)
    return Enroller.from_config(similarity_threshold=threshold, model_name=model)


def _grab_cam(device: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to grab frame from camera")
    return frame


def cmd_enroll_path(args):
    enroller = _load_enroller(args.gallery, args.threshold, args.model)
    fd = enroller.enroll_from_path(args.identity, args.image,
                                   metadata={"source": args.image})
    enroller.save(str(args.gallery))
    print(f"[enrolled] {args.identity}  (age:{fd.age}, gender:{fd.gender})")


def cmd_enroll_cam(args):
    print(f"[camera] Press SPACE to capture, ESC to cancel")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera", file=sys.stderr)
        sys.exit(1)
    enroller = _load_enroller(args.gallery, args.threshold, args.model)
    detector = Detector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame, max_num=1)
        disp = frame.copy()
        if dets:
            b = dets[0].bbox.astype(int)
            cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(disp, f"Enroll: {args.identity}  [SPACE] confirm  [ESC] cancel",
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Face Recognition - Enroll", disp)
        key = cv2.waitKey(30)
        if key == 32:  # SPACE
            if dets:
                fd = enroller.enroll(args.identity, frame, metadata={"source": "camera"})
                enroller.save(str(args.gallery))
                print(f"[enrolled] {args.identity}  (age:{fd.age}, gender:{fd.gender})")
            else:
                print("[fail] No face detected")
            break
        elif key == 27:  # ESC
            print("[cancel]")
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_search(args):
    enroller = _load_enroller(args.gallery, args.threshold, args.model)
    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read {args.image}", file=sys.stderr)
        sys.exit(1)
    detector = Detector()
    dets = detector.detect(img)
    if not dets:
        print("[result] No faces detected")
        return
    results = enroller.search(img)
    viz = Visualizer()
    identities = [r.identity for r in results]
    confidences = [r.confidence for r in results]
    labeled = viz.draw_search_results(img, dets, identities, confidences)
    if args.out:
        cv2.imwrite(args.out, labeled)
        print(f"[export] {args.out}")
    for r in results:
        flag = "" if r.identity == "unknown" else " ✓"
        print(f"  {r.identity:<20} {r.confidence:.4f}{flag}")

    cv2.imshow("Face Recognition - Search", labeled)
    print("[info] Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cmd_search_cam(args):
    enroller = _load_enroller(args.gallery, args.threshold, args.model)
    if enroller.size == 0:
        print("[warn] Gallery is empty — enroll someone first", file=sys.stderr)
    detector = Detector()
    viz = Visualizer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera", file=sys.stderr)
        sys.exit(1)
    print("[camera] Press ESC to quit")
    frame_idx = 0
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        results = enroller.search(frame) if dets else []
        identities = [r.identity for r in results]
        confidences = [r.confidence for r in results]
        labeled = viz.draw_search_results(frame, dets, identities, confidences)
        cv2.putText(labeled, f"Gallery: {enroller.size} identities  [ESC] quit",
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Face Recognition - Live Recognition", labeled)
        key = cv2.waitKey(30)
        if key == 27:
            break
        if out_dir and (frame_idx % 30 == 0):
            out_path = out_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_path), labeled)
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()


def cmd_export(args):
    detector = Detector()
    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read {args.image}", file=sys.stderr)
        sys.exit(1)
    dets = detector.detect(img)
    viz = Visualizer()
    labeled = viz.draw_detections(
        img, dets,
        labels=[f"face_{i}" for i in range(len(dets))],
    )
    cv2.imwrite(args.out, labeled)
    print(f"[export] {args.out}  ({len(dets)} face(s) labeled)")


def cmd_export_cam(args):
    print(f"[camera] Press SPACE to capture, ESC to cancel")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera", file=sys.stderr)
        sys.exit(1)
    detector = Detector()
    viz = Visualizer()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        labeled = viz.draw_detections(
            frame, dets,
            labels=[f"face_{i}" for i in range(len(dets))],
        )
        cv2.putText(labeled, "[SPACE] capture  [ESC] cancel",
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Face Recognition - Export Camera", labeled)
        key = cv2.waitKey(30)
        if key == 32:
            out = Path(args.out)
            cv2.imwrite(str(out), labeled)
            print(f"[export] {out}  ({len(dets)} face(s))")
            break
        elif key == 27:
            print("[cancel]")
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_list(args):
    enroller = _load_enroller(args.gallery, args.threshold, args.model)
    if enroller.size == 0:
        print("[gallery] empty")
        return
    print(f"[gallery] {enroller.size} identity(s):")
    for ident in enroller.identities:
        print(f"  - {ident}")


def cmd_remove(args):
    enroller = _load_enroller(args.gallery, args.threshold, args.model)
    enroller.remove(args.identity)
    enroller.save(str(args.gallery))
    print(f"[removed] {args.identity}")


def cmd_clear(args):
    enroller = _load_enroller(args.gallery, args.threshold, args.model)
    enroller.clear()
    enroller.save(str(args.gallery))
    print("[cleared]")


def main():
    parser = argparse.ArgumentParser(
        description="face_recognition — face enrollment & recognition",
    )
    parser.add_argument("--threshold", type=float, default=0.36, help="similarity threshold")
    parser.add_argument("--model", default="buffalo_l", help="model pack name")
    parser.add_argument("--gallery", default=None, help="gallery pickle path")

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p = sub.add_parser("enroll-path")
    p.add_argument("identity")
    p.add_argument("image")
    p.set_defaults(func=cmd_enroll_path)

    p = sub.add_parser("enroll-cam")
    p.add_argument("identity")
    p.set_defaults(func=cmd_enroll_cam)

    p = sub.add_parser("search")
    p.add_argument("image")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("search-cam")
    p.add_argument("--out", default=None, help="export frames directory")
    p.set_defaults(func=cmd_search_cam)

    p = sub.add_parser("export")
    p.add_argument("image")
    p.add_argument("out")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("export-cam")
    p.add_argument("out")
    p.set_defaults(func=cmd_export_cam)

    p = sub.add_parser("list")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("remove")
    p.add_argument("identity")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("clear")
    p.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    args.gallery = _get_gallery_path(args.gallery)
    args.func(args)


if __name__ == "__main__":
    main()
