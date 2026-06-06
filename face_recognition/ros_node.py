#!/usr/bin/env python3
"""ROS2 node: subscribes to camera frames, runs face recognition, publishes results."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from face_recognition import Detector, Enroller, Visualizer


class FaceRecognitionNode(Node):

    def __init__(self):
        super().__init__('face_recognition_node')

        self._gallery_path = Path(
            self.declare_parameter('gallery_path', '/root/.face_recognition/gallery.pkl').value
        )
        threshold = self.declare_parameter('similarity_threshold', 0.36).value

        try:
            self._enroller = Enroller.load(str(self._gallery_path), similarity_threshold=threshold)
            self.get_logger().info(f'Loaded gallery: {self._gallery_path} ({self._enroller.size} identities)')
        except Exception as e:
            self.get_logger().warn(f'No gallery at {self._gallery_path} ({e}), creating empty enroller')
            self._enroller = Enroller.from_config(similarity_threshold=threshold)

        self._detector = Detector()
        self._viz = Visualizer()
        self._pending_enroll_identity: str | None = None

        self._results_pub = self.create_publisher(String, '/cube/face_recognition/results', 10)
        self._frame_pub = self.create_publisher(CompressedImage, '/cube/face_recognition/frame', 10)
        self._enroll_result_pub = self.create_publisher(String, '/cube/face_recognition/enroll_result', 10)

        self.create_subscription(Image, '/camera0/color/image_raw', self._on_camera_frame, 10)
        self.create_subscription(String, '/cube/face_recognition/enroll', self._on_enroll, 10)

        self.get_logger().info('Face recognition node ready')

    def _on_enroll(self, msg: String):
        try:
            data = json.loads(msg.data)
            identity = data.get('name', '').strip()
        except json.JSONDecodeError:
            identity = msg.data.strip()

        if not identity:
            self.get_logger().warn('Enroll request with empty name, ignored')
            return

        self._pending_enroll_identity = identity
        self.get_logger().info(f'Pending enroll for "{identity}" — next detected face will be enrolled')

    def _on_camera_frame(self, msg: Image):
        try:
            frame = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((msg.height, msg.width, 3))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().warn(f'Failed to decode frame: {e}')
            return

        detections = self._detector.detect(frame_bgr)
        if not detections:
            self._publish_results([], [], [], time.time())
            return

        pending = self._pending_enroll_identity
        if pending is not None:
            self._pending_enroll_identity = None
            try:
                fd = self._enroller.enroll(pending, frame_bgr)
                self._enroller.save(str(self._gallery_path))
                self.get_logger().info(
                    f'Enrolled "{pending}" (age:{fd.age}, gender:{"M" if fd.gender == 1 else "F"})'
                )
                self._publish_enroll_result(pending, fd)
            except Exception as e:
                self.get_logger().error(f'Enroll failed for "{pending}": {e}')
                self._publish_enroll_result(pending, None, error=str(e))
            return

        results = self._enroller.search(frame_bgr)
        identities = [r.identity for r in results]
        confidences = [r.confidence for r in results]
        bboxes = [d.bbox.astype(float).tolist() for d in detections]
        det_scores = [float(d.det_score) for d in detections]

        self._publish_results(identities, confidences, bboxes, det_scores, time.time())

        if self._frame_pub.get_subscription_count() > 0:
            labeled = self._viz.draw_search_results(frame_bgr, detections, identities, confidences)
            success, jpeg_buf = cv2.imencode('.jpg', labeled, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                frame_msg = CompressedImage()
                frame_msg.format = 'jpeg'
                frame_msg.data = jpeg_buf.tobytes()
                frame_msg.header.stamp = self.get_clock().now().to_msg()
                frame_msg.header.frame_id = 'camera0'
                self._frame_pub.publish(frame_msg)

    def _publish_results(self, identities, confidences, bboxes, det_scores, timestamp):
        faces = [
            {
                'identity': ident,
                'confidence': round(conf, 4),
                'bbox': bbox,
                'det_score': round(det_score, 4),
            }
            for ident, conf, bbox, det_score in zip(identities, confidences, bboxes, det_scores)
        ]
        payload = json.dumps({'faces': faces, 'count': len(faces), 'time': timestamp})
        self._results_pub.publish(String(data=payload))

    def _publish_enroll_result(self, identity: str, face_data, error: str | None = None):
        payload = json.dumps({
            'identity': identity,
            'success': error is None,
            'age': face_data.age if face_data is not None else None,
            'gender': face_data.gender if face_data is not None else None,
            'error': error,
        })
        self._enroll_result_pub.publish(String(data=payload))


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
