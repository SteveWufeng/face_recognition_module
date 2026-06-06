#!/usr/bin/env python3
"""Test script: publish test_image.jpg to camera topic, observe recognition results.

Usage:
  # Search mode — publish image, observe recognition
  python test_ros_node.py search

  # Enroll mode — enroll a person first, then search
  python test_ros_node.py enroll "John Doe"

  # Wait time between publish and shutdown (default 3s)
  python test_ros_node.py search --wait 5

Requires a running face_recognition ROS2 node + Zenoh router.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class TestNode(Node):

    def __init__(self, image_path: str, mode: str, identity: str | None, wait: float):
        super().__init__('face_recognition_test')

        self._mode = mode
        self._wait = wait
        self._start = time.monotonic()
        self._done = False
        self._result_received = False

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            self.get_logger().error(f'Cannot read {image_path}')
            sys.exit(1)

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self._results_sub = self.create_subscription(
            String, '/cube/face_recognition/results', self._on_results, 10
        )
        self._enroll_result_sub = self.create_subscription(
            String, '/cube/face_recognition/enroll_result', self._on_enroll_result, 10
        )
        self._image_pub = self.create_publisher(Image, '/camera0/color/image_raw', 10)

        if mode == 'enroll' and identity:
            enroll_pub = self.create_publisher(String, '/cube/face_recognition/enroll', 10)
            self.get_logger().info(f'Requesting enroll for "{identity}"...')
            enroll_pub.publish(String(data=json.dumps({'name': identity})))
            time.sleep(0.5)

        self.get_logger().info(f'Publishing {image_path} ({w}x{h})...')
        msg = Image()
        msg.height = h
        msg.width = w
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = w * 3
        msg.data = img_rgb.tobytes()
        self._image_pub.publish(msg)

        self.create_timer(0.1, self._tick)

    def _on_results(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Bad results JSON: {e}')
            return

        count = data.get('count', 0)
        if count == 0:
            print('\n[result] No faces detected')
        else:
            print(f'\n[result] {count} face(s) detected:')
            print(f'  {"Identity":<20} {"Confidence":<12} {"Det Score":<10}')
            print(f'  {"─"*42}')
            for face in data.get('faces', []):
                ident = face.get('identity', '?')
                conf = face.get('confidence', 0)
                score = face.get('det_score', 0)
                flag = '' if ident == 'unknown' else ' ✓'
                print(f'  {ident:<20} {conf:<12.4f} {score:<10.4f}{flag}')

        self._result_received = True

    def _on_enroll_result(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Bad enroll result JSON: {e}')
            return

        identity = data.get('identity', '?')
        success = data.get('success', False)
        if success:
            age = data.get('age', '?')
            gender = {1: 'M', 0: 'F'}.get(data.get('gender'), '?')
            print(f'\n[enroll] "{identity}" — age:{age}, gender:{gender} ✓')
        else:
            error = data.get('error', 'unknown error')
            print(f'\n[enroll] "{identity}" failed: {error}')

        self._result_received = True

    def _tick(self):
        elapsed = time.monotonic() - self._start
        if self._result_received or elapsed >= self._wait:
            self.get_logger().info(
                f'Done (result={"received" if self._result_received else "timeout"})'
            )
            self._done = True


def main():
    parser = argparse.ArgumentParser(description='Test face_recognition ROS2 node')
    parser.add_argument('mode', choices=['search', 'enroll'], help='test mode')
    parser.add_argument('identity', nargs='?', default=None, help='person name (enroll mode)')
    parser.add_argument('--image', default='test_image.jpg', help='test image path')
    parser.add_argument('--wait', type=float, default=3.0, help='seconds to wait for result')

    args = parser.parse_args()

    if args.mode == 'enroll' and not args.identity:
        parser.error('enroll mode requires a person name argument')

    if not Path(args.image).exists():
        print(f'[error] Image not found: {args.image}', file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = TestNode(args.image, args.mode, args.identity, args.wait)

    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
