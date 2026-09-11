#!/usr/bin/env python3
"""Check full ROS-published trajectories against the eight base-test traces."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Path as PathMessage
from visualization_msgs.msg import MarkerArray

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', required=True, type=Path)
parser.add_argument('--case', help='Check only this artifact case')
parser.add_argument('--artifacts-root', type=Path, help='Override the base-artifacts directory')
args = parser.parse_args()
build = args.build.resolve()
rclpy.init()
qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
try:
    artifact_root = args.artifacts_root.resolve() if args.artifacts_root else build / 'base-artifacts'
    for case in sorted(artifact_root.iterdir()):
        if args.case and case.name != args.case:
            continue
        bundle = case / case.name
        if not (bundle / 'trace.csv').exists():
            continue
        with (bundle / 'trace.csv').open() as stream:
            ego = [row for row in csv.DictReader(stream) if row['actor'] == 'ego']
        node = rclpy.create_node('base_replay_check_' + bundle.name)
        messages = []
        marker_messages = []
        marker_sub = node.create_subscription(MarkerArray, '/dro_mpc/markers', marker_messages.append, qos)
        sub = node.create_subscription(PathMessage, '/dro_mpc/ego_path', messages.append, qos)
        env = dict(os.environ, ROS_LOG_DIR='/tmp/dro-base-ros-logs')
        process = subprocess.Popen([str(build / 'dro_mpc_rviz_replay'),
                                    '--artifact', str(bundle), '--rate', '20'], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if messages and len(messages[-1].poses) == len(ego):
                    break
            assert messages, 'no ROS trajectory received'
            points = messages[-1].poses
            assert len(points) == len(ego), (len(points), len(ego))
            for point, row in zip(points, ego):
                assert abs(point.pose.position.x - float(row['x'])) < 1e-12
                assert abs(point.pose.position.y - float(row['y'])) < 1e-12
            with (bundle / 'sampled_scenarios.csv').open() as stream:
                expected_samples = bool(list(csv.DictReader(stream)))
            visible_samples = [m for message in marker_messages for m in message.markers
                               if m.ns == 'sampled_scenarios']
            assert bool(visible_samples) == expected_samples, 'sample overlay missing/unexpected'
            for message in marker_messages:
                colors = {(m.color.r, m.color.g, m.color.b) for m in message.markers if m.ns == 'obstacle'}
                for marker in message.markers:
                    if marker.ns == 'sampled_scenarios':
                        assert (marker.color.r, marker.color.g, marker.color.b) in colors
                        assert marker.points, 'empty scenario marker'
            print(f'PASS {bundle.name}: ROS trajectory matches all {len(ego)} recorded ego states', flush=True)
        finally:
            process.terminate()
            output, _ = process.communicate(timeout=5)
            if process.returncode not in (0, -15):
                print(output)
            node.destroy_node()
finally:
    rclpy.shutdown()
