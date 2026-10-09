"""Bonus: perturb the camera extrinsic yaw and measure innovation / gating / RMSE.

The detector runs once; its detections are cached and the tracker is replayed
for every yaw offset with the same camera-noise seed. Code in ``platform/`` and
``student/workspace`` is untouched; the yaw offset is applied only to the
``Sensor`` object used by the tracker (the pixels still come from the true
ground-truth boxes). Output goes to ``student/bonus/``, never to ``artifacts/``.

Run from the repo root:
    python student/bonus/calibration_sweep.py --config student/config/paths.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

from fusion_lab import tracking_params
from fusion_lab.evaluation import (
    aggregate_records, detection_counts, tracking_counts, valid_ground_truth,
)
from fusion_lab.scripts import run_lab
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace

YAW_DEG = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


class RecordingFilter(Filter):
    """Filter that stores the innovation of every camera update it applies."""

    def __init__(self, kalman_mod):
        super().__init__(kalman_mod)
        self.camera_gamma, self.camera_d2 = [], []

    def update(self, track, meas):
        if meas.sensor.name == "camera":
            H = meas.sensor.get_H(track.x)
            gamma = self._k.innovation(track.x, meas)
            S = self._k.innovation_covariance(track.P, meas, H)
            self.camera_gamma.append(np.asarray(gamma).ravel())
            self.camera_d2.append(float(np.asarray(gamma.T @ np.linalg.inv(S) @ gamma).item()))
        super().update(track, meas)


def rotate_extrinsic(sensor: Sensor, yaw_deg: float) -> None:
    """Rotate the camera about the vehicle z axis (wrong extrinsic yaw)."""
    a = np.deg2rad(yaw_deg)
    Rz = np.eye(4)
    Rz[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T = Rz @ np.asarray(sensor.sens_to_veh)
    sensor.sens_to_veh = np.asmatrix(T)
    sensor.veh_to_sens = np.asmatrix(np.linalg.inv(T))


def cache_detections(cfg, ws, frame_start, frame_end):
    """Run the detector once and keep everything the tracker replay needs."""
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils
    from fusion_lab.lidar_pcl import pcl_from_range_image

    det_pipe, bev = ws["detection_pipeline"], ws["bev_mapping"]
    weights = run_lab._resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)
    tfrecord = Path(cfg["waymo_dir"]) / cfg["segment"]
    cache, calibs = [], {}
    for cnt, frame in enumerate(WaymoDataFileReader(str(tfrecord))):
        if cnt < frame_start:
            continue
        if cnt > frame_end:
            break
        if not calibs:
            calibs["lidar"] = waymo_utils.get(
                frame.context.laser_calibrations, dataset_pb2.LaserName.TOP)
            calibs["camera"] = waymo_utils.get(
                frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)
        group = next((g for g in frame.camera_labels
                      if g.name == dataset_pb2.CameraName.FRONT), None)
        centres = None if group is None else [
            np.array([l.box.center_x, l.box.center_y]) for l in group.labels
            if l.type == label_pb2.Label.Type.TYPE_VEHICLE]
        cache.append((cnt, detections, labels, centres))
        print(f"detected frame {cnt}", flush=True)
    return cache, calibs, det_cfg, tfrecord.name


def replay(ws, cache, calibs, det_cfg, mode, yaw_deg, seed):
    """Replay the tracker on cached detections and return (metrics, camera stats)."""
    cam = ws["camera_fusion"]
    KF = RecordingFilter(ws["kalman"])
    manager = TrackManager(ws["track_management"])
    assoc = ws["association"]
    rng = np.random.default_rng(seed)
    lidar = Sensor("lidar", calibs["lidar"], cam)
    camera = None
    if mode == "fused":
        camera = Sensor("camera", calibs["camera"], cam)
        rotate_extrinsic(camera, yaw_deg)
    records, n_meas = [], 0
    for cnt, detections, labels, centres in cache:
        obs = run_lab._lidar_observations(cnt, detections, lidar, det_cfg)
        for track in manager.track_list:
            KF.predict(track)
            track.set_t(cnt * tracking_params.dt)
        assoc.associate_and_update(manager, obs, KF, lidar)
        if camera is not None and centres is not None:
            cam_obs = []
            for centre in centres:  # same rng draw order as run_lab
                camera.generate_measurement(cnt, centre + rng.normal(0, 0.5, 2), cam_obs)
            n_meas += len(cam_obs)
            assoc.associate_and_update(manager, cam_obs, KF, camera)
        records.append({
            "mode": mode, "frame": cnt, **detection_counts(labels, detections, ws["detection_metrics"]),
            "valid_gt": len(labels), **tracking_counts(manager.track_list, labels),
        })
    metrics = aggregate_records(records, mode, [cache[0][0], cache[-1][0]], seed, "bonus")
    d2 = np.array(KF.camera_d2)
    gamma = np.array(KF.camera_gamma).reshape(-1, 2)
    stats = {
        "camera_measurements": n_meas,
        "camera_updates": int(len(d2)),
        "not_used_by_gate_or_fov": int(n_meas - len(d2)),
        "mean_abs_innovation_px_i": float(np.abs(gamma[:, 0]).mean()) if len(d2) else None,
        "mean_abs_innovation_px_j": float(np.abs(gamma[:, 1]).mean()) if len(d2) else None,
        "mean_signed_innovation_px_i": float(gamma[:, 0].mean()) if len(d2) else None,
        "mean_d2": float(d2.mean()) if len(d2) else None,
        "chi2_gate_d2": float(chi2.ppf(tracking_params.gating_threshold, 2)),
    }
    return metrics["tracking"][mode], stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("student/config/paths.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("student/bonus/calibration_sweep.json"))
    args = parser.parse_args()
    run_lab._setup_import_paths()
    cfg = run_lab._load_paths_config(args.config)
    cfg.setdefault("segment", "training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord")
    ws = load_workspace()
    start, end = int(cfg.get("frame_start", 0)), int(cfg.get("frame_end", 20))
    cache, calibs, det_cfg, segment = cache_detections(cfg, ws, start, end)

    results = {"segment": segment, "frames": [start, end], "seed": args.seed, "rows": []}
    lidar_metrics, _ = replay(ws, cache, calibs, det_cfg, "lidar", 0.0, args.seed)
    results["lidar_only"] = lidar_metrics
    for yaw in YAW_DEG:
        tracking, stats = replay(ws, cache, calibs, det_cfg, "fused", yaw, args.seed)
        results["rows"].append({"yaw_deg": yaw, "tracking": tracking, "camera": stats})
        print(f"yaw={yaw}: rmse={tracking['rmse']:.4f} updates={stats['camera_updates']}"
              f"/{stats['camera_measurements']} mean_d2={stats['mean_d2']}", flush=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
