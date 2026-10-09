"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

params = get_tracking_params()
_MIN_DEPTH = 1e-6


def _position_in_sensor(x: Matrix, sensor: Any) -> np.ndarray:
    """Return the vehicle-frame position (first 3 state entries) in sensor axes."""
    p = np.asarray(x, dtype=float).reshape(-1)[:3]
    T = np.asarray(sensor.veh_to_sens)
    return T[:3, :3] @ p + T[:3, 3]


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    x_s, y_s, z_s = _position_in_sensor(x, sensor)
    if not np.isfinite([x_s, y_s, z_s]).all():
        return False
    if sensor.name == "camera" and x_s <= _MIN_DEPTH:
        return False
    angle = np.arctan2(y_s, x_s)
    return bool(sensor.fov[0] <= angle <= sensor.fov[1])


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    x_s, y_s, z_s = _position_in_sensor(x, sensor)
    # Check before dividing: depth x_s is the denominator of the pinhole model.
    if not np.isfinite([x_s, y_s, z_s]).all() or x_s <= _MIN_DEPTH:
        raise ValueError(
            "Camera projection needs finite coordinates and depth > "
            f"{_MIN_DEPTH:g}; sensor position=({x_s}, {y_s}, {z_s})"
        )
    # Waymo camera axes: x forward, y left, z up -> image i grows to the right.
    u = sensor.c_i - sensor.f_i * y_s / x_s
    v = sensor.c_j - sensor.f_j * z_s / x_s
    return np.matrix([[u], [v]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    z_mat = np.matrix([[float(z[0])], [float(z[1])]])
    R = np.matrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2]))
    return {"z": z_mat, "R": R, "sensor": sensor}
