#!/usr/bin/env python3
"""ROS-free markerless hand-eye calibration: CAD-to-depth point cloud alignment.

Finds T_base_cam (camera pose expressed in the robot base frame) by aligning the
gripper's CAD model, posed with the arm's forward kinematics, to the gripper's
depth point cloud seen by a fixed (eye-to-hand) camera.

Two stages, so the hardware part and the math part are independent:

  capture : live. Shows the camera, you press 'c' at each arm pose, drag a box
            around the gripper, and the script saves color + averaged depth +
            intrinsics + flange pose to  <out-dir>/pose_XXX.npz
  solve   : offline (runs anywhere, no camera or arm needed). Loads the
            pose_*.npz files, extracts gripper points, runs multi-pose ICP,
            saves hand_eye.yaml and overlay PNGs for checking.
  selftest: synthetic data, checks that the solver works. No hardware.

Examples
  python calibrate_cad_depth_noros.py selftest
  python calibrate_cad_depth_noros.py capture --arm nero --out-dir poses
  python calibrate_cad_depth_noros.py solve --pose-dir poses \
      --cad nero --mesh-dir /path/to/agx_arm_description/agx_arm_urdf

Frames: camera frame is the optical frame (x right, y down, z forward).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------- #
# Geometry / point-cloud helpers (pure NumPy / SciPy)
# --------------------------------------------------------------------------- #


def to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


def fit_table_plane_ransac(pts, n_iter=150, thresh=0.015, rng=None):
    """Dominant plane (normal, d) with normal . x + d = 0, or None."""
    if len(pts) < 100:
        return None
    rng = rng or np.random.default_rng()
    best_inliers, best_plane = 0, None
    for _ in range(n_iter):
        p1, p2, p3 = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p2 - p1, p3 - p1)
        ln = np.linalg.norm(n)
        if ln < 1e-5:
            continue
        n /= ln
        d = -float(n @ p1)
        inl = int(np.sum(np.abs(pts @ n + d) < thresh))
        if inl > best_inliers:
            best_inliers, best_plane = inl, (n, d)
    return best_plane


def remove_statistical_outliers(pts, k=20, std_ratio=1.6):
    if len(pts) < k + 1:
        return pts
    dists, _ = cKDTree(pts).query(pts, k=k + 1)
    m = dists[:, 1:].mean(axis=1)
    return pts[m < m.mean() + std_ratio * m.std()]


def extract_points(depth, K, roi, rng, max_points=1800):
    """Gripper points (camera frame) from an averaged depth image and a box."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, w, h = [int(v) for v in roi]

    # Table plane from the whole image (subsampled for speed).
    vs, us = np.where((depth > 0.4) & (depth < 1.4))
    if len(vs) > 20000:
        sel = rng.choice(len(vs), 20000, replace=False)
        vs, us = vs[sel], us[sel]
    z = depth[vs, us]
    full = np.column_stack([(us - cx) * z / fx, (vs - cy) * z / fy, z])
    plane = fit_table_plane_ransac(full, rng=rng)

    roi_d = depth[y:y + h, x:x + w]
    m = (roi_d > 0.25) & (roi_d < 1.4)
    if m.sum() < 80:
        return None
    v, u = np.where(m)
    u = u + x
    v = v + y
    z = roi_d[m]
    pts = np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])

    if plane is not None:
        n, d = plane
        kept = pts[np.abs(pts @ n + d) > 0.025]  # drop points within 25 mm of table
        if len(kept) > 100:
            pts = kept

    pts = pts[np.abs(pts[:, 2] - np.median(pts[:, 2])) < 0.10]
    pts = remove_statistical_outliers(pts)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts if len(pts) >= 80 else None


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rigid transform T with  Q ~ R P + t  (least squares)."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return to_T(R, cq - R @ cp)


def centroid_init(cad_list, obs_list) -> np.ndarray:
    """Rough T_base_cam from per-pose centroids (needs >= 4 spread-out poses)."""
    P = np.array([c.mean(0) for c in cad_list])  # base frame
    Q = np.array([o.mean(0) for o in obs_list])  # camera frame
    return np.linalg.inv(kabsch(P, Q))


def icp_multi_pose(cad_base_list, obs_cam_list, init_T_base_cam,
                   max_iter=100, tol=1e-7, max_corr_dist=0.12, min_corr_dist=0.010, decay=0.92):
    """One global T_base_cam from all poses.

    Each OBSERVED point is matched to its nearest CAD point (so CAD surfaces the
    camera cannot see do not matter). The match radius starts wide (12 cm) and
    shrinks to 1 cm, which lets it recover from an initial guess that is off by
    several cm / degrees.
    """
    cur = np.linalg.inv(init_T_base_cam)  # T_cam_base
    prev, cost = float("inf"), float("inf")
    for it in range(max_iter):
        thr = max(min_corr_dist, max_corr_dist * (decay ** it))
        Ps, Qs, dall = [], [], []
        for M_base, O in zip(cad_base_list, obs_cam_list):
            M_cam = apply_T(cur, M_base)
            d, idx = cKDTree(M_cam).query(O)
            ok = d < thr
            if ok.sum() > 20:
                Ps.append(M_cam[idx[ok]])
                Qs.append(O[ok])
                dall.append(d[ok])
        if not Ps:
            break
        cost = float(np.concatenate(dall).mean())
        cur = kabsch(np.vstack(Ps), np.vstack(Qs)) @ cur
        if thr <= min_corr_dist * 1.0001 and abs(prev - cost) < tol:
            break
        prev = cost
    return np.linalg.inv(cur), cost


def explained_cost(cad_base_list, obs_cam_list, T_base_cam, cap=0.03):
    """Mean over poses of the mean distance (capped at `cap`) from each observed
    point to the CAD surface. Lower is better; unexplained points cost `cap`."""
    T_cb = np.linalg.inv(T_base_cam)
    vals = []
    for M, O in zip(cad_base_list, obs_cam_list):
        d, _ = cKDTree(apply_T(T_cb, M)).query(O)
        vals.append(np.minimum(d, cap).mean())
    return float(np.mean(vals))


def solve_global(cad_list, obs_list, T0, n_starts=30, seed=0, verbose=True):
    """Multi-start ICP: perturb the initial guess, keep the solution that explains
    the observed points best. Returns (T_base_cam, explained_cost)."""
    rng = np.random.default_rng(seed)
    sub_c = [c[rng.choice(len(c), min(len(c), 1500), replace=False)] for c in cad_list]
    sub_o = [o[rng.choice(len(o), min(len(o), 600), replace=False)] for o in obs_list]
    cands = []
    for i in range(n_starts + 1):
        if i == 0:
            T_start = T0
        else:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            ang = np.radians(rng.uniform(0, 25))
            dT = to_T(Rotation.from_rotvec(axis * ang).as_matrix(), rng.normal(scale=0.05, size=3))
            T_start = T0 @ dT
        T_i, _ = icp_multi_pose(sub_c, sub_o, T_start)
        cands.append((explained_cost(sub_c, sub_o, T_i), T_i))
        if verbose:
            print(f"\r  multi-start search {i + 1}/{n_starts + 1}", end="", flush=True)
    if verbose:
        print()
    cands.sort(key=lambda x: x[0])
    best = None
    for _, T_i in cands[:3]:   # refine the best few with all points
        T_r, _ = icp_multi_pose(cad_list, obs_list, T_i, max_corr_dist=0.04)
        score = explained_cost(cad_list, obs_list, T_r)
        if best is None or score < best[0]:
            best = (score, T_r)
    return best[1], best[0]


def pose_residuals(cad_base_list, obs_cam_list, T_base_cam, thr=0.02):
    """Per pose: (mean distance in mm of explained observed points, fraction of
    observed gripper points within `thr` of the CAD surface)."""
    T_cb = np.linalg.inv(T_base_cam)
    out = []
    for M, O in zip(cad_base_list, obs_cam_list):
        d, _ = cKDTree(apply_T(T_cb, M)).query(O)
        ok = d < thr
        out.append((float(d[ok].mean() * 1000) if ok.any() else float("nan"), float(ok.mean())))
    return out


# --------------------------------------------------------------------------- #
# CAD model (needs trimesh)
# --------------------------------------------------------------------------- #


def build_nero_cad(mesh_dir: str, n: int = 5000) -> np.ndarray:
    """link7 + agx_gripper (closed) surface points in the link7 frame.

    mesh_dir = .../agx_arm_description/agx_arm_urdf. The fixed transforms below
    are copied unchanged from the original ROS script.
    """
    import trimesh

    def load(*p):
        return trimesh.load(os.path.join(mesh_dir, *p), force="mesh")

    m_l7 = load("nero", "meshes", "link7.stl")
    m_gb = load("agx_gripper", "meshes", "gripper_base.stl")
    m_g1 = load("agx_gripper", "meshes", "gripper_link1.stl")
    m_g2 = load("agx_gripper", "meshes", "gripper_link2.stl")

    T_l7_flange = to_T(Rotation.from_euler("xyz", [-1.5708, 0.0, -1.5708]).as_matrix(),
                       [0.031, 0.0, -0.0235])
    T_flange_gb = np.eye(4)
    T_flange_gb[2, 3] = 0.006
    T_l7_gb = T_l7_flange @ T_flange_gb
    T_gb_mesh = to_T(Rotation.from_euler("xyz", [0, 0, np.pi]).as_matrix(), np.zeros(3))
    T_gb_g1 = to_T(Rotation.from_euler("xyz", [np.pi / 2, 0.0, np.pi]).as_matrix(), [0, 0, 0.138])
    T_gb_g2 = to_T(Rotation.from_euler("xyz", [np.pi / 2, 0.0, 0.0]).as_matrix(), [0, 0, 0.138])

    s = lambda m, f: trimesh.sample.sample_surface(m, int(n * f))[0]
    return np.vstack([
        s(m_l7, 0.15),
        apply_T(T_l7_gb @ T_gb_mesh, s(m_gb, 0.45)),
        apply_T(T_l7_gb @ T_gb_g1, s(m_g1, 0.20)),
        apply_T(T_l7_gb @ T_gb_g2, s(m_g2, 0.20)),
    ])


def load_triangles(path: str) -> np.ndarray:
    """Triangles (M,3,3) from a binary STL or an OBJ file. NumPy only."""
    if path.lower().endswith(".obj"):
        verts, tris = [], []
        with open(path) as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(t) for t in line.split()[1:4]])
                elif line.startswith("f "):
                    idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                    idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                    for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                        tris.append([idx[0], idx[k], idx[k + 1]])
        v = np.asarray(verts, float)
        return v[np.asarray(tris, int)]
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[80:84], "<u4")[0])
    if 84 + 50 * n != len(raw):
        raise ValueError(f"{path}: not a binary STL (convert it, or use an OBJ)")
    rec = np.frombuffer(raw[84:84 + 50 * n],
                        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)


def sample_triangles(tri: np.ndarray, n: int, rng) -> np.ndarray:
    """Area-weighted uniform surface samples."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[idx] + r1[:, None] * (b[idx] - a[idx]) + r2[:, None] * (c[idx] - a[idx])


def build_franka_hand_cad(hand_path: str, finger_path: Optional[str] = None,
                          scale: float = 1.0, n: int = 5000, finger_open: float = 0.0) -> np.ndarray:
    """Franka Hand surface points in the FLANGE frame.

    Frames follow the MJCF (fr3_hand.xml): hand frame = flange twisted -45 deg
    about z (its 0.107 m offset is from link7 = our flange). Fingers (finger_0.obj) sit at z=0.0584
    in the hand frame and slide along +-y: left finger identity, right finger
    rotated 180 deg about z. finger_open is each finger's slide in metres
    (0 = closed, up to 0.04).
    """
    rng = np.random.default_rng(0)
    hand = sample_triangles(load_triangles(hand_path) * scale, n if finger_path is None else int(n * 0.6), rng)
    parts = [hand]
    if finger_path:
        ft = load_triangles(finger_path) * scale
        m = int(n * 0.2)
        left = sample_triangles(ft, m, rng) + np.array([0.0, finger_open, 0.0])
        right = sample_triangles(ft, m, rng) + np.array([0.0, finger_open, 0.0])
        right = right * np.array([-1.0, -1.0, 1.0])  # 180 deg about z, applied after the slide
        z = np.array([0.0, 0.0, 0.0584])
        parts += [left + z, right + z]
    pts_hand = np.vstack(parts)
    # The 0.107 m in the MJCF is measured from link7; our flange pose (franka_flange_fk)
    # is already that point, so only the 45 deg twist remains.
    R_flange_hand = Rotation.from_euler("z", -45, degrees=True).as_matrix()
    return pts_hand @ R_flange_hand.T


def build_cad_from_file(path: str, scale: float = 1.0, n: int = 5000) -> np.ndarray:
    """Any mesh already expressed in the frame whose pose you record (the flange
    frame). scale converts the mesh units to metres (0.001 for millimetre STLs)."""
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    return trimesh.sample.sample_surface(mesh, n)[0] * scale


# --------------------------------------------------------------------------- #
# Hardware: camera and arm (only used by `capture`)
# --------------------------------------------------------------------------- #


def register_depth_to_color(depth_m, Kd, Kc, R_dc, t_dc, out_shape):
    """Software depth-to-color alignment (NumPy only).

    depth_m: HxW depth in metres in the depth camera. Kd, Kc: 3x3 intrinsics of
    the depth and color cameras. R_dc, t_dc: depth->color extrinsics. Returns a
    depth image in the color camera's pixel grid (0 = no data); where several
    depth points land on one pixel the nearest wins. Lens distortion is ignored.
    """
    v, u = np.nonzero(depth_m > 0)
    z = depth_m[v, u].astype(np.float64)
    P = np.column_stack([(u - Kd[0, 2]) * z / Kd[0, 0], (v - Kd[1, 2]) * z / Kd[1, 1], z])
    Pc = P @ R_dc.T + t_dc
    ok = Pc[:, 2] > 1e-3
    Pc = Pc[ok]
    uc = np.round(Kc[0, 0] * Pc[:, 0] / Pc[:, 2] + Kc[0, 2]).astype(int)
    vc = np.round(Kc[1, 1] * Pc[:, 1] / Pc[:, 2] + Kc[1, 2]).astype(int)
    H, W = out_shape
    ok = (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
    uc, vc, zc = uc[ok], vc[ok], Pc[ok, 2]
    flat = vc * W + uc
    order = np.lexsort((zc, flat))           # by pixel, then nearest first
    _, first = np.unique(flat[order], return_index=True)
    sel = order[first]
    out = np.zeros(H * W, np.float32)
    out[flat[sel]] = zc[sel]
    return out.reshape(H, W)


class RealSenseCam:
    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)  # depth aligned to the color camera
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        sp_c = prof.get_stream(rs.stream.color).as_video_stream_profile()
        sp_d = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        i, j = sp_c.get_intrinsics(), sp_d.get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        self._Kd = np.array([[j.fx, 0, j.ppx], [0, j.fy, j.ppy], [0, 0, 1.0]])
        e = sp_d.get_extrinsics_to(sp_c)
        self._R_dc = np.array(e.rotation, float).reshape(3, 3).T  # librealsense is column-major
        self._t_dc = np.array(e.translation, float)
        self._align_failures = 0
        self._soft_align = False   # switched on if the built-in aligner keeps crashing
        self._read_failures = 0

    def read(self):
        """-> (color BGR uint8, depth in metres aligned to color), or (None, None)."""
        try:
            fs = self.pipe.wait_for_frames(5000)
        except RuntimeError as e:
            self._read_failures += 1
            if self._read_failures > 50:
                raise RuntimeError(f"camera stopped delivering frames: {e}")
            return None, None
        self._read_failures = 0
        c0, d0 = fs.get_color_frame(), fs.get_depth_frame()
        if not c0 or not d0:          # incomplete frame set: skip it
            return None, None
        if not self._soft_align:
            try:
                fr = self.align.process(fs)
                c, d = fr.get_color_frame(), fr.get_depth_frame()
                if c and d:
                    self._align_failures = 0
                    return (np.asanyarray(c.get_data()).copy(),
                            np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale)
            except RuntimeError:
                pass
            self._align_failures += 1
            if self._align_failures >= 3:
                self._soft_align = True
                print("  [info] RealSense's built-in depth alignment keeps failing; "
                      "switching to software alignment")
            return None, None
        color = np.asanyarray(c0.get_data()).copy()
        depth = np.asanyarray(d0.get_data()).astype(np.float32) * self.depth_scale
        return color, register_depth_to_color(depth, self._Kd, self.K, self._R_dc,
                                              self._t_dc, color.shape[:2])

    def averaged_depth(self, n=25, timeout=4.0):
        """Per-pixel median over n frames; suppresses sensor noise."""
        frames, t_end = [], time.monotonic() + timeout
        while len(frames) < n and time.monotonic() < t_end:
            _, d = self.read()
            if d is not None:
                frames.append(d)
        if not frames:
            return None
        stack = np.stack(frames)
        with np.errstate(all="ignore"):
            avg = np.nanmedian(np.where(stack > 0.1, stack, np.nan), axis=0)
        return np.nan_to_num(avg, nan=0.0).astype(np.float32)

    def close(self):
        self.pipe.stop()


class NeroArm:
    """Uses the same SDK calls as the original script (nero_arm.agx_sdk).

    Needs that package importable and a working CAN link. Check that importing
    it does not itself require ROS on your machine.
    """

    recheck = True

    def __init__(self, channel="can0", fw="v121"):
        from nero_arm.agx_sdk import connect, flange_from_joints

        self._fk = flange_from_joints
        self.model, _ = connect(model="nero", channel=channel, fw=fw)

    def flange_pose(self):
        q = self.model.get_joint_angles()
        if q is None or q.msg is None:
            raise RuntimeError("could not read joint angles (arm powered? CAN up?)")
        joints = list(q.msg)[:7]
        p, R = self._fk(self.model, joints)
        return np.asarray(p, float), np.asarray(R, float), np.asarray(joints, float)


class ManualArm:
    """No SDK: type the flange pose (in the base frame) for every capture.
    Useful for trying the pipeline, or when you read the pose from a teach
    pendant / another program."""

    recheck = False  # typed values cannot be re-read to detect motion

    def flange_pose(self):
        s = input("  flange pose in base frame: x y z qx qy qz qw (metres, quat xyzw): ")
        v = [float(t) for t in s.replace(",", " ").split()]
        if len(v) != 7:
            raise RuntimeError("need exactly 7 numbers")
        return np.array(v[:3]), Rotation.from_quat(v[3:]).as_matrix(), np.zeros(0)


# --- Franka FR3 / Panda: forward kinematics in pure NumPy -------------------- #
# Craig (modified) DH parameters (a, d, alpha) as I recall them from Franka's
# documentation; FR3 is documented as having the same kinematic parameters as
# the Panda. CHECK THEM against your robot with the `franka-fk` subcommand
# (compare with O_T_F / the flange pose the robot reports) before trusting a
# calibration. Result is the pose of the flange (link8) in the base frame.
_FRANKA_DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
_FRANKA_FLANGE_D = 0.107


def _dh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0.0, 0.0, 0.0, 1.0]])


def franka_flange_fk(q):
    """Flange pose (p[3], R[3x3]) in the base frame for 7 joint angles (rad)."""
    T = np.eye(4)
    for (a, d, alpha), qi in zip(_FRANKA_DH, q):
        T = T @ _dh(a, d, alpha, qi)
    T = T @ _dh(0.0, _FRANKA_FLANGE_D, 0.0, 0.0)
    return T[:3, 3].copy(), T[:3, :3].copy()


def parse_numbers(s: str) -> List[float]:
    return [float(t) for t in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


class FrankaManualArm:
    """Franka FR3/Panda without libfranka: you type the 7 joint angles, the
    script computes the flange pose itself. Read the angles from wherever the
    robot is running, e.g. on the Franka computer:
        ros2 topic echo /joint_states --once        (or franka_robot_state)
    Paste the 7 arm joint values (fingers excluded); brackets and commas are
    fine. The arm must be stationary: the script cannot detect motion here."""

    recheck = False

    def __init__(self, degrees=False):
        self.degrees = degrees

    # FR3 joint limits (rad), from fr3.xml. Used to catch wrong numbers.
    LIMITS = np.array([[-2.9007, 2.9007], [-1.8361, 1.8361], [-2.9007, 2.9007],
                       [-3.0770, -0.1169], [-2.8763, 2.8763], [0.4398, 4.6216],
                       [-3.0508, 3.0508]])

    @staticmethod
    def _flush_stdin():
        try:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)  # drop stale keystrokes
        except Exception:
            pass

    def flange_pose(self):
        unit = "degrees" if self.degrees else "radians"
        self._flush_stdin()
        print(f"  Paste the 7 arm joint angles q1..q7 in {unit}, on one line or one per line.")
        print("  (Only the arm joints, not the fingers. Press Enter on an empty line to cancel.)")
        nums: List[float] = []
        while len(nums) < 7:
            s = input("  > ")
            if not s.strip():
                raise RuntimeError("cancelled: no joint angles entered")
            nums += parse_numbers(s)
        q = np.array(nums[:7], dtype=float)
        if len(nums) > 7:
            print(f"  [note] got {len(nums)} numbers; using the first 7")
        qr = np.radians(q) if self.degrees else q
        bad = [i + 1 for i in range(7)
               if not (self.LIMITS[i, 0] - 0.05 <= qr[i] <= self.LIMITS[i, 1] + 0.05)]
        if bad:
            raise RuntimeError(
                f"joint(s) {bad} are outside the FR3 joint limits ({np.round(qr, 3).tolist()}). "
                f"Wrong values, wrong order, or degrees given as radians (use --deg)? Not recorded.")
        print(f"  joints used ({unit}): {np.round(q, 4).tolist()}")
        q = qr
        p, R = franka_flange_fk(q)
        print(f"  flange in base: x={p[0]:.4f} y={p[1]:.4f} z={p[2]:.4f} m")
        return p, R, q


# Add your own arm (e.g. Franka via franky / libfranka on Linux) by writing a
# class with  flange_pose() -> (p[3], R[3x3], joints[n])  and registering it here.
def make_arm(args):
    if args.arm == "nero":
        return NeroArm(args.channel, args.fw)
    if args.arm == "manual":
        return ManualArm()
    if args.arm == "franka":
        return FrankaManualArm(degrees=args.deg)
    raise SystemExit(f"unknown arm {args.arm}")


# --------------------------------------------------------------------------- #
# Stage 1: capture
# --------------------------------------------------------------------------- #


def capture_cmd(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cam = RealSenseCam(args.width, args.height, args.fps)
    arm = make_arm(args)
    n = len(glob.glob(os.path.join(args.out_dir, "pose_*.npz")))
    win = "Live camera: c = capture, q = quit"
    print("Click on the camera window so it receives keys.  c = capture pose, q = quit.")
    print("Keep the arm completely still from pressing 'c' until 'saved' is printed.")
    try:
        while True:
            color, _ = cam.read()
            if color is None:
                continue
            disp = color.copy()
            cv2.putText(disp, f"saved poses: {n}   c=capture  q=quit", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("c"):
                continue

            try:
                p0, R0, q0 = arm.flange_pose()  # read at the moment of the key press
            except Exception as e:
                print(f"  [error] {e}")
                continue
            frozen = color.copy()
            roi = cv2.selectROI("Drag a box around the gripper, then SPACE/ENTER",
                                frozen, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Drag a box around the gripper, then SPACE/ENTER")
            if roi[2] < 10 or roi[3] < 10:
                print("  [cancel] no box selected")
                continue
            print("  averaging depth frames ...", flush=True)
            depth = cam.averaged_depth(args.avg_frames)
            if depth is None:
                print("  [error] no depth frames")
                continue
            p1 = p0
            if getattr(arm, "recheck", False):
                try:
                    p1, _, _ = arm.flange_pose()
                except Exception:
                    p1 = p0
            moved = np.linalg.norm(p1 - p0) * 1000
            if moved > 2.0:
                print(f"  [rejected] arm moved {moved:.1f} mm during capture. Hold still and retry.")
                continue
            path = os.path.join(args.out_dir, f"pose_{n:03d}.npz")
            np.savez_compressed(path, color=frozen, depth=depth, K=cam.K,
                                roi=np.array(roi), flange_p=p0, flange_R=R0, joints=q0)
            n += 1
            print(f"  saved {path}")
    finally:
        cv2.destroyAllWindows()
        cam.close()


# --------------------------------------------------------------------------- #
# Stage 2: solve
# --------------------------------------------------------------------------- #


def parse_init(args) -> Optional[np.ndarray]:
    if args.init:
        v = [float(t) for t in args.init.replace(",", " ").split()]
        if len(v) != 7:
            raise SystemExit("--init needs: x y z qx qy qz qw  (T_base_cam)")
        return to_T(Rotation.from_quat(v[3:]).as_matrix(), v[:3])
    if args.init_yaml:
        with open(args.init_yaml) as f:
            c = yaml.safe_load(f)
        return to_T(Rotation.from_quat(c["rotation_xyzw"]).as_matrix(), c["translation"])
    return None


def draw_overlay(color, cad_base, T_base_cam, K, rms_mm):
    img = color.copy()
    cam = apply_T(np.linalg.inv(T_base_cam), cad_base)
    for p in cam[::4]:
        if p[2] > 0.1:
            u = int(round(K[0, 0] * p[0] / p[2] + K[0, 2]))
            v = int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
    cv2.putText(img, f"CAD overlay (green)  RMS {rms_mm:.2f} mm", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def solve_cmd(args):
    files = sorted(glob.glob(os.path.join(args.pose_dir, "pose_*.npz")))
    if not files:
        raise SystemExit(f"no pose_*.npz in {args.pose_dir}")
    if args.cad == "nero":
        if not args.mesh_dir:
            raise SystemExit("--cad nero needs --mesh-dir .../agx_arm_urdf")
        cad_local = build_nero_cad(args.mesh_dir)
    elif args.cad == "franka-hand":
        if not args.cad_file:
            raise SystemExit("--cad franka-hand needs --cad-file hand.stl "
                             "(and optionally --finger-mesh finger_0.obj)")
        cad_local = build_franka_hand_cad(args.cad_file, args.finger_mesh, args.cad_scale)
        print(f"  CAD: Franka hand{' + closed fingers' if args.finger_mesh else ' (no fingers)'}, "
              f"{len(cad_local)} points")
    else:
        if not args.cad_file:
            raise SystemExit("--cad file needs --cad-file mesh.stl (in the flange frame)")
        cad_local = build_cad_from_file(args.cad_file, args.cad_scale)

    rng = np.random.default_rng(0)
    cads, obs, colors, Ks, names = [], [], [], [], []
    for f in files:
        d = np.load(f)
        K = d["K"]
        pts = extract_points(d["depth"], K, d["roi"], rng)
        if pts is None:
            print(f"  skip {os.path.basename(f)}: too few valid gripper points")
            continue
        cads.append(apply_T(to_T(d["flange_R"], d["flange_p"]), cad_local))
        obs.append(pts)
        colors.append(d["color"])
        Ks.append(K)
        names.append(os.path.basename(f))
        print(f"  {names[-1]}: {len(pts)} gripper points")
    if len(cads) < 3:
        raise SystemExit("need at least 3 usable poses (8-15 recommended)")

    T0 = parse_init(args)
    if T0 is None:
        if len(cads) < 4:
            raise SystemExit("give --init or --init-yaml, or capture >= 4 poses for an automatic start")
        T0 = centroid_init(cads, obs)
        print("  [init] no initial guess given: using a rough centroid-based start")

    print(f"  start guess explains observed points with mean error "
          f"{explained_cost(cads, obs, T0) * 1000:.1f} mm (30 mm = nothing explained)")
    T, cost = solve_global(cads, obs, T0, n_starts=args.starts)
    res = pose_residuals(cads, obs, T)
    fracs = np.array([r[1] for r in res])
    rms = float(np.nanmean([r[0] for r in res]))
    q = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
    t = T[:3, 3].tolist()
    median_frac = float(np.median(fracs))
    good = median_frac >= 0.6
    print("\n" + "-" * 58)
    print(f"  translation (m)       : {[round(v, 4) for v in t]}")
    print(f"  rotation (xyzw)       : {[round(v, 4) for v in q]}")
    print(f"  error of explained pts: {rms:.2f} mm")
    print(f"  mean error, capped 30mm: {cost * 1000:.1f} mm")
    for nme, (r, frac) in zip(names, res):
        flag = "" if frac >= 0.6 else "   <-- poor"
        print(f"  {nme}: {frac * 100:3.0f}% of observed gripper points explained, {r:5.1f} mm{flag}")
    print(f"  median explained: {median_frac * 100:.0f}%  ->  "
          f"{'looks plausible' if good else 'NOT RELIABLE'}")
    print("-" * 58)

    os.makedirs(args.overlay_dir, exist_ok=True)
    for i, nme in enumerate(names):
        img = draw_overlay(colors[i], cads[i], T, Ks[i], rms)
        cv2.imwrite(os.path.join(args.overlay_dir, nme.replace(".npz", "_overlay.png")), img)
    print(f"  overlays saved to {args.overlay_dir}/  (open them and check the green dots sit on the gripper)")

    out = {
        "parent_frame": "base_link",
        "child_frame": "camera_color_optical_frame",
        "translation": [round(float(v), 6) for v in t],
        "rotation_xyzw": [round(float(v), 6) for v in q],
        "n_board_poses": len(cads),
        "method": "cad_depth_icp_noros",
        "rms_error_mm": round(float(rms), 2),
        "median_explained_fraction": round(median_frac, 3),
    }
    path = args.out if good else args.out + ".UNRELIABLE"
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=None)
    if good:
        print(f"  wrote {path}")
    else:
        print(f"  [warning] most observed points are not explained, so this is probably wrong.\n"
              f"  Saved as {path} (hand_eye.yaml was NOT written). Send me this output.")


# --------------------------------------------------------------------------- #
# Self test (synthetic data, no hardware)
# --------------------------------------------------------------------------- #


def selftest(_args=None):
    rng = np.random.default_rng(1)
    # Ellipsoid "gripper" surface in the flange frame (all axes different).
    u = rng.normal(size=(3000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cad_local = u * np.array([0.04, 0.03, 0.08]) + np.array([0.0, 0.0, 0.08])

    T_true = to_T(Rotation.from_euler("xyz", [-115, 5, 80], degrees=True).as_matrix(),
                  [0.9, 0.15, 0.45])  # T_base_cam
    T_cb = np.linalg.inv(T_true)

    cads, obs = [], []
    for _ in range(10):
        Rf = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()
        pf = rng.uniform([0.3, -0.3, 0.15], [0.6, 0.3, 0.45])
        cad_base = apply_T(to_T(Rf, pf), cad_local)
        cam = apply_T(T_cb, cad_base)
        c = cam.mean(0)
        visible = ((cam - c) @ (-c)) > 0  # roughly: the half facing the camera
        o = cam[visible] + rng.normal(scale=0.002, size=(visible.sum(), 3))
        cads.append(cad_base)
        obs.append(o)

    def err(T):
        dR = Rotation.from_matrix(T[:3, :3].T @ T_true[:3, :3]).magnitude()
        return np.degrees(dR), np.linalg.norm(T[:3, 3] - T_true[:3, 3]) * 1000

    T_bad = T_true @ to_T(Rotation.from_euler("xyz", [4, -3, 5], degrees=True).as_matrix(),
                          [0.02, -0.015, 0.02])
    T, c = icp_multi_pose(cads, obs, T_bad)
    r, t = err(T)
    print(f"ICP from a guess ~5 deg / 3 cm off : rot err {r:.2f} deg, trans err {t:.1f} mm "
          f"(matched dist {c * 1000:.2f} mm)")
    T0 = centroid_init(cads, obs)
    r0, t0 = err(T0)
    T2, c2 = icp_multi_pose(cads, obs, T0)
    r2, t2 = err(T2)
    print(f"centroid start alone               : rot err {r0:.2f} deg, trans err {t0:.1f} mm")
    print(f"ICP from the centroid start        : rot err {r2:.2f} deg, trans err {t2:.1f} mm "
          f"(matched dist {c2 * 1000:.2f} mm)")

    # Harder case: 18 poses, 25% stray points (arm, background) mixed into every box,
    # and a start that is 10 cm / 20 deg off.
    cads2, obs2 = [], []
    for _ in range(18):
        Rf = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()
        pf = rng.uniform([0.3, -0.3, 0.15], [0.6, 0.3, 0.45])
        cb = apply_T(to_T(Rf, pf), cad_local)
        cam = apply_T(T_cb, cb)
        c = cam.mean(0)
        vis = cam[((cam - c) @ (-c)) > 0]
        o = vis + rng.normal(scale=0.003, size=vis.shape)
        stray = c + rng.normal(scale=0.06, size=(len(o) // 4, 3))
        cads2.append(cb)
        obs2.append(np.vstack([o, stray]))
    T_far = T_true @ to_T(Rotation.from_euler("xyz", [12, -10, 14], degrees=True).as_matrix(),
                          [0.07, -0.05, 0.06])
    r_f, t_f = err(T_far)
    Tg, cg = solve_global(cads2, obs2, T_far, n_starts=12, verbose=False)
    rg, tg = err(Tg)
    fr = np.median([x[1] for x in pose_residuals(cads2, obs2, Tg)])
    print(f"hard case, start {r_f:.0f} deg / {t_f:.0f} mm off : multi-start gives rot err {rg:.2f} deg, "
          f"trans err {tg:.1f} mm, {fr * 100:.0f}% of observed points explained")


def franka_fk_cmd(args):
    q = np.array(parse_numbers(args.q), dtype=float)
    if len(q) != 7:
        raise SystemExit(f"need 7 joint angles, got {len(q)}")
    if args.deg:
        q = np.radians(q)
    p, R = franka_flange_fk(q)
    print(f"flange position in base (m): {np.round(p, 4).tolist()}")
    print("flange rotation matrix:")
    print(np.round(R, 4))
    print("Compare with the flange pose the robot reports for the same joints "
          "(O_T_F, or O_T_EE with the hand's 0.1034 m / 45 deg removed).")


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="live capture of poses")
    c.add_argument("--out-dir", default="poses")
    c.add_argument("--arm", choices=["nero", "manual", "franka"], default="nero")
    c.add_argument("--deg", action="store_true", help="franka: joint angles typed in degrees")
    c.add_argument("--channel", default="can0")
    c.add_argument("--fw", default="v121")
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--avg-frames", type=int, default=25)
    c.set_defaults(fn=capture_cmd)

    s = sub.add_parser("solve", help="offline ICP on captured poses")
    s.add_argument("--pose-dir", default="poses")
    s.add_argument("--out", default="hand_eye.yaml")
    s.add_argument("--overlay-dir", default="overlays")
    s.add_argument("--cad", choices=["nero", "franka-hand", "file"], default="nero")
    s.add_argument("--mesh-dir", help="agx_arm_urdf folder (for --cad nero)")
    s.add_argument("--cad-file", help="mesh in the flange frame (for --cad file)")
    s.add_argument("--cad-scale", type=float, default=1.0, help="0.001 for mm meshes")
    s.add_argument("--finger-mesh", help="franka-hand: finger_0.obj, adds both closed fingers")
    s.add_argument("--init", help='initial T_base_cam: "x y z qx qy qz qw"')
    s.add_argument("--init-yaml", help="previous hand_eye.yaml as a warm start")
    s.add_argument("--max-iter", type=int, default=90)
    s.add_argument("--starts", type=int, default=30, help="random restarts of the search")
    s.set_defaults(fn=solve_cmd)

    t = sub.add_parser("selftest", help="synthetic test, no hardware")
    t.set_defaults(fn=selftest)

    f = sub.add_parser("franka-fk", help="print the Franka flange pose for given joint angles")
    f.add_argument("--q", required=True, help='7 joint angles, e.g. "0 -0.785 0 -2.356 0 1.571 0.785"')
    f.add_argument("--deg", action="store_true")
    f.set_defaults(fn=franka_fk_cmd)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()