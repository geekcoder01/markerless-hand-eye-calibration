#!/usr/bin/env python3
"""Check a hand-eye calibration with an ArUco marker.

Live mode (needs the RealSense; on macOS run with sudo):
    sudo -E env "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH" $(which python) aruco_verify.py \
        --yaml hand_eye.yaml --dict DICT_5X5_1000 --id 0

    Marker centre: intersection of the two diagonals of the detected marker.
    Depth: median depth in a small patch around that pixel.
    Keys: s = average 30 frames, print camera and base xyz with their spread,
              append the result to aruco_results.jsonl
          q = quit

Robot mode (no camera): the gripper tip position in the base frame, for the
joint angles you read when the tip touches the marker centre, and its
distance to the last camera measurement in aruco_results.jsonl:
    python aruco_verify.py --q "q1 q2 q3 q4 q5 q6 q7"

Joints must be in order 1..7. /joint_states does NOT list them in that order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# T_base_cam used when the YAML file is missing (run with --yaml none to force it).
T_BASE_CAM_PLACEHOLDER = np.array([
    [ 0.652174,  0.757838, -0.018744, -0.306555],
    [ 0.439892, -0.398465, -0.804811,  0.057062],
    [-0.617385,  0.516631, -0.593235,  1.475168],
    [ 0.0,       0.0,       0.0,       1.0     ],
])

# Franka Hand TCP: 0.1034 m along the flange z axis (centre between the
# fingertips of the closed gripper).
DEFAULT_TIP_OFFSET = 0.1034


def load_T(path):
    if not os.path.exists(path):
        print(f"  [info] {path} not found: using the T_BASE_CAM_PLACEHOLDER matrix "
              "from the top of aruco_verify.py.")
        return T_BASE_CAM_PLACEHOLDER.copy(), False
    if path.endswith(".UNRELIABLE"):
        print(f"  [WARN] {path} is marked UNRELIABLE; expect large errors.")
    import yaml
    from scipy.spatial.transform import Rotation
    with open(path) as f:
        c = yaml.safe_load(f)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(c["rotation_xyzw"]).as_matrix()
    T[:3, 3] = c["translation"]
    return T, True


def marker_centre(corners):
    """Intersection of the diagonals 0-2 and 1-3 of a 4x2 corner array."""
    p0, p1, p2, p3 = corners.astype(float)
    d1, d2 = p2 - p0, p3 - p1
    A = np.array([d1, -d2]).T
    try:
        s, _ = np.linalg.solve(A, p1 - p0)
    except np.linalg.LinAlgError:
        return corners.mean(axis=0)
    return p0 + s * d1


def patch_depth(depth, u, v, r=3):
    H, W = depth.shape
    ui, vi = int(round(u)), int(round(v))
    patch = depth[max(vi - r, 0):min(vi + r + 1, H), max(ui - r, 0):min(ui + r + 1, W)]
    good = patch[(patch > 0.1) & (patch < 5.0)]
    return float(np.median(good)) if good.size >= 3 else None


def deproject(K, u, v, z):
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def make_detector(dict_name):
    import cv2
    if not hasattr(cv2.aruco, dict_name):
        raise SystemExit(f"unknown dictionary {dict_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dictionary, params)
    return det


def detect_all(det, color):
    """-> {marker id: 4x2 corner array} for every marker seen."""
    import cv2
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten())}


def detect(det, color, marker_id):
    return detect_all(det, color).get(marker_id)


def measure(cam, det, marker_id, K, T, n=30, timeout=6.0):
    """Average n frames. Returns dict or None."""
    cams, t_end = [], time.monotonic() + timeout
    while len(cams) < n and time.monotonic() < t_end:
        color, depth = cam.read()
        if color is None:
            continue
        c = detect(det, color, marker_id)
        if c is None:
            continue
        u, v = marker_centre(c)
        z = patch_depth(depth, u, v)
        if z is None:
            continue
        cams.append(deproject(K, u, v, z))
    if len(cams) < max(5, n // 3):
        return None
    P = np.array(cams)
    Pb = P @ T[:3, :3].T + T[:3, 3]
    return {"n": len(P), "cam": P.mean(0), "cam_std": P.std(0),
            "base": Pb.mean(0), "base_std": Pb.std(0)}


def live(args):
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from calibrate import RealSenseCam

    T, real = load_T(args.yaml)
    det = make_detector(args.dict)
    cam = RealSenseCam()
    K = cam.K
    print("  s = measure (30 frames), q = quit. Click the window first.")
    win = "aruco_verify"
    try:
        while True:
            color, depth = cam.read()
            if color is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            img = color.copy()
            found = detect_all(det, color)
            c = found.get(args.id)
            if c is not None:
                cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
                u, v = marker_centre(c)
                cv2.drawMarker(img, (int(round(u)), int(round(v))), (0, 0, 255),
                               cv2.MARKER_CROSS, 24, 2)
                z = patch_depth(depth, u, v)
                if z is not None:
                    pb = T[:3, :3] @ deproject(K, u, v, z) + T[:3, 3]
                    txt = f"z={z:.3f} m  base=[{pb[0]:.3f} {pb[1]:.3f} {pb[2]:.3f}]"
                else:
                    txt = "no depth at marker centre"
                cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                msg = f"marker id {args.id} not found"
                if found:   # other markers visible: show them so --id can be fixed
                    msg += f"; seen ids: {sorted(found)} (use --id)"
                    for i, ci in found.items():
                        cv2.polylines(img, [ci.astype(np.int32)], True, (0, 165, 255), 2)
                        cv2.putText(img, str(i), tuple(ci[0].astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(win, img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                print("  measuring... keep everything still")
                r = measure(cam, det, args.id, K, T, n=args.frames)
                if r is None:
                    print("  measurement failed: marker or depth missing in too many frames")
                    continue
                mm = lambda a: " ".join(f"{x * 1000:.1f}" for x in a)
                print(f"  frames used : {r['n']}")
                print(f"  CAM  xyz (m): {np.round(r['cam'], 4).tolist()}   spread (mm): {mm(r['cam_std'])}")
                print(f"  BASE xyz (m): {np.round(r['base'], 4).tolist()}   spread (mm): {mm(r['base_std'])}"
                      + ("" if real else "   (placeholder matrix)"))
                rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "marker_id": args.id,
                       "yaml": args.yaml if real else "placeholder", "frames": r["n"],
                       "cam_xyz": r["cam"].tolist(), "cam_std": r["cam_std"].tolist(),
                       "base_xyz": r["base"].tolist(), "base_std": r["base_std"].tolist()}
                with open(args.results, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  appended to {args.results}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


def robot(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from calibrate import franka_flange_fk, parse_numbers

    q = np.array(parse_numbers(args.q), float)
    if len(q) != 7:
        raise SystemExit(f"need 7 joint angles in order 1..7, got {len(q)}")
    if args.deg:
        q = np.radians(q)
    p, R = franka_flange_fk(q)
    tip = p + R[:, 2] * args.tip_offset
    print(f"  flange (m)          : {np.round(p, 4).tolist()}")
    print(f"  tip, +{args.tip_offset} m on z: {np.round(tip, 4).tolist()}")
    if not os.path.exists(args.results):
        print(f"  no {args.results} yet: measure the marker with the camera first")
        return
    with open(args.results) as f:
        lines = [l for l in f if l.strip()]
    last = json.loads(lines[-1])
    cam_b = np.array(last["base_xyz"])
    d = tip - cam_b
    print(f"  camera BASE xyz (m) : {np.round(cam_b, 4).tolist()}   ({last['time']})")
    print(f"  difference tip-cam  : {np.round(d * 1000, 1).tolist()} mm,  "
          f"distance {np.linalg.norm(d) * 1000:.1f} mm")
    e = np.linalg.norm(d)
    verdict = ("good (< 2 cm)" if e < 0.02 else
               "usable, retake poses with a clean background (2-5 cm)" if e < 0.05 else
               "do not trust this calibration (> 5 cm)")
    print(f"  verdict             : {verdict}")
    print(f"  transform used      : {last.get('yaml')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", default="hand_eye.yaml")
    ap.add_argument("--dict", default="DICT_5X5_1000",
                    help="marker dictionary; DICT_5X5_1000 also finds markers from DICT_5X5_50/100/250")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--results", default="aruco_results.jsonl")
    ap.add_argument("--q", help="robot mode: 7 joint angles (order 1..7) at the moment the tip touches the marker")
    ap.add_argument("--deg", action="store_true", help="--q is in degrees")
    ap.add_argument("--tip-offset", type=float, default=DEFAULT_TIP_OFFSET,
                    help="tip distance from the flange along its z axis, metres (default 0.1034, Franka Hand TCP)")
    args = ap.parse_args()
    if args.q:
        robot(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
