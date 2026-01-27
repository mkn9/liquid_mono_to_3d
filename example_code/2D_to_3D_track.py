#from chatgpt

import numpy as np
import cv2

# === Intrinsic camera matrices for both cameras ===
K1 = np.array([[1000, 0, 640],
               [0, 1000, 360],
               [0, 0, 1]])

K2 = np.array([[1000, 0, 640],
               [0, 1000, 360],
               [0, 0, 1]])

# === Extrinsic parameters (rotation and translation) ===
# Camera 1 is the origin
R1 = np.eye(3)
t1 = np.zeros((3, 1))

# Camera 2 is translated along X axis by 0.5 meters
R2 = np.eye(3)
t2 = np.array([[0.5], [0], [0]])

# === Projection matrices ===
P1 = K1 @ np.hstack((R1, t1))
P2 = K2 @ np.hstack((R2, t2))

# === Simulated 2D observations of a point across time (in pixels) ===
# Format: [(u1, v1), (u2, v2)] for each frame
sensor1_track = [(650, 370), (655, 375), (660, 380)]
sensor2_track = [(630, 370), (635, 375), (640, 380)]

# === Function to triangulate 3D points from two views ===
def triangulate_tracks(sensor1_track, sensor2_track, P1, P2):
    points_3d = []
    for pt1, pt2 in zip(sensor1_track, sensor2_track):
        p1 = np.array(pt1, dtype=np.float32).reshape(2, 1)
        p2 = np.array(pt2, dtype=np.float32).reshape(2, 1)

        point_homog = cv2.triangulatePoints(P1, P2, p1, p2)
        point_3d = (point_homog[:3] / point_homog[3]).flatten()  # Convert from homogeneous
        points_3d.append(point_3d)
    return np.array(points_3d)

# === Estimate 3D track ===
track_3d = triangulate_tracks(sensor1_track, sensor2_track, P1, P2)

# === Print results ===
for i, point in enumerate(track_3d):
    print(f"Frame {i}: 3D Position = {point}")
