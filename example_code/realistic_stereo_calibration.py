#from chatgpt
#https://chatgpt.com/c/6842558e-6094-8010-8d2b-a3e6c2943f3a
import numpy as np
import cv2
import os
import glob

# === CONFIGURATION ===
checkerboard_size = (9, 6)
square_size = 40  # pixels per square
num_images = 10
image_size = (640, 480)

# === STEP 1: GENERATE SYNTHETIC CHECKERBOARD IMAGES ===
def generate_checkerboard_image(cb_size, square_px):
    rows, cols = cb_size[1] + 1, cb_size[0] + 1
    img = np.zeros((rows * square_px, cols * square_px), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                cv2.rectangle(img,
                              (c * square_px, r * square_px),
                              ((c + 1) * square_px, (r + 1) * square_px),
                              255, -1)
    return img

def generate_stereo_images():
    os.makedirs("calibration_images", exist_ok=True)
    cb = generate_checkerboard_image(checkerboard_size, square_size)
    cb = cv2.resize(cb, image_size)

    for i in range(num_images):
        dx = i % 3 + 1  # simulate small shifts between cameras
        left = np.roll(cb, shift=-dx, axis=1)
        right = np.roll(cb, shift=dx, axis=1)

        cv2.imwrite(f"calibration_images/left_{i:02d}.png", left)
        cv2.imwrite(f"calibration_images/right_{i:02d}.png", right)

generate_stereo_images()

# === STEP 2: PREPARE OBJECT POINTS ===
objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
objp[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)
objp *= square_size / 1000.0  # convert pixels to meters

objpoints = []
imgpoints_left = []
imgpoints_right = []

images_left = sorted(glob.glob('calibration_images/left_*.png'))
images_right = sorted(glob.glob('calibration_images/right_*.png'))

# === STEP 3: DETECT CORNERS IN IMAGE PAIRS ===
for fname_left, fname_right in zip(images_left, images_right):
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_left, checkerboard_size)
    ret_r, corners_r = cv2.findChessboardCorners(gray_right, checkerboard_size)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)

# === STEP 4: CALIBRATE MONO CAMERAS ===
ret_l, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_r, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# === STEP 5: STEREO CALIBRATION ===
_, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K1, D1, K2, D2, gray_left.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

# === STEP 6: STEREO RECTIFICATION ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, gray_left.shape[::-1], R, T, alpha=0)

left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, gray_right.shape[::-1], cv2.CV_16SC2)

# === STEP 7: SAVE OUTPUT ===
np.savez('calibration_output_checkerboard.npz',
         K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         left_map1=left_map1, left_map2=left_map2,
         right_map1=right_map1, right_map2=right_map2)

print("✅ Calibration complete. Output saved to calibration_output_checkerboard.npz")
