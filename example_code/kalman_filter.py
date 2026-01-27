#from chatgpt
class SimpleKalman3D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i+3] = 1.0
        self.kf.measurementMatrix = np.hstack([np.eye(3), np.zeros((3, 3))]).astype(np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(np.array(measurement, dtype=np.float32).reshape(3, 1))

# Apply Kalman filtering per object
kf_results = {}
kalman_filters = {obj: SimpleKalman3D() for obj in results}
for obj, points in results.items():
    smoothed = []
    kf = kalman_filters[obj]
    for pt in points:
        kf.predict()
        smoothed_pt = kf.correct(pt)
        smoothed.append(smoothed_pt[:3].flatten())
    kf_results[obj] = smoothed
