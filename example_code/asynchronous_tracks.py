#from chatgpt
def synchronize_tracks(track1, track2, timestamps1, timestamps2):
    """Synchronize two tracks by finding the closest timestamp matches."""
    synced = []
    for t1, pt1 in zip(timestamps1, track1):
        closest_idx = np.argmin(np.abs(np.array(timestamps2) - t1))
        pt2 = track2[closest_idx]
        synced.append((pt1, pt2))
    return synced

# Example usage
timestamps1 = [0.0, 1.0, 2.0]
timestamps2 = [0.1, 1.1, 2.1]
track1 = [(650, 370), (655, 375), (660, 380)]
track2 = [(630, 370), (635, 375), (640, 380)]

synchronized_pairs = synchronize_tracks(track1, track2, timestamps1, timestamps2)
