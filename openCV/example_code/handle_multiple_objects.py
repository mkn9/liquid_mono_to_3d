#from chatgpt
# Tracks per object
sensor1_tracks = {
    'obj1': [(650, 370), (655, 375), (660, 380)],
    'obj2': [(620, 360), (625, 365), (630, 370)],
}
sensor2_tracks = {
    'obj1': [(630, 370), (635, 375), (640, 380)],
    'obj2': [(600, 360), (605, 365), (610, 370)],
}

# Triangulate for each object
results = {}
for obj in sensor1_tracks:
    points_3d = []
    for pt1, pt2 in zip(sensor1_tracks[obj], sensor2_tracks[obj]):
        point_3d = triangulate_2d_points(pt1, pt2, P1, P2)
        points_3d.append(point_3d)
    results[obj] = points_3d
