import glob

import cv2 as cv
import numpy as np

from utilities import draw_cube, draw_world_axes, draw, sharpen_image

# CONSTANTS
GRID_SHAPE = (6, 9)
WEBCAM = True

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points and set the axis and cube axis
objp = np.zeros((np.prod(GRID_SHAPE), 3), np.float32)
objp[:, :2] = np.mgrid[0 : GRID_SHAPE[0], 0 : GRID_SHAPE[1]].T.reshape(-1, 2)

# desired measurements of the world axes and the cube size
axis = np.float32([[5, 0, 0], [0, 5, 0], [0, 0, -5]]).reshape(-1, 3)
cube_axis = np.float32(
    [
        [0, 0, 0],
        [0, 3, 0],
        [3, 3, 0],
        [3, 0, 0],
        [0, 0, -3],
        [0, 3, -3],
        [3, 3, -3],
        [3, 0, -3],
    ]
)

# Load previously sinstrisics values
with np.load("camera_intrinsics.npz") as X:
    mtx1, dist1, mtx2, dist2, mtx3, dist3 = [
        X[i] for i in ("mtx1", "dist1", "mtx2", "dist2", "mtx3", "dist3")
    ]

mtx = np.array([mtx1, mtx2, mtx3])
dist = np.array([dist1, dist2, dist3])


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((np.prod(GRID_SHAPE), 3), np.float32)
objp[:,:2] = np.mgrid[0:GRID_SHAPE[0],0:GRID_SHAPE[1]].T.reshape(-1,2)
axis = np.float32([[5,0,0], [0,5,0], [0,0,-5]]).reshape(-1,3)
cube_axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

# if the user wants to use the live webcam
if (WEBCAM == True):
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        raise IOError("Webcam not accessible")
    while True:
        ret, frame = cap.read()
        copyFrame = frame.copy()
        img = draw(copyFrame, criteria, objp, mtx[0], dist[0], axis, cube_axis)
        cv.imshow('img', img)
        key = cv.waitKey(1)

        if key == ord("q"):
            break
    cap.release()

# if the user wants to use a stored image instead of the webcam live feed
else:    
    for i in range(len(mtx)):
        print(mtx[i], dist[i])
        img = cv.imread('Images/test/test4.jpg')
        img = draw(img, criteria, objp, mtx[i], dist[i], axis, cube_axis)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord("s"):
            cv.imwrite(f"test_image_run_" + str(i+1) + ".jpg",img)
            
cv.destroyAllWindows()