import glob
import cv2 as cv
import numpy as np

from utilities import checkQuality, linear_interpolation, select_corners, sharpen_image

# CONSTANTS
GRID_SHAPE = (6, 9)
CELL_SIZE = 19


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.prod(GRID_SHAPE), 3), np.float32)
objp[:, :2] = (
    np.mgrid[0 : GRID_SHAPE[0], 0 : GRID_SHAPE[1]].T.reshape(-1, 2)*CELL_SIZE
)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
rvecs_list = []  # Rotation vectors
tvecs_list = []  # Translation vectors

# train path
train_images = glob.glob("images/train/*.jpg")

for fname in train_images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners automatically
    ret, corners = cv.findChessboardCorners(gray, GRID_SHAPE, None)
    # check the quaity of the image
    # the check is made only in the case of automatic detection because in the manual case it is supposed that the user ensures the accuracy beforehand
    if ret and checkQuality(gray, corners, 10, 5, 400):
        continue

    # If not found: manual detection and add object points and image points
    if ret == False:
        # select manually the four corners from the the horizzontal top left corner to the bottom left clockwise
        corners = select_corners(img, 4)

        # Linearly compute the linear interpolations from the selected corners taking into accaunt the prespective
        corners2 = linear_interpolation(corners, GRID_SHAPE[1], GRID_SHAPE[0])
        corners2 = cv.cornerSubPix(gray, corners2, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)

    # If found: refine the points then add object points and image points
    else:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, GRID_SHAPE, corners2, ret)
    cv.imshow("img", img)
    cv.waitKey(0)
    if ret == False:
        cv.imshow("img", img)
        cv.waitKey(0)

cv.destroyAllWindows()

# Run 1: Calibrate the camera using all the images
ret1, mtx1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print(cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
))
# Run 2: Calibrate the camera only using ten images with automatically detected corners
objpoints2 = objpoints[:10]
imgpoints2 = imgpoints[:10]
ret2, mtx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(
    objpoints2, imgpoints2, gray.shape[::-1], None, None
)
print(cv.calibrateCamera(
    objpoints2, imgpoints2, gray.shape[::-1], None, None
))
# Run 3: Calibrate the camera only using five out of the ten images from Run 2
objpoints3 = objpoints2[:5]
imgpoints3 = imgpoints2[:5]
ret3, mtx3, dist3, rvecs3, tvecs3 = cv.calibrateCamera(
    objpoints3, imgpoints3, gray.shape[::-1], None, None
)

# Save the intrinsics parameters for all the runs
np.savez(
    "camera_intrinsics.npz",
    mtx1=mtx1,
    mtx2=mtx2,
    mtx3=mtx3,
    dist1=dist1,
    dist2=dist2,
    dist3=dist3,
)

# Save the extrinsics parameters for all the runs
np.savez(
    "camera_extrinsics.npz",
    rvecs1=rvecs1,
    tvecs1=tvecs1,
    rvecs2=rvecs2,
    tvecs2=tvecs2,
    rvecs3=rvecs3,
    tvecs3=tvecs3,
)

print("Calibration completed successfully.")
