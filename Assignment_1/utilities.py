# importing the module
import cv2
import numpy as np

GRID_SHAPE = (6, 9)


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    img, corners = params

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates o the Shell
        # print(x, " ", y)

        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)
        corners.append((x, y))

# function to store the given 4 chessboard corner coordinates from the click inputs
def select_corners(img, num_corners=4):
    corners = []

    cv2.imshow("image", img)

    cv2.setMouseCallback("image", click_event, (img, corners))
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    if len(corners) != num_corners:
        raise ValueError(f"You must select exactly {num_corners} points on the image")
    corners_array = np.array(corners, dtype=np.float32)
    return corners_array

# function that applies linear interpolation using the manually annotated chessboard corners and returns
# all the corners points of the squares within the grid pattern
def linear_interpolation(corners, num_cells_row, num_cells_column):
    upper_left, upper_right, bottom_right, bottom_left = corners

    # Compute the width and the height given the corners
    diff_upper_right = np.subtract(upper_right, upper_left)
    diff_bottom_left = np.subtract(bottom_left, upper_left)

    width = np.linalg.norm(diff_upper_right)
    height = np.linalg.norm(diff_bottom_left)

    # Compute the step for both axes
    step_x = width / (num_cells_row - 1)
    step_y = height / (num_cells_column - 1)

    # Generate the cordinates of all the inner points of a non tilted chessboard by linear interpolation
    uniform_points = np.zeros(
        (num_cells_row * num_cells_column, 1, 2), dtype=np.float32
    )
    for j in range(num_cells_column):
        for i in range(num_cells_row):
            uniform_points[j * num_cells_row + i, 0, 0] = i * step_x
            uniform_points[j * num_cells_row + i, 0, 1] = j * step_y

    # Generate the untilted 2D rectangle with dimensions derived from the differences of the four corners inserted manually
    orig_points = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    # Finds the perspective matrix that takes the orig_points and obtain the corners provided
    transform_mat = cv2.getPerspectiveTransform(orig_points, corners)
    # Apply the transformation to all the uniform point (all the inner points)
    inner_square_coordinates = cv2.perspectiveTransform(uniform_points, transform_mat)

    return inner_square_coordinates.reshape(-1, 2)

# function to draw the XYZ world axes onto the given chessboard image
def draw_world_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 3)
    return img

# function to draw the NxNxN cube onto the given chessboard image
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 255), 2)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0,255,255), 2)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 255), 2)
    return img

# function to be used in the online phase that finds the chessboard corners automatically and projects
# the wanted shapes onto an image using the calibrated camera intrinstic parameters and distortion coefficients matrices
def draw(img, criteria, objp, mtx, dist, axis, cube_axis):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, GRID_SHAPE,None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        cubeimgpts, jac = cv2.projectPoints(cube_axis, rvecs, tvecs, mtx, dist)
        img = draw_world_axes(img,corners2,imgpts)
        img = draw_cube(img, corners2, cubeimgpts)

        return img
    else:
        return img

# function to check the quality of the given image in the offline phase and
# return an acceptance estimation based on a sharpness and brightness thresholds
def checkQuality(gray, corners, sharpness_limit, min_brightness_limit, max_brightness_limit):
    quality, _ = cv2.estimateChessboardSharpness(gray, GRID_SHAPE, corners)
    avg_sharpness, avg_min_brightness, avg_max_brightness, _ = quality
    if avg_sharpness > sharpness_limit or avg_min_brightness < min_brightness_limit or avg_max_brightness > max_brightness_limit:
        return True
    

# function that sharpens the images and helps increase the number of pictures for which
# cv2.findChessboardCorners() can find the corners automatically
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    return image_sharp