import numpy as np
import cv2
import glob
import sys
import json


folder = 'left_images_subset/'
do_resize = True
if do_resize:
    resize_size = (640, 480)
save_filename = '_intr_640_480.json'




which_cam = ''

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((8*6,3), np.float32)
# objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(folder + which_cam + '*.png')

for index, fname in enumerate(images):
    # print(index, fname)
    img = cv2.imread(fname)
    if do_resize:
        img = cv2.resize(img, resize_size, interpolation = cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)        
        # img = cv2.drawChessboardCorners(img, (6,8), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(100)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# print(mtx)
# print(dist)
# print(rvecs)
# print(tvecs)

mean_error = 0
tot_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", float(mean_error)/float(len(objpoints)), tot_error

intr_dict = {}
intr_dict['cam_mat'] = np.asarray(mtx).tolist()
intr_dict['dist_coeff'] = np.asarray(dist).tolist()
intr_dict['repr_err'] = float(mean_error)/float(len(objpoints))

print(intr_dict, len(objpoints))

with open(folder + which_cam + save_filename, 'w') as f:
	json.dump(intr_dict, f)

cv2.destroyAllWindows()
