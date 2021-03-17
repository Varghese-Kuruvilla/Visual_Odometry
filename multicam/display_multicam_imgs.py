import os
import shutil
import glob
import sys
import numpy as np
import cv2

'''
Splitting multicam images captured for 3 camera sources. 
This script is just to display images, does not split data. It can be used to print the start and stop of sequence. Use other script after this to write.

For this, the input is a folder of images of the format : camx_timestamp.jpeg where timestamp is a float; eg: cam2_13234.234.jpeg.
Cam1 is for center cam
Cam2 is for left cam (for downfacing camera maybe)
Cam3 is for right cam (for right camera)

Press:
    a. 'b' to print file index when the start of a sequence is encountered.
    b. 'e' to print file index when end of sequence is encountered.
    c. 'i' to print file index for some intermediate frame (not needed for us)
    d. 'w' to change waitkey value to 0. i.e. manually iterate.
    e. 'r' to revert waitkey value back to 1. (video display mode kindof)
    f. 'q' to quit.
'''

folder = 'foldername/'

center_cam_filenames = glob.glob(folder + 'cam1_*.jpeg')
center_cam_timestamps =  [float('.'.join(filename.split('_')[-1].split('.')[:2])) for filename in center_cam_filenames]

side_cam_filenames_left = glob.glob(folder + 'cam2_*.jpeg')
side_cam_timestamps_left =  [float('.'.join(filename.split('_')[-1].split('.')[:2])) for filename in side_cam_filenames_left]

side_cam_filenames_right = glob.glob(folder + 'cam3_*.jpeg')
side_cam_timestamps_right =  [float('.'.join(filename.split('_')[-1].split('.')[:2])) for filename in side_cam_filenames_right]


sorted_center_indices = np.argsort(center_cam_timestamps)
center_cam_timestamps = np.asarray(center_cam_timestamps)

sorted_side_indices_left = np.argsort(side_cam_timestamps_left)
side_cam_timestamps_left = np.asarray(side_cam_timestamps_left)

sorted_side_indices_right = np.argsort(side_cam_timestamps_right)
side_cam_timestamps_right = np.asarray(side_cam_timestamps_right)

start_write = 0
waitkeynum = 1

begins = []
ends = []
intermediates = []
for counter, (index1, index2, index3) in enumerate(zip(sorted_center_indices, sorted_side_indices_left, sorted_side_indices_right)):

    # print(center_cam_timestamps[index1], side_cam_timestamps[index2])
    center_filename = center_cam_filenames[index1]
    img = cv2.imread(center_filename)
    cv2.imshow('CENTER_IMAGE', img)

    side_filename = side_cam_filenames_left[index2]
    img = cv2.imread(side_filename)
    cv2.imshow('SIDE_IMAGE_LEFT', img)

    side_filename = side_cam_filenames_right[index3]
    img = cv2.imread(side_filename)
    cv2.imshow('SIDE_IMAGE_RIGHT', img)

    ch = cv2.waitKey(waitkeynum)
    if ch == ord('b'):
        print("Begin : ", counter)
        begins.append(counter)

    if ch == ord('e'):
        print("End : ", counter)
        ends.append(counter)

    if ch == ord('i'):
        print("Intermediate : ", counter)
        intermediates.append(counter)

    if ch == ord('q'):
        break
    if ch == ord('w'):
        waitkeynum = 0
    if ch == ord('r'):
        waitkeynum = 1

print(begins, ends)
cv2.destroyAllWindows()