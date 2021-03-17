import cv2
import os
import shutil
import glob
import numpy as np

'''
Splitting multicam images captured for 3 camera sources. Specify the folder, and the start and end numbers obtained from previous script.
'''

starts = [52, 786, 1592, 52, 786, 1592, 52, 786, 1592]
stops = [220, 1020, 1791, 220, 1020, 1791, 220, 1020, 1791]

dispno = 0
ctr = 0
framectr = 0
start = False
start_frame = 0

images_folder = 'foldername/'

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

segments_covered =  0
segments_idx = 0
covering_segment = False

for counter, (index1, index2, index3) in enumerate(zip(sorted_center_indices, sorted_side_indices_left, sorted_side_indices_right)):

    center_filename = center_cam_filenames[index1]
    side_filename_left = side_cam_filenames_left[index2]
    side_filename_right = side_cam_filenames_right[index3]

    framectr += 1
    if covering_segment == False and segments_covered < len(starts):
        if framectr == starts[segments_covered]:
            segments_idx = 0
            covering_segment = True
            directory_name = 'all_sequences/%02d'%segments_covered
            print("WRITING TO ", directory_name)
            os.mkdir(directory_name)

    elif covering_segment == True:
        if framectr == stops[segments_covered]:
            covering_segment = False
            segments_covered += 1

    if covering_segment:
        shutil.copy2(center_filename, directory_name + '/color_center_%06d.jpg'%segments_idx)
        shutil.copy2(side_filename_left, directory_name + '/color_side_left_%06d.jpg'%segments_idx)
        shutil.copy2(side_filename_right, directory_name + '/color_side_right_%06d.jpg'%segments_idx)
        segments_idx += 1
