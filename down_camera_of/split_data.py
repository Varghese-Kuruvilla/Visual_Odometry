'''
Splitting data into sequences for optical flow data. Please copy all images and paste it in one folder called optical_flow_all_images.
Create a folder called all_sequences
'''

import cv2
import numpy as np
import glob
import os
import shutil

seq_no = 0
start = 0
stop = 0
write_img = False
waitkeynum = 1

folder = 'optical_flow_all_images'
left_filenames = sorted(glob.glob(folder + '/*'))

for left_filename in left_filenames:
    left_img = cv2.imread(left_filename)
    cv2.imshow('LEFT IMG', left_img)
    ch = cv2.waitKey(waitkeynum)

    if ch == ord('b'):
        print("STARTED WRITING")
        write_img = True
        img_counter = 0
        write_folder = 'all_sequences%02d'%seq_no
        if os.path.exists(write_folder):
            shutil.rmtree(write_folder)
        os.mkdir(write_folder)
        waitkeynum = 1
    if ch == ord('e'):
        write_img = False
        seq_no += 1
        waitkeynum = 1

    if ch == ord('w'):
        waitkeynum = 0
    
    if write_img:
        left_filename_to_write = write_folder + '/left_%06d.png'%img_counter
        shutil.copy2(left_filename, left_filename_to_write)  
        img_counter += 1


