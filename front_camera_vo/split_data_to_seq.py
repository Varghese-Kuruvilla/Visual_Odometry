'''
This script is a helper script to split timestamped color images, and corresponding depth images to different sequences. There are two ways you can use this:

1. No predefined start and end frame names, split dynamically. Set predefined_start_and_end to False. Iterates through every image in order:
    a. Press 'w' to change waitkey value to 0. i.e. moves to next frame only on keypress. If not, it shows images without any input.
    b. Press 'a' to switch back to waitkey = 1, i.e. show images one after another like a video.
    c. Press 'b' to begin writing a new sequence, from that particular image.
    d. Press 'e' to stop writing the sequence.
    e. Press 'p' to print Filename. (You can just keep printing p during the first run, and move to the second part of this readme)

2. If the start and end filenames for each sequence are known (through the rightimagesubset, or the previous step of printing filenames), 
    a. set predefined_start_and_end to True
    b. Fill up starts_fr and ends_fr
    c. Run
'''

import cv2
import numpy as np
import glob
import os
import shutil


predefined_start_and_end = False

seq_no = 0
start = 0
stop = 0
write_img = False
waitkeynum = 1

imgfolder = 'data/homography/homo_{seqtype}_leftimages/left_images'
depfolder = 'data/homography/homo_{seqtype}_depthimages'


if predefined_start_and_end:

    starts_fr = ['_1615061068097609700.png', '_1615061118633950800.png', '_1615061162702689600.png', '_1615061068097609700.png', '_1615061118633950800.png', '_1615061162702689600.png', '_1615061068097609700.png', '_1615061118633950800.png', '_1615061162702689600.png']
    ends_fr = ['_1615061083231593100.png', '_1615061132567640300.png', '_1615061175370102500.png', '_1615061083231593100.png', '_1615061132567640300.png', '_1615061175370102500.png', '_1615061083231593100.png', '_1615061132567640300.png', '_1615061175370102500.png']

for seq in ['short', 'medium', 'long']:

    left_filenames = sorted(glob.glob(imgfolder.format(seqtype = seq) + '/*'))
    dep_filenames = sorted(glob.glob(depfolder.format(seqtype = seq) + '/*'))

    # print(left_filenames)

    for dep_filename,  left_filename in zip(dep_filenames, left_filenames):
        if not predefined_start_and_end:
            dep_img = np.load(dep_filename)
            cv2.imshow('DEP IMG', dep_img)
            left_img = cv2.imread(left_filename)
            cv2.imshow('LEFT IMG', left_img)
        ch = cv2.waitKey(waitkeynum)

        if predefined_start_and_end:
            if os.path.basename(left_filename) == starts_fr[seq_no]:
                ch = ord('b')
            if os.path.basename(left_filename) == ends_fr[seq_no]:
                ch = ord('e')

        if ch == ord('b'):
            print("STARTED WRITING")
            write_img = True
            img_counter = 0
            write_folder = '/home/olorin/Desktop/IISc/OdometryProject/ParkingSpot/data/mar8/data/homography/all_sequences/%02d'%seq_no
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

        if ch == ord('a'):
            waitkeynum = 1

        if ch == ord('p'):
            print(left_filename)
        
        if write_img:
            left_filename_to_write = write_folder + '/left_%06d.png'%img_counter
            depth_filename_to_write = write_folder + '/depth_%06d.npy'%img_counter
            shutil.copy2(left_filename, left_filename_to_write)  
            shutil.copy2(dep_filename, depth_filename_to_write)  
            img_counter += 1



