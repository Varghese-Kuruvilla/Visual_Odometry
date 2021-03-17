'''
Simple code to view images from each sequence. The first image of each sequence is displayed in a window named Image0. 
Can find out the pixel location of the goal point from here.
Press 'e' if you have identified the midpoint and wish to move to the next sequence.
'''

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parent_folder = 'data/homography/all_sequences'
for seq in range(0,9):
    print(seq)
    seq_folder = parent_folder + '/%02d'%seq
    left_files = sorted(glob.glob(seq_folder + '/l*'))
    dep_files = sorted(glob.glob(seq_folder + '/d*'))
    for idx, (lf, df) in enumerate(zip(left_files, dep_files)):
        cv2.imshow('LEFT', cv2.resize(cv2.imread(lf), (640,480), interpolation = cv2.INTER_LANCZOS4))
        depimg = cv2.resize(np.load(df), (640,480), interpolation = cv2.INTER_NEAREST)
        cv2.imshow('DIMG', depimg/25.0)
        if idx == 0:
            cv2.imshow('IMAGE0', cv2.resize(cv2.imread(lf), (640,480), interpolation = cv2.INTER_LANCZOS4))

        ch = cv2.waitKey(0)
        if ch == ord('e'):
            break   
