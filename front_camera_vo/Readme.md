## FrontFacingVO

Contains utils for front facing VO taken from zed. 
1. To create sequences:
    a. Run depth_utils.py to split the depth npy files to individual depth images and write in folder.
    b. Run split_data_to_seq.py to split the images to sequences.

2. To find out the parking spot pixel location of each sequence in first frame, run show_first_frame_midpoint.py

3. To find the pixel locations for calculation of homography matrix, run show_homography_images.py

4. Run road_segment_manual.py to manually segment road images.
