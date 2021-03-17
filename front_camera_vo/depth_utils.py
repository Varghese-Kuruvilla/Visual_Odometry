'''
This script reads the depth part images, first renames them to 6 precision (numbered from 000000 to 000010). Then it opens every depth part file and writes the image.
'''

import glob
import numpy as np
import shutil
import os


depth_filenames = sorted(glob.glob('data/homography/homo_%s_depthdata/depth_data/*.npy'%seq))
new_foldername = 'data/homography/homo_%s_depthimages/'%seq

for seq in ['short', 'medium', 'long']:
    depth_filename_counter = 0
    for full_path in depth_filenames:
        filename = os.path.splitext(os.path.basename(full_path))[0]
        num = int(float(filename.split('_')[1]) * 1)
        print(num)
        new_filename = os.path.join(os.path.dirname(full_path), 'depth_part_%06d.npy'%num)
        shutil.copy2(full_path, new_filename)
        os.remove(full_path)

for seq in ['short', 'medium', 'long']:
    depth_filename_counter = 0
    for depth_filename in depth_filenames:
        print(depth_filename)
        dep_imgs = np.load(depth_filename)
        for dep_img in dep_imgs:
            np.save(os.path.join(new_foldername, 'depth_%06d.npy'%depth_filename_counter), dep_img)
            depth_filename_counter += 1
