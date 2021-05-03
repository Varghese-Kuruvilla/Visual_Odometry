import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from PIL import Image
from VisualOdometry_Stereo import VisualOdometry
# from seg_utils import *
from park_utils import *

idx = 0

midpoints = [(168, 352), (175, 328), (183, 313), (204, 306), (187, 296), (203, 292), (229, 279), (235, 269), (235, 269)]

numpyseeds = [8214, 4123, 9451, 36215, 58926]
numpyseeds = [8214]


for seq_no in range(1, 9):
  t0 = time.time()
  print(seq_no + 5)
  for numpyseed in numpyseeds:
    np.random.seed(numpyseed)

    img_dir =     images_directory = "/home/olorin/Desktop/IISc/OdometryProject/ParkingSpot/data/feb27/data/homography/all_sequences/%02d"%(seq_no)
    strartno = 0
    left_image_files = sorted(glob.glob(img_dir + '/left*'))[strartno::4]
    dep_image_files = sorted(glob.glob(img_dir + '/depth*'))[strartno::4]
    cam_intr = np.asarray([[332.9648157406122, 0.0, 310.8472797033171], [0.0, 444.0950902369522, 252.76060777256825], [0.0, 0.0, 1.0]])
    # cam_intr = np.asarray([[664.3920764204493, 0.0, 620.5068279568037], [0.0, 664.5147822695388, 378.8579370468793], [0.0, 0.0, 1.0]])

    vo = VisualOdometry(cam_intr, seq_no)

    for index, (left_image_file, dep_image_file) in tqdm(enumerate(zip(left_image_files, dep_image_files))):
        # t_x = time.time()
        left_frame = cv2.imread(left_image_file)
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        left_frame = cv2.resize(left_frame, (640,480), interpolation = cv2.INTER_LANCZOS4)
        dep_img = np.load(dep_image_file) * 0.97
        dep_img = cv2.resize(dep_img, (640,480), interpolation = cv2.INTER_NEAREST)
        frame_pose = vo.process_frame(left_frame, dep_img, midpoints[seq_no], index)
        cv2.waitKey(1)
        # print(time.time() - t_x)
    print(frame_pose.t.T)
