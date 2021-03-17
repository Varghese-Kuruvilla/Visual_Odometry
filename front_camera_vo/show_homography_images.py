'''
Simple matplotlib code to display the image for selecting points for homography.
Using matplotlib, you can zoom into the image, and note down the exact pixel for homography calculations.
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib
import glob

fil = 'homography_file.png'
print(fil)
img = mpimg.imread(fil)
img = img[:, :1280]
print(img.shape)
matplotlib.pyplot.imsave("img.png", img)
img = cv2.resize(img, (640,480), interpolation = cv2.INTER_LANCZOS4)
cv2.imwrite('sample_image.jpeg', img)
imgplot = plt.imshow(img)
plt.show()