import cv2
import os
import random
import subprocess
import numpy as np


import pim
from utils import *
from kernels import *


def string(img):
    img = pim.invert(img)
    bbxs = pim.find_connected_components_bboxes(img, connectivity=8)
    print(bbxs)

    for b in bbxs:
        y, x, y2, x2 = b
        # img = pim.rectangle(img, (y, x), (y2, x2), 255, 1)
    pim.write_ppm_file("./output/test.ppm", img)


img = pim.read_ppm_file("./output/ocr.ppm")
# string(img)
a = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0],
])
