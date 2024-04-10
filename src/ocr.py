import cv2
import os
import random
import subprocess
import numpy as np


import pim
from utils import *
from kernels import *


def string(image):
    img = image.copy()
    bbxs = pim.find_connected_components_bboxes(image)
    for b in bbxs:
        y, x, y2, x2 = b
        pim.rectangle(img, (x, y), (x2, y2), 255, 1)
    pim.write_ppm_file("./output/ocr.ppm", img)
