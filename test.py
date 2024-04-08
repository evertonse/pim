import cv2
import os
import random
import numpy as np
from kernels import *
from ocr import distance

b1 = (0, 0, 0, 0)
b2 = (0, 0, 0, 0)

d = distance(b1,b2)
print(f'{d=}')
