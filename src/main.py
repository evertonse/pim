import os
import random
import subprocess
import numpy as np


import ocr 
from utils import *
from kernels import *

ECONOMY_MODE = True # If turn off, the algorithm works for more cases
GO_FAST = True
if GO_FAST:
    import cv2
    median_blur = cv2.medianBlur
    dilate = cv2.dilate
    erode = cv2.erode

def pbm(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image

def invert(image):
    image = cv2.bitwise_not(image)
    return image

def main():
    # image_path = "./assets/ocr/lorem_s12_c02_noise.pbm" # 583 words, 52 lines, 2 columns, 7 blocks.
    # image_path = "./assets/ocr/extra/lorem_s16_c02.png" # 318 words, 39 lines, 2 columns, 4 blocks.
    # (exibits wrong blocks because of heights) image_path = "./assets/ocr/lorem_s12_c03_just.pbm"  # 557 words, 52 lines, 3 columns, 8 blocks.
    image_path = "./assets/ocr/extra/arial_s14_c04_left.png"

    image_path = (
        "./assets/ocr/lorem_s12_c03.pbm"  # 557 words, 52 lines, 3 columns, 8 blocks.
    )
    image_path = "./assets/ocr/extra/cascadia_code_s16_c02_center.png"
    image_path = "./assets/ocr/extra/cascadia_code_s10_c02_right_bold.pbm"  # 395 words, 42 lines, 2 columns, 5 blocks.
    image_path = "./assets/ocr/extra/cascadia_code_s10_c02_right_bold_noisy.pbm"  # 395 words, 42 lines, 2 columns, 5 blocks.

    ppm_file = ocr.read_ppm_file(image_path)
    ocr.write_ppm_file(f"./output/ppm_file.ppm", ppm_file)
    orig = ocr.convert_to_rgb(ppm_file)
    print(f"{orig.shape=}")
    print(f"{ppm_file.shape=}")

    def do_the_noise_invert_thing():  # BEWARE to not mess with ready to work files
        ocr.write_ppm_file(f'{image_path[:image_path.rfind(".")]}.pbm', ocr.pbm(orig))
        ocr.write_ppm_file(
            f'{image_path[:image_path.rfind(".")]}_noisy.pbm',
            ocr.noisify(ocr.invert(ocr.pbm(orig))),
        )


    image = ppm_file
    # Small resolution, ain't worth it
    if image.shape[0] * image.shape[1] > 795 * 795:
        image = median_blur(image, 3)

    ocr.write_ppm_file("./output/.threshold.ppm", image)

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ocr.write_ppm_file("./output/threshold.ppm", image)

    image = ocr.dilate(image, horz_kernel_5x5_truncated, iterations=2)
    ocr.write_ppm_file("./output/dilate.ppm", image)

    # kernel = create_text_kernel(5)
    # kernel = create_circular_kernel(3)
    kernel = horz_kernel_3x3
    print(f"{kernel=}")

    if not ECONOMY_MODE:
        image = ocr.closing(image, kernel)
        image = ocr.opening(image, kernel)
        ocr.write_ppm_file("./output/open.ppm", image)
        image = ocr.closing(image, kernel)
        ocr.write_ppm_file(f"./output/close.ppm", image)

    image = dilate(
        image,
        block_kernel,
        iterations=0,
    )


    ocr.write_ppm_file("./output/processed.ppm", image)

    bboxes = ocr.find_connected_components_bbox(image)

    best_height = ocr.choose_best_height(bboxes)
    min_area = (best_height * best_height) / (2.5)
    print(f"{min_area, best_height=}")

    # Filter it to eliminate punctuation bbox
    bboxes = list(filter(lambda x: ocr.bbox_area(x) > min_area, bboxes))

    words = 0
    for bbox in bboxes:
        min_x, min_y, max_x, max_y = bbox
        area = ocr.bbox_area(bbox)
        words += 1
        ocr.rectangle(orig, (min_y, min_x), (max_y, max_x), (255, 0, 0), 1)

    list_of_bboxes = ocr.group_bboxes(bboxes, max_distance=best_height, image=orig)
    for bbxs in list_of_bboxes:
        x, y, x2, y2 = ocr.enclosing_bbox(bbxs)
        ocr.rectangle(orig, (y, x), (y2, x2), (0, 244, 55), 4)

    ocr.write_ppm_file(f"./output/detected.ppm", orig)

    lines = ocr.count_lines(bboxes, orig)
    columns = ocr.count_columns(bboxes)

    ocr.create_video_from_images(ocr.vid_images, "./output/video.mp4")

    print(
        f"\n# {words} words, {lines} lines, {columns} columns, {len(list_of_bboxes)} blocks."
    )


if __name__ == "__main__":
    main()
