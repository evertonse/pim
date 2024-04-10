import os
import random
import subprocess

import pim # My (P)rocessing (IM)age functions
from utils import *
from kernels import *

ECONOMY_MODE = True # If turn off, the algorithm works for more cases

def pbm(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    ppm_file = pim.read_ppm_file(image_path)
    pim.write_ppm_file(f"./output/ppm_file.ppm", ppm_file)
    orig = pim.convert_to_rgb(ppm_file)
    print(f"{orig.shape=}")
    print(f"{ppm_file.shape=}")

    def do_the_noise_invert_thing():  # BEWARE to not mess with ready to work files
        pim.write_ppm_file(f'{image_path[:image_path.rfind(".")]}.pbm', pim.pbm(orig))
        pim.write_ppm_file(
            f'{image_path[:image_path.rfind(".")]}_noisy.pbm',
            pim.noisify(pim.invert(pim.pbm(orig))),
        )


    image = ppm_file

    # Small resolution, ain't worth it
    if image.shape[0] * image.shape[1] > 795 * 795:
        image = pim.median_blur(image, 3)
        pim.write_ppm_file("./output/median_blur.ppm", image)


    image = pim.invert(image)
    pim.write_ppm_file("./output/threshold.ppm", image)

    image = pim.dilate(image, horz_kernel_5x5_truncated, iterations=2)
    pim.write_ppm_file("./output/dilate.ppm", image)

    # kernel = create_text_kernel(5)
    # kernel = create_circular_kernel(3)
    kernel = horz_kernel_3x3
    print(f"{kernel=}")

    if not ECONOMY_MODE:
        image = pim.closing(image, kernel)
        image = pim.opening(image, kernel)
        pim.write_ppm_file("./output/open.ppm", image)
        image = pim.closing(image, kernel)
        pim.write_ppm_file(f"./output/close.ppm", image)

    # Optional dilate with block kernel, sometimes it's good.
    # But idk how to specify programatically when it's good, hence iterations being zero.
    image = pim.dilate(
        image,
        block_kernel,
        iterations=0,
    )


    pim.write_ppm_file("./output/processed.ppm", image)

    bboxes = pim.find_connected_components_bboxes(image)

    best_height = pim.choose_best_height(bboxes)
    min_area = (best_height * best_height) / (2.5)
    print(f"{min_area, best_height=}")

    # Filter it to eliminate punctuation bbox, and keep only words
    words_bboxes = list(filter(lambda x: pim.bbox_area(x) > min_area, bboxes))
    for bbox in words_bboxes:
        min_y, min_x, max_y, max_x = bbox
        pim.rectangle(orig, (min_y, min_x), (max_y, max_x), (255, 0, 0), 1)

    list_of_bboxes = pim.group_bboxes(bboxes, max_distance=best_height, image=orig)
    for bbxs in list_of_bboxes:
        y, x, y2, x2 = pim.enclosing_bbox(bbxs)
        pim.rectangle(orig, (y, x), (y2, x2), (0, 244, 55), 4)

    pim.write_ppm_file(f"./output/detected.ppm", orig)

    words = len(words_bboxes)
    lines = pim.count_lines(bboxes, orig)
    columns = pim.count_columns(bboxes, orig)

    pim.create_video_from_images(pim.vid_images, "./output/video.mp4")

    import ocr
    if words_bboxes:
        y, x, y2, x2 = words_bboxes[0]
        s = ocr.string(ppm_file[y:y2, x:x2])

    print(
        f"\n# {words} words, {lines} lines, {columns} columns, {len(list_of_bboxes)} blocks."
    )


if __name__ == "__main__":
    main()
