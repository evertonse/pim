import sys

import pim # My (P)rocessing (IM)age functions
from utils import *
from kernels import *

# If any of these are true, the algorithm works for more cases
INCREASE_ACCURACY_BY_CLOSING = not False
INCREASE_ACCURACY_BY_BIGGER_KERNEL = not False

# Just a mode for developing
DEBUG = not False


@timer
def main(image_path):
    ppm_file = pim.read_ppm_file(image_path)
    pim.write_ppm_file(f"./output/ppm_file.ppm", ppm_file)
    orig = pim.convert_to_rgb(ppm_file)

    print(f"Image is {ppm_file.shape[1]} width and {ppm_file.shape[0]} height.")

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
    pim.write_ppm_file("./output/inverted.ppm", image)

    if INCREASE_ACCURACY_BY_BIGGER_KERNEL:
        # image = pim.dilate(image, horz_kernel_5x5_truncated, iterations=2)
        image = pim.dilate(image, horz_kernel_5x5, iterations=2)
    else:
        # Use a smaller kernel to be faster, but do one more iteration
        image = pim.dilate(image, horz_kernel_3x3, iterations=3)
    pim.write_ppm_file("./output/dilate.ppm", image)


    # kernel = create_text_kernel(5)
    # kernel = create_circular_kernel(3)
    kernel = horz_kernel_3x3

    if INCREASE_ACCURACY_BY_CLOSING:
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

    # Filter it to eliminate punctuation bbox, and keep only words
    words_bboxes = list(filter(lambda x: pim.bbox_area(x) > min_area, bboxes))

    print(f"{min_area=}")
    print(f"height={best_height}")
    for bbox in words_bboxes:
        min_y, min_x, max_y, max_x = bbox
        pim.rectangle(orig, (min_y, min_x), (max_y, max_x), (255, 0, 0), 2)

    blocks = pim.group_bboxes(words_bboxes, max_distance=best_height, image=orig)
    nblocks = len(blocks)
    for bbxs in blocks:
        y, x, y2, x2 = pim.enclosing_bbox(bbxs)
        pim.rectangle(orig, (y, x), (y2, x2), (0, 244, 55), 4)

    pim.write_ppm_file(f"./output/detected.ppm", orig)


    nwords = len(words_bboxes)
    nlines = pim.count_lines(words_bboxes, orig)
    ncolumns = pim.count_columns(words_bboxes, orig)

    pim.create_video_from_images(pim.vid_images, "./output/video.mp4")

    write_words = False
    if write_words:
        for b in words_bboxes:
            y, x, y2, x2 = b
            pim.write_ppm_file(f'./output/words/word_{y}x{x}.ppm', ppm_file[y:y2, x:x2])

    print(
        f"\n# {nwords} words, {nlines} lines, {ncolumns} columns, {nblocks} blocks."
    )


if __name__ == "__main__":
    image_path = ''
    if not DEBUG:
        if len(sys.argv) < 2:
            print('usage: python src/main.py [path/to/image.pbm]')
            exit(1)
        image_path = sys.argv[1]
    else:
        # image_path = "./assets/ocr/lorem_s12_c02_noise.pbm" # 583 words, 52 lines, 2 columns, 7 blocks.
        # (exibits wrong blocks because of heights) image_path = "./assets/ocr/lorem_s12_c03_just.pbm"  # 557 words, 52 lines, 3 columns, 8 blocks.
        image_path = "./assets/ocr/extra/arial_s14_c04_left.png"

        image_path = (
            "./assets/ocr/lorem_s12_c03.pbm"  # 557 words, 52 lines, 3 columns, 8 blocks.
        )
        image_path = "./assets/ocr/extra/cascadia_code_s16_c02_center.png"
        image_path = "./assets/ocr/extra/cascadia_code_s10_c02_right_bold.pbm"  # 395 words, 42 lines, 2 columns, 5 blocks.
        image_path = "./assets/ocr/extra/arial_s18_c04_left.pbm" 
        image_path = "./assets/ocr/extra/arial_s13_c02_left_space.pbm" #  132 words.
        image_path = "./assets/ocr/extra/cascadia_code_s10_c02_right_bold_noisy.pbm"  # 395 words, 42 lines, 2 columns, 5 blocks.
        image_path = "./assets/ocr/extra/lorem_s16_c02.pbm" # 318 words, 39 lines, 2 columns, 4 blocks.

    main(image_path)
