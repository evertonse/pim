import sys

import pim  # My (P)rocessing (IM)age functions
from utils import *
from kernels import *

# If any of these are true, the algorithm works for more cases
INCREASE_ACCURACY_BY_CLOSING = not False
INCREASE_ACCURACY_BY_BIGGER_KERNEL = not False
INCREASE_ACCURACY_BY_OPENING_FOLLOWED_BY_CLOSING = False

# "Grupo 13"
# Aluon: ÉVERTON SANTOS DE ANDRADE JUNIOR
# Curso: CIÊNCIA DA COMPUTAÇÃO
# Matrícula: 202100011379
# Usuário: evertonse
# E-mail:  evertonse.junior@gmail.com
GROUP_NUMBER = 13


# Write ppm files for each word the algortihm finds
WRITE_PPM_FILES_FOR_WORDS = False

# Just a mode for developing
DEBUG = not False


def main():
    possible_image_paths = [
        "./assets/extra/arial_s14_c04_left.png",
        "./assets/lorem_s12_c03.pbm",  # 557 words, 52 lines, 3 columns, 8 blocks.
        "./assets/extra/cascadia_code_s16_c02_center.png",
        "./assets/extra/cascadia_code_s10_c02_right_bold.pbm",  # 395 words, 42 lines, 2 columns, 5 blocks.
        "./assets/extra/arial_s18_c04_left.pbm",
        "./assets/extra/arial_s13_c02_left_space.pbm",  #  132 words.
        "./assets/extra/cascadia_code_s10_c02_right_bold_noisy.pbm",  # 395 words, 42 lines, 2 columns, 5 blocks.
        "./assets/extra/grupo_13_arial_colunas_2_blocos_4_linhas_39_palavras_318.pbm"  # 318 words, 39 lines, 2 columns, 4 blocks.
        # "./assets/lorem_s12_c02_noise.pbm" # 583 words, 52 lines, 2 columns, 7 blocks.
        # (exibits wrong blocks because of heights) "./assets/ocr/lorem_s12_c03_just.pbm"  # 557 words, 52 lines, 3 columns, 8 blocks.
    ]

    # Choose last possible image paths as the default image path
    image_path = possible_image_paths[-1]

    if len(sys.argv) < 2:
        print("USAGE: python3 src/main.py [path/to/image.pbm]\n")
        print(f"INFO: using default image path `{image_path}`")
    else:
        image_path = sys.argv[1]

    process(image_path)


# The whole project is in here
def process(image_path):
    ppm_file = pim.read_ppm_file(image_path)
    orig = pim.convert_to_rgb(ppm_file)

    print(f"INFO: Image is {ppm_file.shape[1]}x{ppm_file.shape[0]} (width x height).")

    image = ppm_file

    # Small resolution, ain't worth it
    if image.shape[0] * image.shape[1] > 795 * 795:
        image = pim.median_blur(image, 3)
        pim.write_ppm_file("./output/median_blur.ppm", image)

    image = pim.invert(image)
    pim.write_ppm_file("./output/inverted.ppm", image)

    # Add the current image to the video list of frames
    pim.video_frames.extend([pim.convert_to_rgb(image)]*3)

    # Both of the branches below are using horizontal kernel
    # To try to connect letter that are closed to gether
    # It's assume letter are written left to write as opposed to up and down
    # Hence using a horizontal kernel
    if INCREASE_ACCURACY_BY_BIGGER_KERNEL:
        image = pim.dilate(image, horz_kernel_5x5_truncated, iterations=2)
    else:
        # Use a smaller kernel to be faster, but do one more iteration for precision
        image = pim.dilate(image, horz_kernel_3x3, iterations=3)

    # Save the dilated image as a intermediate file
    pim.write_ppm_file("./output/dilate.ppm", image)
    pim.video_frames.extend([pim.convert_to_rgb(image)]*3)


    # Below are possible choices considered for kernel when doing a open and closing operations
    # turns out the algorithm works well with horizonatal kernel only, as opposed to a "crufix" kernel
    # or even a circular one.
    # Anothe thing is that, it's not necessary to do the closing and opening sequence
    # for the algotihm to work well. Therefore, it's optional

    # kernel = create_text_kernel(5)
    # kernel = create_circular_kernel(3)
    kernel = horz_kernel_3x3

    if INCREASE_ACCURACY_BY_CLOSING:
        image = pim.closing(image, kernel)
        pim.write_ppm_file(f"./output/close1.ppm", image)

    if INCREASE_ACCURACY_BY_OPENING_FOLLOWED_BY_CLOSING:
        image = pim.opening(image, kernel)
        pim.write_ppm_file("./output/open.ppm", image)

        image = pim.closing(image, kernel)
        pim.write_ppm_file(f"./output/close2.ppm", image)

    # Optional dilate with block kernel, sometimes it's good.
    # But idk how to specify programatically when it's good, hence iterations being zero.
    image = pim.dilate(
        image,
        block_kernel,
        iterations=0,
    )

    # Save final snap shot for the preprocessed image and add to the video frame list
    pim.write_ppm_file("./output/preprocessed.ppm", image)
    pim.video_frames.extend([pim.convert_to_rgb(image)]*3)

    bboxes = pim.find_connected_components_bboxes(image)
    best_height = pim.choose_best_height(bboxes)
    min_area = (best_height * best_height) / (2.5)

    # Filter it to eliminate punctuation bbox, and keep only words
    words_bboxes = list(filter(lambda x: pim.bbox_area(x) > min_area, bboxes))

    for bbox in words_bboxes:
        min_y, min_x, max_y, max_x = bbox
        pim.rectangle(orig, (min_y, min_x), (max_y, max_x), (255, 0, 0), 2)

    blocks = pim.group_bboxes(words_bboxes, max_distance=best_height, image=orig)
    nblocks = len(blocks)
    for bbxs in blocks:
        y, x, y2, x2 = pim.enclosing_bbox(bbxs)
        pim.rectangle(orig, (y, x), (y2, x2), (0, 244, 55), 4)

    # Save the nummber of words, lines and columns
    nwords = len(words_bboxes)
    nlines = pim.count_lines(words_bboxes, orig)
    ncolumns = pim.count_columns(words_bboxes, orig)

    # Create the video using ffmpeg for a interactive view of the algorithms implemented
    pim.create_video_from_images(pim.video_frames, f"./group_{GROUP_NUMBER}_video.mp4")

    # Write each work in a new image separately, to use in the OCR attempt `ocr.py`
    if WRITE_PPM_FILES_FOR_WORDS:
        for b in words_bboxes:
            y, x, y2, x2 = b
            pim.write_ppm_file(f"./output/words/word_{y}x{x}.ppm", ppm_file[y:y2, x:x2])

    # Print the words, lines columns and blocks found to the user
    print(f"\n> {nwords} words, {nlines} lines, {ncolumns} columns, {nblocks} blocks.")

    # Write the final image with the bouding boxes for the words and blocks
    pim.write_ppm_file(
        f"./group_{GROUP_NUMBER}_detected_colunas_{ncolumns}_blocos_{nblocks}_linhas_{nlines}_palavras_{nwords}.ppm",
        orig,
    )


if __name__ == "__main__":
    main()
