import cv2
import pytesseract
import numpy as np

# orig = cv2.imread("./assets/ocr/lorem_s12_c02_noise.pbm") # 583 words, 52 lines, 2 columns, 7 blocks.
# orig = cv2.imread("./assets/ocr/extra/lorem_s16_c02.png") # 318 words, 39 lines, 2 columns, 4 blocks.
# orig = cv2.imread('./assets/ocr/lorem_s12_c03.pbm') # 557 words, 52 lines, 3 columns, 8 blocks.
#(exibits wrong blocks because of heights) orig = cv2.imread("./assets/ocr/lorem_s12_c03_just.pbm")  # 557 words, 52 lines, 3 columns, 8 blocks.
# Needs fixing orig = cv2.imread("./assets/ocr/extra/cascadia_code_s16_c02_center.png")
orig = cv2.imread("./assets/ocr/extra/cascadia_code_s10_c02_right_bold.png") # 391 words, 42 lines, 2 columns, 5 blocks.
# orig = cv2.imread("./assets/ocr/extra/arial_s14_c04_left.png")
print(f"{orig.shape=}")


def im_open(image, kernel, iterations=1):
    for _ in range(iterations):
        image = cv2.erode(image, kernel,  iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)
    return image


def im_close(image, kernel, iterations=1):
    for _ in range(iterations):
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel,  iterations=1)
    return image


# Convert the image to grayscale
image = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"./output/orig.png", image)

# Small resolution, ain't worth it
if image.shape[0] * image.shape[1] > 1124 * 795:
    image = cv2.medianBlur(image, 3)

cv2.imwrite(f"./output/noise_free.png", image)


kernel_block_3x3 = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)


_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite(f"./output/threshold.png", image)



def hit_or_miss(binary_image, structuring_element=None):
    # Define the default structuring element if not provided
    if structuring_element is None:
        structuring_element = np.array(
            [
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
        # structuring_element = np.array(
        #     [
        #         [0, 0, 0, 1, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 1, 1, 1, 0, 0],
        #     ],
        #     dtype=np.uint8,
        # )

    # Invert the binary image
    inverted_image = cv2.bitwise_not(binary_image)
    cv2.imwrite("./output/inverted_image.png", inverted_image)

    # Perform erosion with the structuring element
    erosion = cv2.erode(inverted_image, structuring_element, iterations=1)
    cv2.imwrite("./output/inverted_eroded.png", erosion)

    # Invert the result to get the hit-or-miss operation
    result = cv2.bitwise_not(erosion)
    cv2.imwrite("./output/result.png", result)

    return result


# Define a kernel for morphological operations
def create_text_kernel(size):
    # Create a kernel with vertical and horizontal lines
    kernel = np.zeros((size, size), dtype=np.uint8)

    # Add vertical line in the center
    kernel[:, size // 2] = 1

    # Add horizontal line in the center
    kernel[size // 2, :] = 1
    # kernel = np.ones((7, 7), np.uint8)
    return kernel


def create_circular_kernel(radius):
    size = radius
    kernel = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if x * x + y * y < radius * radius:
                kernel[x][y] = 1

    return kernel


horz_kernel_3x3 = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

horz_kernel_5x5 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

horz_kernel_5x5_truncated = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

horz_kernel_3x3 = np.array(
    [
        [0,  0,  0],
        [1,  1,  1],
        [0,  0,  0],
    ],
    dtype=np.uint8,
)

horz_kernel_7x7 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)


horz_kernel_9x9 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

vert_kernel_5x5 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

vert_kernel_7x7 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height
    return area


# Morphological operations



image = cv2.dilate(image, horz_kernel_5x5_truncated, iterations=2)
cv2.imwrite(f"./output/dilate.png", image)

# image = cv2.erode(image, horz_kernel_3x3, iterations=1)
# cv2.imwrite(f"./output/erode.png", image)

kernel = create_text_kernel(5)
kernel = create_circular_kernel(3)
kernel = horz_kernel_3x3
print(f"{kernel=}")

image = im_close(image, kernel)
image = im_open(image, kernel)
cv2.imwrite("./output/open.png", image)
image = im_close(image, kernel)
cv2.imwrite(f"./output/close.png", image)




block_kernel = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype=np.uint8,
)

image = cv2.dilate(
    image,
    block_kernel,
    iterations=0,
)



# image = hit_or_miss(image)
# cv2.imwrite("./output/hit_or_miss.png", image)

finished_image = image

cv2.imwrite("./output/finished.png", finished_image)


def find_median_height(bboxes):
    # List to store heights of bounding boxes
    heights = []

    # Iterate over all bounding boxes
    for bbox in bboxes:
        # Extract ymin and ymax from each bounding box
        min_x, min_y, max_x, max_y = bbox
        # Calculate height and append to the list
        height = abs(max_x - min_x)
        heights.append(height)

    # Sort the list of heights
    heights.sort()

    # Calculate the median height
    if len(heights) % 2 == 0:
        # If number of heights is even, take average of middle two heights
        median_height = (
            heights[len(heights) // 2 - 1] + heights[len(heights) // 2]
        ) / 2
    else:
        # If number of heights is odd, take the middle height
        median_height = heights[len(heights) // 2]

    return median_height


def find_connected_components_bbox(image, min_area=0, connectivity=4):
    def dfs(x, y):
        min_x, min_y, max_x, max_y = x, y, x, y
        stack = [(x, y)]
        while stack != []:
            cx, cy = stack.pop()
            if (
                0 <= cx < image.shape[0]
                and 0 <= cy < image.shape[1]
                and not visited[cx, cy]
                and image[cx, cy] == 255
            ):
                visited[cx, cy] = True
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
                nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if connectivity == 8:
                    nbrs.append(( 1, 1))
                    nbrs.append(( 1,-1))
                    nbrs.append((-1, 1))
                    nbrs.append((-1,-1))
                for dx, dy in nbrs:
                    stack.append((cx + dx, cy + dy))
        return min_x, min_y, max_x, max_y

    visited = np.zeros_like(image, dtype=bool)
    bounding_boxes = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not visited[i, j] and image[i, j] == 255:
                min_x, min_y, max_x, max_y = dfs(i, j)
                bounding_boxes.append((min_x, min_y, max_x, max_y))

    return bounding_boxes


bboxes = find_connected_components_bbox(finished_image)
median_height = find_median_height(bboxes)
min_area = (median_height * median_height) / (2.5)
bboxes = list(filter(lambda x: bbox_area(x) > min_area, bboxes))
print(f"{min_area, median_height=}")
words = 0
for bbox in bboxes:
    min_x, min_y, max_x, max_y = bbox
    area = bbox_area(bbox)
    words += 1
    cv2.rectangle(orig, (min_y, min_x), (max_y, max_x), (0, 0, 255), 1)


def count_lines(bboxes):
    global orig
    if len(bboxes) == 0:
        return 0

    bboxes = sorted(bboxes, key=lambda bbox: (bbox[0], bbox[2]))
    video_writer = cv2.VideoWriter(
        "./output/video.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        30,
        (orig.shape[1], orig.shape[0]),
    )

    lines = 1
    prev1, _, prev2, _ = bboxes[0]
    # Iterate through the sorted bounding boxes starting from the second one
    xdiffs_bboxes = []
    for bbox in bboxes[1:]:
        y1, _, y2, _ = bbox
        overlap = max(0, min(prev2, y2) - max(prev1, y1))
        # print(f"{overlap=}")
        if overlap > abs(prev1 - prev2) / 100:
            continue
        lines += 1
        prev1 = y1
        prev2 = y2

        vis_img = orig.copy()
        cv2.rectangle(vis_img, (0, y1), (orig.shape[0] - 1, y2), (255, 0, 0), 2)
        video_writer.write(vis_img)

    video_writer.release()
    return lines


def count_columns(bboxes):
    # Sort the bounding boxes based on their left x-coordinate
    sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[1])

    # Initialize variables
    num_columns = 0
    max_right = float("-inf")

    # Iterate through sorted bounding boxes
    for bbox in sorted_bboxes:
        _, left, _, right = bbox

        # If the left x-coordinate of the bounding box is greater than the current rightmost x-coordinate,
        # it indicates the start of a new column
        if left > max_right:
            num_columns += 1

        # Update the current rightmost x-coordinate
        max_right = max(max_right, right)

    return num_columns


def distance(bbox1, bbox2):
    import math


def distance(bbox1, bbox2):
    """
    Calculate the distance between the closest edges of two bounding boxes.
    """
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    # Initialize the minimum distance with a large value
    min_distance = float("inf")

    for x1, y1 in [(left1, top1), (right1, top1), (right1, bottom1), (left1, bottom1)]:
        for x2, y2 in [
            (left2, top2),
            (right2, top2),
            (right2, bottom2),
            (left2, bottom2),
        ]:
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < min_distance:
                min_distance = dist

    return min_distance


def _bbox_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    """
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    # overlap = max(0, min(prev2, y2) - max(prev1, y1))

    return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)


def bbox_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    """
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    # Calculate the overlap area
    overlap_left = max(left1, left2)
    overlap_top = max(top1, top2)
    overlap_right = min(right1, right2)
    overlap_bottom = min(bottom1, bottom2)

    # If the overlap area is valid (non-negative), the bounding boxes overlap
    if overlap_right > overlap_left and overlap_bottom > overlap_top:
        return True
    else:
        return False


def group_bboxes(bboxes, max_distance) -> list[list]:
    result = []
    rest_idxs = [i for i in range(len(bboxes))]
    while len(rest_idxs) > 0:
        close_idxs = [rest_idxs.pop()]
        bbox = bboxes[close_idxs[0]]
        counter = 0
        while counter < len(rest_idxs):
            r_idx = rest_idxs[counter]
            counter += 1
            dist = distance(bbox, bboxes[r_idx])
            if dist <= max_distance or bbox_overlap(bbox, bboxes[r_idx]):
                close_idxs.append(r_idx)
                rest_idxs.remove(r_idx)
                bbox = enclosing_bbox([bbox, bboxes[r_idx]])
                counter = 0
        result.append([bboxes[i] for i in close_idxs])
    return result


def distance_bboxes_matrix(bboxes):
    """
    Compute a matrix of distances between a given bounding box and every other bounding box.
    """
    num_bboxes = len(bboxes)
    distances = [[-1 for _ in range(num_bboxes)] for _ in range(num_bboxes)]

    # Compute distances between the given bbox and every other bbox
    for i in range(num_bboxes):
        for j in range(num_bboxes):
            distances[i][j] = distance(bboxes[i], bboxes[j])
    return distances


def enclosing_bbox(bboxes):
    if not bboxes:
        return None

    # Initialize min and max coordinates with the first bounding box
    x_min, y_min, x_max, y_max = bboxes[0]

    # Iterate through the rest of the bounding boxes to update min and max coordinates
    for bbox in bboxes[1:]:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

    # Return the enclosing bounding box
    return (x_min, y_min, x_max, y_max)


list_of_bboxes = group_bboxes(bboxes, max_distance=max(median_height * 1.75, 10))
for bbxs in list_of_bboxes:
    x, y, x2, y2 = enclosing_bbox(bbxs)
    cv2.rectangle(orig, (y, x), (y2, x2), (0, 244, 55), 4)


print(
    f"# {words} words, {count_lines(bboxes)} lines, {count_columns(bboxes)} columns, {len(list_of_bboxes)} blocks."
)
cv2.imwrite("./output/detected.png", orig)
