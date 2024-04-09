import os
import random
from kernels import *
import numpy as np
from utils import *


def noisify(image):
    noisy_image = image.copy()
    for i in range(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            # Generate a random value between -1 and 1
            noise = random.uniform(0, 1)
            if noise > 0.989:
                noisy_image[i, j] = 0
    return noisy_image


def rectangle(image, pt1, pt2, color, thickness=1):
    assert image.shape[-1] == len(color)
    x1, y1 = pt1
    x2, y2 = pt2

    image[y1 : y1 + thickness, x1 : x2 + 1] = color
    image[y2 : y2 + thickness, x1 : x2 + 1] = color

    image[y1 : y2 + thickness, x1 : x1 + thickness] = color
    image[y1 : y2 + thickness, x2 : x2 + thickness] = color

    return image


@timer
def opening(image, kernel, iterations=1):
    for _ in range(iterations):
        image = erode(image, kernel, iterations=1)
        image = dilate(image, kernel, iterations=1)
    return image


@timer
def closing(image, kernel, iterations=1):
    for _ in range(iterations):
        image = dilate(image, kernel, iterations=1)
        image = erode(image, kernel, iterations=1)
    return image


@timer
def median_blur(image, filter_size=3):
    assert len(image.shape) == 2

    convolved = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            colors = list()
            for j in range(filter_size):
                for i in range(filter_size):
                    i_offset = i - filter_size // 2
                    j_offset = j - filter_size // 2

                    color = 0
                    if (
                        (x + i_offset) >= 0
                        and (x + i_offset) < width
                        and y + j_offset >= 0
                        and y + j_offset < height
                    ):
                        color = image[y + j_offset, x + i_offset]
                    colors.append(color)
            colors.sort()
            convolved[y, x] = colors[(len(colors) // 2)]
    return convolved


@timer
def erode(image, kernel, iterations=1):
    assert kernel.shape[0] % 2 != 0, f"{kernel.shape=}"

    result = image.copy()

    height = image.shape[0]
    width = image.shape[1]

    kernel_height = kernel.shape[1]
    kernel_width = kernel.shape[0]
    for y in range(height):
        for x in range(width):
            all_good = True
            for j in range(kernel_height):
                for i in range(kernel_width):
                    i_offset = i - kernel_width // 2
                    j_offset = j - kernel_height // 2
                    color = 0
                    if (
                        (x + i_offset) >= 0
                        and (x + i_offset) < width
                        and y + j_offset >= 0
                        and y + j_offset < height
                    ):
                        color = image[y + j_offset, x + i_offset]
                    kernel_color = kernel[j, i]
                    if kernel_color > 0:
                        if color == 0:
                            all_good = False
            if all_good:
                result[y, x] = 255
            else:
                result[y, x] = 0
    return result

@timer
def dilate(image, kernel, iterations=1):
    def internal_dilate(image, kernel):
        assert kernel.shape[0] % 2 != 0, f"{kernel.shape=}"
        result = np.zeros_like(image)
        height = image.shape[0]
        width = image.shape[1]
        kernel_height = kernel.shape[1]
        kernel_width = kernel.shape[0]
        kernel_width_delta = kernel_width // 2
        kernel_height_delta = kernel_height // 2
        for y in range(height):
            for x in range(width):
                all_good = False
                for j in range(kernel_height):
                    for i in range(kernel_width):
                        i_offset = i - kernel_width_delta
                        j_offset = j - kernel_height_delta
                        color = 0
                        if (
                            (x + i_offset) >= 0
                            and (x + i_offset) < width
                            and y + j_offset >= 0
                            and y + j_offset < height
                        ):
                            color = image[y + j_offset, x + i_offset]
                        kernel_color = kernel[j, i]
                        if kernel_color > 0 and color > 0:
                            all_good = True
                if all_good:
                    result[y, x] = 255
                else:
                    result[y, x] = 0
        return result

    for _ in range(iterations):
        image = internal_dilate(image, kernel)
    return image


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height
    return area


@timer
def choose_best_height(bboxes):
    # List to store heights of bounding boxes
    heights = list()

    # Iterate over all bounding boxes
    for bbox in bboxes:
        # Extract ymin and ymax from each bounding box
        min_x, min_y, max_x, max_y = bbox
        # Calculate height and append to the list
        height = abs(max_x - min_x)
        heights.append(height)

    # Calculate the median height
    heights.sort()
    median_height = heights[len(heights) // 2]
    print(f"{heights=}")

    # Filter out outliers
    filtered_heights = list()
    for height in heights:
        dh = abs(height - median_height)
        if dh < 0.5 * median_height:
            filtered_heights.append(height)
    filtered_heights.sort()
    print(f"{filtered_heights=}")

    if len(filtered_heights) % 2 == 0:
        median_height = (
            filtered_heights[len(filtered_heights) // 2 - 1]
            + filtered_heights[len(filtered_heights) // 2]
        ) / 2
    else:
        median_height = filtered_heights[len(filtered_heights) // 2]
    mean_height = sum(filtered_heights) / len(filtered_heights)
    print(f"{mean_height=}")
    return max(median_height, filtered_heights[-1] / 2, round(mean_height))
    return median_height


@timer
def find_connected_components_bbox(image, min_area=0, connectivity=8):
    def dfs(x, y):
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if connectivity == 8:
            nbrs.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

        min_x, min_y, max_x, max_y = x, y, x, y
        stack = [(x, y)]
        while stack != list():
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

                for dx, dy in nbrs:
                    stack.append((cx + dx, cy + dy))
        return min_x, min_y, max_x, max_y

    visited = np.zeros_like(image, dtype=bool)
    bounding_boxes = list()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not visited[i, j] and image[i, j] == 255:
                min_x, min_y, max_x, max_y = dfs(i, j)
                bounding_boxes.append((min_x, min_y, max_x, max_y))

    return bounding_boxes


vid_images = list()


def count_lines(bboxes, orig):
    global vid_images
    if len(bboxes) == 0:
        return 0

    bboxes = sorted(bboxes, key=lambda bbox: (bbox[0], bbox[2]))

    lines = 1
    prev1, _, prev2, _ = bboxes[0]
    # Iterate through the sorted bounding boxes starting from the second one
    xdiffs_bboxes = list()
    for bbox in bboxes[1:]:
        y1, _, y2, _ = bbox
        overlap = max(0, min(prev2, y2) - max(prev1, y1))
        # print(f"{overlap=}")
        if overlap > abs(prev1 - prev2) / 100:
            continue
        lines += 1
        prev1 = y1
        prev2 = y2

        vid_img = orig.copy()
        rectangle(vid_img, (0, y1), (orig.shape[0] - 1, y2), (0, 0, 255), 2)
        vid_images.append(vid_img)

    return lines


def count_columns(bboxes):
    # Sort by left x-coordinate
    sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[1])

    num_columns = 0
    max_right = float("-inf")

    for bbox in sorted_bboxes:
        _, left, _, right = bbox

        # If the left x-coordinate of the bounding box is greater than the current rightmost x-coordinate,
        # it indicates the start of a new column
        if left > max_right:
            num_columns += 1

        # Update current rightmost x-coordinate
        max_right = max(max_right, right)

    return num_columns


def distance(bbox1, bbox2):
    """
        Considering manhatan distance and overlapping too.
        Calculate the distance between the closest edges of two bounding boxes.
    """
    if bbox_overlap(bbox1, bbox2):
        return 0

    min_distance = float("inf")

    l1, t1, r1, b1 = bbox1
    l2, t2, r2, b2 = bbox2
    assert l1 < r1 and t1 < b1
    assert l2 < r2 and t2 < b2

    overlap_y = min(b1, b2) - max(t1, t2)
    overlap_x = min(r1, r2) - max(l1, l2)
    if overlap_y > 0:
        if overlap_x > 0:
            print(f"{bbox1,bbox2=}")
            exit(1)
        l1_dist = min(abs(l1 - l2), abs(l1 - r2))
        r1_dist = min(abs(r1 - l2), abs(r1 - r2))
        dist = min(r1_dist, l1_dist)
        if dist < min_distance:
            min_distance = dist

    if overlap_x > 0:
        assert not overlap_y > 0
        b1_dist = min(abs(b1 - b2), abs(b1 - r2))
        t1_dist = min(abs(t1 - b2), abs(t1 - t2))
        dist = min(t1_dist, b1_dist)
        if dist < min_distance:
            min_distance = dist

    if overlap_y > 0:
        l1_dist = min(abs(l1 - l2), abs(l1 - r2))
        r1_dist = min(abs(r1 - l2), abs(r1 - r2))
        dist = min(r1_dist, l1_dist)
        if dist < min_distance:
            min_distance = dist

    for x1, y1 in [(l1, t1), (r1, t1), (r1, b1), (l1, b1)]:
        for x2, y2 in [
            (l2, t2),
            (r2, t2),
            (r2, b2),
            (l2, b2),
        ]:
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < min_distance:
                min_distance = dist

    return min_distance


def bbox_overlap(bbox1, bbox2):
    """
        Check if two bounding boxes overlap.
    """
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    overlap_left = max(left1, left2)
    overlap_top = max(top1, top2)
    overlap_right = min(right1, right2)
    overlap_bottom = min(bottom1, bottom2)

    # If the overlap area is non-negative, the bounding boxes overlap
    if overlap_right > overlap_left and overlap_bottom > overlap_top:
        overlap_y = min(bottom1, bottom2) - max(top1, top2)
        overlap_x = min(right1, right2) - max(left1, left2)
        assert overlap_y > 0 and overlap_x > 0
        return True
    else:
        return False


@timer
def group_bboxes(bboxes, max_distance, image) -> list[list]:
    """
        Group bboxes based on a max distance.
        `image` is video for purposes
    """
    result = list()
    rest_idxs = [i for i in range(len(bboxes))]
    while len(rest_idxs) > 0:
        close_idxs = [rest_idxs.pop()]
        bbox = bboxes[close_idxs[0]]
        counter = 0
        while counter < len(rest_idxs):
            r_idx = rest_idxs[counter]
            counter += 1
            dist = distance(bbox, bboxes[r_idx])
            if dist < max_distance or bbox_overlap(bbox, bboxes[r_idx]):
                close_idxs.append(r_idx)
                rest_idxs.remove(r_idx)
                bbox = enclosing_bbox([bbox, bboxes[r_idx]])
                counter = 0

                vid_img = image.copy()
                x, y, x2, y2 = bbox
                rectangle(vid_img, (y, x), (y2, x2), (0, 244, 55), 2)
                vid_images.append(vid_img)

        x, y, x2, y2 = bbox
        rectangle(image, (y, x), (y2, x2), (0, 244, 55), 2)
        result.append([bboxes[i] for i in close_idxs])
    return result


@timer
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


@timer
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
