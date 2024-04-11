import os
import random
import numpy as np

from kernels import *
from utils import *

vid_images = list()

def invert(image):
    return np.ones(image.shape) * 255 - image


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
    try:
        assert len(image.shape) == 3 and image.shape[-1] == len(color)  # 3 dimensional
    except:
        assert isinstance(int(color), int) and len(image.shape) == 2  # 1 dimensional
    y1, x1 = pt1
    y2, x2 = pt2

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
def fast_median_blur(image, filter_size=3):
    assert len(image.shape) == 2

    height = image.shape[0]
    width = image.shape[1]

    result = image.copy()

    for y in range(height-filter_size):
        for x in range(width-filter_size):
            # Get only the part that we care about
            region = image[y:y+filter_size, x:x+filter_size]
            # transform in a 1d array
            values = region.flatten()
            values.sort()
            median_value = values[len(values) // 2]
            result[y, x] = median_value

    return result


@timer
def understandable_median_blur(image, filter_size=3):
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
def understandable_erode(image, kernel):
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

def fast_erode(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    kernel_width_delta = kernel_width // 2
    kernel_height_delta = kernel_height // 2

    # We pad by the kernel delta, top, bottom, left and right
    padded_image = pad(
        image,
        kernel_height_delta,
        kernel_height_delta,
        kernel_width_delta,
        kernel_width_delta,
    )

    eroded = np.ones(image.shape) * 255
    for j in range(kernel_height):
        for i in range(kernel_width):
            if kernel[j, i] == 1:
                shifted_sub_image = padded_image[j : j + height, i : i + width]
                eroded = np.minimum(
                    eroded, shifted_sub_image
                )
    return eroded

@timer
def understandable_dilate(image, kernel, iterations=1):
    result = np.zeros(image.shape)
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
                    kcolor = kernel[j, i]
                    # if kernel_color > 0 and color > 0: # single cause of complete lag
                    if int(kcolor) * int(color):
                        all_good = True
                        break
                if all_good:
                    break
            if all_good:
                result[y, x] = 255
            else:
                result[y, x] = 0
    return result


def pad(image, left, right, top, bottom):
    assert len(image.shape) == 2, "Padding only works for gray_scale images right now."
    return np.pad(
        image, ((top, bottom), (left, right)), mode="constant", constant_values=0
    )


@timer 
def fast_dilate2(image, kernel):
    global counter 
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    kernel_width_delta = kernel_width // 2
    kernel_height_delta = kernel_height // 2

    # We pad by the kernel delta, top, bottom, left and right
    padded_image = pad(
        image,
        kernel_height_delta,
        kernel_height_delta,
        kernel_width_delta,
        kernel_width_delta,
    )

    dilated = np.zeros(image.shape, dtype=np.uint8)
    for j in range(kernel_height):
        for i in range(kernel_width):
            if kernel[j, i] == 1:
                shifted_sub_image = padded_image[j : j + height, i : i + width]
                dilated = np.maximum(
                    dilated, shifted_sub_image
                )

    return dilated


@timer
def fast_dilate(image, kernel):
    assert kernel.shape[0] % 2 != 0, f"{kernel.shape=}"

    result = np.zeros(image.shape, dtype=np.uint8)
    height = image.shape[0]
    width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    kernel_width_delta = kernel_width // 2
    kernel_height_delta = kernel_height // 2

    image = pad(
        image,
        kernel_height_delta,
        kernel_height_delta,
        kernel_width_delta,
        kernel_width_delta,
    )

    
    for y in range(kernel_height_delta, height):
        for x in range(kernel_width_delta,width):
            sub_result = kernel * image[y-kernel_height_delta: y+kernel_height_delta+1,x-kernel_width_delta: x+kernel_width_delta+1]
            result[y, x] = sub_result.max()

    assert tuple(result.shape) == (height, width)
    return result


def dilate(image, kernel, iterations=1):
    assert kernel.shape[0] % 2 != 0, f"{kernel.shape=}"
    # Professora Bia, aqui eu disponho diferentes implementações de dilatação
    # understandable_dilate é o mais lento que tenta ser o mais entendivel possivel.
    # Ja fast_dilate2 é bem mais rapido, porém usa a função `maximum` da estrutura de dados matrix do  numpy.
    # é uma função que como o nome diz pega o maximo entre valores, mas funciona para matrix.
    # Eu dispus dessa maneira pra mostra que sei fazer todas as diferentes implementações,
    # mas pra ficar mais rapido e interativo é usado o fast_dilate2.
    # NOTE: Qualquer uma dessas implementações funciona para os propositos deste trabalho
    choice = (understandable_dilate, fast_dilate, fast_dilate2)[2]
    
    for _ in range(iterations):
        image = choice(image, kernel)
    return image

def erode(image, kernel, iterations=1):
    # Mesma coisa do `dilate` e `median_blur`
    choice = (understandable_erode, fast_erode)[1]
    
    for _ in range(iterations):
        image = choice(image, kernel)
    return image

def median_blur(image, filter_size, iterations=1):
    assert filter_size % 2 != 0, f"{kernel.shape=}"
    # Professora Bia, aqui ocorre o mesmo que na função `dilate`, tenho duas implementaões
    # que funcionam perfeitamente. Na versão fast, uso o sort do numpy e faço um padding na imagem
    # Novamente, só para deixar claro que é só uma maneira de contornar o sort das litas
    # lentas do python, a diferença é enorme. 
    # Tanto no calo do `dilate` como aqui no `median_blur` o algoritmo continua sendo implementado e entendido pelo grupo,
    # já a versões `fast` usam funções simples do numpy muito similares as do python, mas feito na matrix inteira :).
    choice = (understandable_median_blur, fast_median_blur)[1]
    
    for _ in range(iterations):
        image = choice(image, filter_size)
    return image



def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height
    return area


@timer
def choose_best_height(bboxes, outlier_constant=0.5):
    heights = list()

    for bbox in bboxes:
        min_x, min_y, max_x, max_y = bbox
        height = abs(max_x - min_x)
        heights.append(height)

    # Calculate the median height
    heights.sort()
    median_height = heights[len(heights) // 2]

    # Filter out outliers
    filtered_heights = list()
    for height in heights:
        dh = abs(height - median_height)
        if dh < outlier_constant * median_height:
            filtered_heights.append(height)
    filtered_heights.sort()

    if len(filtered_heights) % 2 == 0:
        median_height = (
            filtered_heights[len(filtered_heights) // 2 - 1]
            + filtered_heights[len(filtered_heights) // 2]
        ) / 2
    else:
        median_height = filtered_heights[len(filtered_heights) // 2]
    mean_height = sum(filtered_heights) / len(filtered_heights)

    return median_height


@timer
def find_connected_components_bboxes(image, min_area=0, connectivity=8):
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        nbrs.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    def dfs(y, x):
        nonlocal nbrs, image
        min_y, min_x, max_x, max_y = y, x, y, x
        stack = [(y, x)]
        while stack != list():
            cy, cx = stack.pop()
            if (
                0 <= cy < image.shape[0]
                and 0 <= cx < image.shape[1]
                and not visited[cy, cx]
                and image[cy, cx] == 255
            ):
                visited[cy, cx] = True
                min_y = min(min_y, cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cy)
                max_y = max(max_y, cx)

                for dy, dx in nbrs:
                    stack.append((cy + dy, cx + dx))
        return min_y, min_x, max_x, max_y

    visited = np.zeros(image.shape, dtype=bool)
    bounding_boxes = list()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if not visited[y, x] and image[y, x] == 255:
                min_y, min_x, max_x, max_y = dfs(y, x)
                bounding_boxes.append((min_y, min_x, max_x, max_y))

    return bounding_boxes


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
        if overlap > abs(prev1 - prev2) / 100:
            continue
        lines += 1
        prev1 = y1
        prev2 = y2

        vid_img = orig.copy()
        rectangle(vid_img, (y1, 0), (y2, orig.shape[1] - 1), (0, 0, 255), 2)
        vid_images.append(vid_img)

    return lines


def count_columns(bboxes, orig):
    # Sort by left x-coordinate
    sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[1])

    num_columns = 0
    max_right = float("-inf")

    vid_img = orig.copy()
    for bbox in sorted_bboxes:
        _, left, _, right = bbox

        # If the left x-coordinate of the bounding box is greater than the current rightmost x-coordinate,
        # it indicates the start of a new column
        if left > max_right:
            num_columns += 1
            rectangle(
                vid_img,
                (0, left),
                (orig.shape[0]-2, left+1),
                (150, 160, 180),
                2,
            )
            vid_images.append(vid_img)

        # Update current rightmost x-coordinate
        if max_right < right:
            max_right = right

    vid_images.append(vid_img)
    return num_columns


def distance(bbox1, bbox2):
    """
    Considering manhatan distance and overlapping too.
    Calculate the distance between the closest edges of two bounding boxes.
    """
    if bbox_overlap(bbox1, bbox2):
        return 0

    min_distance = float("inf")

    # top, left, bottom, right, top is physically top, but has less value than b
    t1, l1, b1, r1 = bbox1
    t2, l2, b2, r2 = bbox2

    assert l1 <= r1 and t1 <= b1, f"{bbox1=}"
    assert l2 <= r2 and t2 <= b2, f"{bbox2=}"

    overlap_y = min(b1, b2) - max(t1, t2)
    overlap_x = min(r1, r2) - max(l1, l2)
    if overlap_y > 0:
        assert not overlap_x > 0
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
    t1, l1, b1, r1 = bbox1  # top, left, bottom, right
    t2, l2, b2, r2 = bbox2

    overlap_left = max(l1, l2)
    overlap_top = max(t1, t2)
    overlap_right = min(r1, r2)
    overlap_bottom = min(b1, b2)

    # If the overlap area is non-negative, the bounding boxes overlap
    if overlap_right > overlap_left and overlap_bottom > overlap_top:
        overlap_y = min(b1, b2) - max(t1, t2)
        overlap_x = min(r1, r2) - max(l1, l2)
        assert overlap_y > 0 and overlap_x > 0
        return True
    else:
        return False


def _bbox_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    """
    b1, l1, t1, r1 = bbox1  # bottom, left, top, right
    b2, l2, t2, r2 = bbox2

    overlap_left = max(l1, l2)
    overlap_top = max(t1, t2)
    overlap_right = min(r1, r2)
    overlap_bottom = min(b1, b2)

    # If the overlap area is non-negative, the bounding boxes overlap
    if overlap_right > overlap_left and overlap_bottom > overlap_top:
        overlap_y = min(b1, b2) - max(t1, t2)
        overlap_x = min(r1, r2) - max(l1, l2)
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
            if (dist < max_distance) or bbox_overlap(bbox, bboxes[r_idx]):
                close_idxs.append(r_idx)
                rest_idxs.remove(r_idx)
                bbox = enclosing_bbox([bbox, bboxes[r_idx]])
                counter = 0

                vid_img = image.copy()
                y, x, y2, x2 = bbox
                rectangle(vid_img, (y, x), (y2, x2), (0, 244, 55), 2)
                vid_images.append(vid_img)

        y, x, y2, x2 = bbox
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


def enclosing_bbox(bboxes):
    if not bboxes:
        return None

    # top is closer to 0 than bottom, left is closer to 0 than right
    t, l, b, r = bboxes[0]

    # Iterate through the rest of the bounding boxes to update min and max coordinates
    for bbox in bboxes[1:]:
        t = min(t, bbox[0])
        l = min(l, bbox[1])
        b = max(b, bbox[2])
        r = max(r, bbox[3])

    # Return the enclosing bounding box
    return t, l, b, r

