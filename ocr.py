import cv2
import pytesseract
import numpy as np

# image = cv2.imread('./assets/ocr/lorem_s12_c03.pbm')
orig = cv2.imread('./assets/ocr/lorem_s12_c02_noise.pbm')

# Convert the image to grayscale
image = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
# Preprocessing (e.g., thresholding)
image = cv2.medianBlur(image, 3)
_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

def hit_or_miss(binary_image, structuring_element=None):
    # Define the default structuring element if not provided
    if structuring_element is None:
        structuring_element = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ], dtype=np.uint8)
    
    # Invert the binary image
    inverted_image = cv2.bitwise_not(binary_image)
    
    # Perform erosion with the structuring element
    erosion = cv2.erode(inverted_image, structuring_element)
    
    # Invert the result to get the hit-or-miss operation
    result = cv2.bitwise_not(erosion)
    
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
            if x*x + y*y <= radius*radius:
                kernel[x][y] = 1
    
    return kernel

kernel = create_text_kernel(7)
kernel = create_circular_kernel(7)

cv2.imwrite('text_kernel.png', kernel)


# Morphological operations: dilation and erosion
image  = cv2.dilate(image, kernel, iterations=1)
image  = cv2.erode(image, kernel, iterations=1)

image  = cv2.dilate(image, kernel, iterations=1)
image  = cv2.erode(image, kernel, iterations=1)

image  = cv2.dilate(image, kernel, iterations=1)
image  = cv2.erode(image, kernel, iterations=1)

cv2.imwrite('morphological.png', image)

image = hit_or_miss(image)
cv2.imwrite('hit_or_miss.png', image)

finished_image = image

cv2.imwrite('finished.png', finished_image)

def find_median_height(bboxes):
    # List to store heights of bounding boxes
    heights = []

    # Iterate over all bounding boxes
    for bbox in bboxes:
        # Extract ymin and ymax from each bounding box
        _, min_y, _, max_y = bbox
        # Calculate height and append to the list
        height = max_y - min_y
        heights.append(height)

    # Sort the list of heights
    heights.sort()

    # Calculate the median height
    if len(heights) % 2 == 0:
        # If number of heights is even, take average of middle two heights
        median_height = (heights[len(heights)//2 - 1] + heights[len(heights)//2]) / 2
    else:
        # If number of heights is odd, take the middle height
        median_height = heights[len(heights)//2]

    return median_height

def find_connected_components_bbox(image):
    def dfs(x, y):
        min_x, min_y, max_x, max_y = x, y, x, y
        stack = [(x, y)]
        while stack != []:
            cx, cy = stack.pop()
            if 0 <= cx < image.shape[0] and 0 <= cy < image.shape[1] and not visited[cx, cy] and image[cx, cy] == 255:
                visited[cx, cy] = True
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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
using_cv = False
if using_cv:
    # Connected component analysis (CCA)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(finished_image)
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        if area > 10*10:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

bboxes = find_connected_components_bbox(finished_image)
median_height = find_median_height(bboxes)
min_area = median_height*median_height
for bbox in bboxes:
    min_x, min_y, max_x, max_y = bbox
    cv2.rectangle(orig, (min_y, min_x), (max_y, max_x), (0, 0, 255), 2)



# print(labels, stats, centroids)
# _, labels, stats, centroids = connectedComponentsWithStats(eroded_image)


# Iterate through the detected components (excluding background)

# Display the result
cv2.imwrite('detected.png', orig)
