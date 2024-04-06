import cv2
import pytesseract
import numpy as np

# image = cv2.imread('./assets/ocr/lorem_s12_c03.pbm')
image = cv2.imread('./assets/ocr/lorem_s12_c02_noise.pbm')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Preprocessing (e.g., thresholding)
median_filtered_image = cv2.medianBlur(gray_image, 3)
_, binary_image = cv2.threshold(median_filtered_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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

kernel = create_text_kernel(7)

cv2.imwrite('text_kernel.png', kernel)

# Morphological operations: dilation and erosion
dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
eroded_image = cv2.erode(dilated_image, kernel, iterations=2)
finished_image = eroded_image

cv2.imwrite('finished.png', finished_image)

# Connected component analysis (CCA)
_, labels, stats, centroids = cv2.connectedComponentsWithStats(finished_image)
print(labels, stats, centroids)
# _, labels, stats, centroids = connectedComponentsWithStats(eroded_image)


# Iterate through the detected components (excluding background)
for i in range(1, len(stats)):
    x, y, w, h, area = stats[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imwrite('detected.png', image)
