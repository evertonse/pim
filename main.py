import cv2
import numpy as np

# Define your custom kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

kernel = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
])


kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1], 
], dtype='float64')

kernel /= np.sum(kernel)

print(np.sum(kernel))


# Load the input image
image = cv2.imread('./assets/images/lua.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the convolution operation
convolved_image = cv2.filter2D(gray_image, -1, kernel)

# Display the original and convolved images
cv2.imwrite('original.png',  gray_image)
cv2.imwrite('convolved.png', convolved_image)

