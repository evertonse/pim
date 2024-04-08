import cv2
import numpy as np

# Define your custom kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

kernel_laplacian = np.array([
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1]
], dtype='float64')

kernel_laplacian_sharp = np.array([
    [-1,  -1, -1],
    [-1,   9, -1],
    [-1,  -1, -1]
], dtype='float64')


kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

kernel_sobel_y = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
])


kernel_gaussian_blur = np.array([
    [1.0/16, 2.0/16, 1.0/16],
    [2.0/16, 4.0/16, 2.0/16],
    [1.0/16, 2.0/16, 1.0/16], 
], dtype='float64')

kernel_gaussian_blur5x5 = np.array([
    [1,4,7,4,1],
    [4,16,26,16,4],
    [7,26,41,26,7],
    [4,16,26,16,4],
    [1,4,7,4,1],
], dtype='float64')/273


def unsharp(img, amount=5):
    # blur = cv2.filter2D(orig, -1,kernel_gaussian_blur5x5)
    blur = cv2.filter2D(img, -1, kernel_gaussian_blur5x5)
    img2 = (img-blur)
    img3 = (img+amount*img2)

    cv2.imwrite('img2.png', img2)
    cv2.imwrite('img3.png', img3)
    return img3 

def laplacian_sharpening(img):
    edges = cv2.filter2D(img, -1, kernel_laplacian)
    return img - edges

def histrogram(img, L=256):
    histogram = [0]*L
    for p in range(L): 
        histogram[p] = 0.0
    for j in range(img.shape[0]) :
        for i in range(img.shape[1]) :
            x,y = i,j
            color = img[x,y]
            intesity = int(color)
            histogram[intesity] += 1.0
    print(histogram)
    histogram /= float(img.shape[0] * img.shape[1])
    for p in range(1,L): 
        histogram[p] += histogram[p-1]
    print(histogram)
    for p in range(L):
        histogram[p] *= L-1
    print(histogram)


# Load the input image
# bgr_image = cv2.imread('./assets/images/beans.png')
# bgr_image = cv2.imread('./assets/images/lua.png')
# bgr_image = cv2.imread('./assets/images/xadrez.png')
# bgr_image = cv2.imread('./assets/images/half_black_half_white.png')
# bgr_image = cv2.imread('./assets/images/salted_1.png')
bgr_image = cv2.imread('./assets/images/lago_escuro.png')

# Convert the image to grayscale
image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
image = np.array(image, dtype='float64')

image = np.array([
    [9,9,9,9,9,9,9,9,9],
    [9,7,7,7,7,7,7,7,9],
    [9,7,5,5,5,5,5,7,9],
    [9,7,5,3,3,3,5,7,9],
    [9,7,5,3,1,3,5,7,9],
    [9,7,5,3,3,3,5,7,9],
    [9,7,5,5,5,5,5,7,9],
    [9,7,7,7,7,7,7,7,9],
    [9,9,9,9,9,9,9,9,9],
], dtype='float32')

# Aplicar filtro de mediana
image = cv2.medianBlur(image, 3)
cv2.imwrite('median.png', image)
print(image)


laplacian_sharped = laplacian_sharpening(image)
cv2.imwrite('laplacian_sharped.png', laplacian_sharped)

# convolved_image = cv2.filter2D(image, -1, kernel_laplacian_sharp)
convolved_image = cv2.filter2D(image, -1, kernel_sobel_x)
for i in range(1000):
    convolved_image = cv2.filter2D(image, -1, kernel_gaussian_blur)
    image = convolved_image
unsharped_image = unsharp(image)

# Display the original and convolved images
cv2.imwrite('original.png',  image)
cv2.imwrite('convolved.png', convolved_image)
cv2.imwrite('unsharp.png', unsharped_image)


