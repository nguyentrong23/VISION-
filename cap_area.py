import cv2
import numpy as np

def get_gradient_sobel(image, index):
    blurred = cv2.pyrMeanShiftFiltering(image,25,15)
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_image',gray_image)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_angle = np.zeros_like(mag, dtype=np.uint8)
    gradient_angle[mag > 30] = 255
    # cv2.imshow('mag', gradient_angle)
    # _, binary_image = cv2.threshold(gray_image,80, 160, cv2.THRESH_BINARY_INV)
    # cv2.imshow('binary_image',binary_image)
    contours, _ = cv2.findContours(gradient_angle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    result_image = np.zeros_like(image)
    cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 1)
    area = cv2.contourArea(largest_contour)
    cv2.imshow(f'edges {index} ',result_image)
    print(f"area  {index} = ",area )
    return area


image1 = cv2.imread('data\imgSrc\captured_image1.jpg')
image1= cv2.pyrUp(image1)
edges1 = get_gradient_sobel(image1,1)

image2 = cv2.imread('data\imgSrc\captured_image2.jpg')
image2 = cv2.pyrUp(image2)
edges2 = get_gradient_sobel(image2,2)
cv2.waitKey(0)
cv2.destroyAllWindows()