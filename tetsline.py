import cv2
import numpy as np
import math

def get_gradient_sobel(image):
    blurred = cv2.pyrMeanShiftFiltering(image, 60, 120)
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    direc_angle = np.degrees(np.arctan2(sobel_y, sobel_x))
    gradient_angle = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle_flip = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle[direc_angle == 90] = 255
    gradient_angle_flip[direc_angle == -90] = 255
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary_image, 100, 200)
    point_top = cv2.bitwise_and(gradient_angle, edges)
    point_bottom = cv2.bitwise_and(gradient_angle_flip, edges)
    data_bottom = np.where(point_top != 0)
    data_bottom = np.column_stack((data_bottom[1], data_bottom[0]))
    data_top = np.where(point_bottom != 0)
    data_top = np.column_stack((data_top[1], data_top[0]))
    cv2.imshow('00', gradient_angle)
    cv2.imshow('01', gradient_angle_flip)
    cv2.imshow('1', point_top)
    cv2.imshow('2', point_bottom)
    return edges, data_top,data_bottom

# Đọc ảnh
image = cv2.imread('data/Test Images/NG01_lite.jpg')
edges, TopLine, Botline = get_gradient_sobel(image)
print(TopLine)
cv2.waitKey(0)
cv2.destroyAllWindows()


