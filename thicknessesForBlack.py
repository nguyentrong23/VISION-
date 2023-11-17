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
    return edges, data_top,data_bottom

def fit_pca(data, src):
    data = np.float32(data)
    mean, eigenvectors = cv2.PCACompute(data, mean=None)
    mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
    cv2.circle(src, mean_point, 3, (0, 0, 255), -1)
    scale = 100
    vector1_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
    cv2.arrowedLine(src, mean_point, vector1_end, (0, 255,255), 1)

    # Find min and max values along the direction of vector1_end
    min_val = np.min(np.dot(data - mean, eigenvectors.T))
    max_val = np.max(np.dot(data - mean, eigenvectors.T))

    # Draw a line covering the entire range of data along vector1_end
    line_start = (int(mean_point[0] + eigenvectors[0][0] * min_val), int(mean_point[1] + eigenvectors[0][1] * min_val))
    line_end = (int(mean_point[0] + eigenvectors[0][0] * max_val), int(mean_point[1] + eigenvectors[0][1] * max_val))
    cv2.line(src, line_start, line_end, (255,255, 0), 1)

    cv2.imshow('3', src)
    return vector1_end,min_val,max_val

def phandoan(src):
    return 0
# Đọc ảnh
image = cv2.imread('data/Test Images/NG001_lite.jpg')
image = cv2.pyrUp(image)
image = cv2.pyrUp(image)
edges, TopLine, Botline = get_gradient_sobel(image)
vector_top,xmin_top, xmax_top=fit_pca(TopLine,image)
vector_bot,xmin_bot, xmax_bot=fit_pca( Botline,image)
distance = cv2.norm(vector_top, vector_bot)
print(f'Khoảng cách giữa hai vector là: {distance}')
print(f'độ phân giải ảnh là: {edges.shape[::]}')
cv2.waitKey(0)
cv2.destroyAllWindows()


