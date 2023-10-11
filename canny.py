import cv2
import numpy as np
image = cv2.imread('tets.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('blurred', blurred)

edges = cv2.Canny(blurred, 100, 150)
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
filtered_image = cv2.filter2D(image, -1, kernel)
cv2.imshow('filtered_image', filtered_image)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
erosion = cv2.erode(edges,kernel,iterations = 1)
alpha = 1.5  # độ tương phản
beta = 30   # độ sáng
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imshow('Processed Image', adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('ten_file_ket_qua.jpg', adjusted)


