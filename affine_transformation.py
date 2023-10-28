import cv2 as cv
import numpy as np

# Đọc ảnh
image = cv.imread("data/imgSrc/teamplate_0_degree.bmp")
(rows, cols) = image.shape[:2]

# Tính toán ma trận biến đổi affine để quay ngược chiều kim đồng hồ
center = (cols / 2, rows / 2)
angle = -90  # Góc âm để quay ngược chiều kim đồng hồ
scale = 1

M = cv.getRotationMatrix2D(center, angle, scale)

# Thực hiện biến đổi affine
output_image = cv.warpAffine(image, M, (cols, rows))

# Hiển thị hình ảnh gốc và hình ảnh sau biến đổi
cv.imshow('Original Image', image)
cv.imshow('Counter Clockwise Rotated Image', output_image)
cv.waitKey(0)
cv.destroyAllWindows()
