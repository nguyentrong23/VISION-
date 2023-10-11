import cv2 as cv
import numpy as np
image = cv.imread("matrix.png")
(rows, cols) = image.shape[:2]

#tinh toan ma tran affine voi mot tam cu the  "center", su dung phep xoay ma tran va co dan~ dc
center = (cols / 2, rows / 2)
angle = 30
scale = 1

# Tính toán ma trận biến đổi affine
M = cv.getRotationMatrix2D(center, angle, scale)
output = cv.warpAffine(image, M, (cols, rows))
cv.imshow('Affine Transformed xoay style 1', output)

#cach khac: cho 3  diem cu the de tien hanh tim ma tran affine

input = np.float32([[0,0],[cols-1,0],[0,rows-1]])
output= np.float32([[0,0],[cols/2,0],[cols/2,rows-1]])
M = cv.getAffineTransform(input,output)
# Thực hiện biến đổi affine
output_image = cv.warpAffine(image, M, (cols, rows))

# Hiển thị hình ảnh gốc và hình ảnh sau biến đổi
cv.imshow('Original Image', image)
cv.imshow('Affine Transformed xoay style 2', output_image)
cv.waitKey(0)
cv.destroyAllWindows()