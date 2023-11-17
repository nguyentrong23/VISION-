import cv2
import numpy as np

# Đọc ảnh từ file
image = cv2.imread('data\Test Images/t1.jpg')
image = cv2.pyrDown(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

b = cv2.GaussianBlur(gray, (7,7), 0)

normalized = cv2.equalizeHist(b)

# Điều chỉnh độ sáng và độ tương phản
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
enhanced = clahe.apply(normalized)

# Tăng độ tương phản bằng cách sử dụng tỷ lệ và chênh lệch
alpha = 1.5  # Tỷ lệ độ tương phản (có thể điều chỉnh)
beta = 40    # Chênh lệch độ tương phản (có thể điều chỉnh)
enhanced_contrast = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)


_, binary_image = cv2.threshold(enhanced_contrast,180, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((15,15), np.uint8)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(binary_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

largest_contour = max(contours, key=cv2.contourArea)

result_image = np.zeros_like(image)

cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)


cv2.imshow('edges',result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
