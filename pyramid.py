import cv2
import numpy as np

# Đọc ảnh gốc và template
image = cv2.imread('data/sample-2-10.bmp', cv2.IMREAD_COLOR)
template = cv2.imread('data/sample_for_template.bmp', cv2.IMREAD_COLOR)

# Khởi tạo tỷ lệ scale (0.8 có thể điều chỉnh)
scale_factor = 0.8
found = None

# Bắt đầu quá trình tìm kiếm với các tỷ lệ khác nhau
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # Thay đổi kích thước template
    resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))

    # Template Matching
    result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # Lưu lại kết quả tốt nhất
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, scale)

# Lấy thông tin kết quả
(_, maxLoc, scale) = found
startX, startY = int(maxLoc[0] / scale), int(maxLoc[1] / scale)
endX, endY = int((maxLoc[0] + template.shape[1]) / scale), int((maxLoc[1] + template.shape[0]) / scale)

# Vẽ hình chữ nhật bao quanh kết quả
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Hiển thị kết quả
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
