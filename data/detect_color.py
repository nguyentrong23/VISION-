import cv2
import numpy as np


def detect_and_draw_contours(frame, lower_limit, upper_limit):
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Tạo mặt nạ dựa trên ngưỡng màu
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    # Tìm các đường viền
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Lấy tọa độ hộp bao quanh đối tượng
        x, y, w, h = cv2.boundingRect(contour)
        # Vẽ hộp bao quanh đối tượng
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Vẽ đường viền
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

    return frame


# Định nghĩa ngưỡng màu cho màu vàng
lower_yellow = np.array([20, 100, 100], dtype=np.uint8)    #  màu vàng trong hsv  HUE(20- 60 độ),
upper_yellow = np.array([40, 255, 255], dtype=np.uint8)    # Saturation độ đậm của màu (nếu =0 = màu trằng)   # Value đại diện cho độ sáng của màu  (nếu =0 = màu đen) vì không có độ sáng

# Mở kết nối video từ camera
cap = cv2.VideoCapture('http://192.168.1.102:8080/video')

while True:
    # Đọc frame từ video
    ret, frame = cap.read()
    frame = cv2.resize(frame, (860, 620))
    # Kiểm tra xem frame có hợp lệ hay không
    if not ret:
        break
    # Gọi hàm nhận diện màu và vẽ đường viền
    result_frame = detect_and_draw_contours(frame, lower_yellow, upper_yellow)
    # Hiển thị frame kết quả
    cv2.imshow('Result Frame', result_frame)

    # Thoát vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
