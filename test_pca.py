import cv2
import numpy as np
import math
import imutils
import time

def fit_angel_pca(contours, src):
    min_area = 50000
    index = np.array([])
    temp_con = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            temp_con = cnt
            cv2.drawContours(src, [cnt], -1, (0, 255, 0), 2, cv2.LINE_AA)
            index = np.append(index,temp_con)
    for i in index:
        data = index[i][:, 0, :].astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(data, mean=None)
        angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
        # Tọa độ điểm mean (làm tròn)
        mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
        # Kích thước vector
        scale = 100
        # Vector theo eigenvector  chính
        vector1_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
        # Vector theo eigenvector phụ
        # vector2_end = (int(mean_point[0] + eigenvectors[1][0] * scale), int(mean_point[1] + eigenvectors[1][1] * scale))
        # Vẽ hình tròn (màu đỏ) tại điểm mean
        cv2.circle(src, mean_point, 5, (0, 0, 255), -1)
        # Vẽ các vector
        cv2.arrowedLine(src, mean_point, vector1_end, (0, 255, 0), 2)
        # cv2.arrowedLine(src, mean_point, vector2_end, (0, 255, 0), 2)
    cv2.imshow("dectect", src)
    return mean, angel, src


def rotate_point(x, y, image, angle):
    (rows, cols) = image.shape[:2]
    # Tính toán ma trận biến đổi affine để quay điểm (x, y)
    center = (cols / 2, rows / 2)
    scale = 1
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # Thực hiện biến đổi affine cho điểm
    new_point = np.dot(M, np.array([x, y, 1]))
    # Lấy tọa độ mới
    new_x, new_y = new_point[:2]
    return int(new_x), int(new_y)



# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("data/imgSrc/sample-5.bmp")
img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
_, edges_src = cv2.threshold(blurred, 100, 160, cv2.THRESH_BINARY_INV)
# edges_src = cv2.Canny(threshold, 120, 150)
contours, _ = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#  đọc  và tiền xử lý template
sr1= cv2.imread("data/imgSrc/template.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
_, template = cv2.threshold(blurred, 100, 160, cv2.THRESH_BINARY_INV)
# edges_template = cv2.Canny(blurred, 100, 150)
#  đọc  và tiền xử lý template flip
sr1_flip= cv2.imread("data/imgSrc/template-flip.bmp")
img_template_flip = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
_, template_inv = cv2.threshold(blurred, 100, 160, cv2.THRESH_BINARY_INV)


# init parameter
# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
topleft = [0,0]
template_copy = np.zeros_like(template)
best_angel = 0
src_copy = np.zeros_like(edges_src)


start_time = time.time()
# resolve angel problem by pca
center_cons, angel_cons, src = fit_angel_pca(contours,sr0)

low_angel = float(angel_cons - 5)
high_angel = float(angel_cons + 5)
threshold = 0.0
i = low_angel
while i< high_angel:
       rotated_src = imutils.rotate(edges_src,i)
       res = cv2.matchTemplate(rotated_src,template, method)
       res_inv = cv2.matchTemplate(rotated_src, template_inv, method)
       _, maxval,_, maxloc = cv2.minMaxLoc(res)
       _, maxval_inv, _, maxloc_inv = cv2.minMaxLoc(res_inv)
       if(maxval>=threshold):
           threshold = maxval
           src_copy = rotated_src
           best_angel = i
           topleft = maxloc
       if (maxval_inv >= threshold):
           threshold = maxval_inv
           src_copy = rotated_src
           best_angel = i
           topleft = maxloc_inv
       i += 0.5
print(threshold, ':', best_angel)
bottomright= (topleft[0]+w,topleft[1]+h)
cv2.rectangle(sr0,topleft,bottomright,(0,255,255),1)
cv2.imshow("src_copy",src_copy)
cv2.imshow("dectect",sr0)
end_time = time.time()
execution_time = end_time - start_time
print(f"Thời gian chạy: {execution_time} giây")
cv2.waitKey(0)
cv2. destroyAllWindows