
import cv2
import numpy as np
import math
import imutils
import time

def fit_angel_pca(contours, src):
    min_area = 40000
    for idx,cnt in enumerate(contours):
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(src, [cnt], -1, (0, 255, 0), 2, cv2.LINE_AA)
            print("index: ",idx,'area: ',cv2.contourArea(cnt) )
            data = contours[idx][:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
            # Tọa độ điểm mean (làm tròn)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            # Kích thước vector
            scale = 100
            # Vector theo eigenvector  chính
            vector1_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
            # Vector theo eigenvector phụ
            vector2_end = (int(mean_point[0] + eigenvectors[1][0] * scale), int(mean_point[1] + eigenvectors[1][1] * scale))
            # Vẽ hình tròn (màu đỏ) tại điểm mean
            cv2.circle(src, mean_point, 5, (0, 0, 255), -1)
            # Vẽ các vector
            cv2.arrowedLine(src, mean_point, vector1_end, (0, 255, 0), 2)
            # cv2.arrowedLine(src, mean_point, vector2_end, (0, 255, 0), 2)
    return mean, angel, src

# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("data/imgSrc/sample-6.bmp")
img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
_, edges_src = cv2.threshold(blurred, 100, 120, cv2.THRESH_BINARY_INV)
# edges_src = cv2.Canny(edges_src, 120, 150)
contours, _ = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # dùng cv2.CHAIN_APPROX_NONE để giữ nguyên tất cả các giá trị contours
cv2.imshow("input for find",edges_src)

#  đọc  và tiền xử lý template
sr1= cv2.imread("data/imgSrc/teamplate_0_degree.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
edges_template = cv2.Canny(blurred, 100, 150)
#  đọc  và tiền xử lý template flip
sr1_flip= cv2.imread("data/imgSrc/teamplate_0_degree.bmp")
img_template_flip = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
edges_template_flip = cv2.Canny(blurred, 100, 150)


# init parameter
# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
topleft = [0,0]
template_copy = np.zeros_like(edges_template)
best_angel = 0
src_copy = np.zeros_like(edges_src)


start_time = time.time()
# resolve angel problem by pca
center_cons, angel_cons, src = fit_angel_pca(contours,sr0)
rotated_template = imutils.rotate(edges_template,-angel_cons)
res = cv2.matchTemplate(edges_src, rotated_template, method)
minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
# cv2.imshow("template_copy",rotated_template )
cv2.imshow("dectect",sr0)
print(angel_cons)
print(maxval)

cv2.waitKey(0)
cv2. destroyAllWindowss