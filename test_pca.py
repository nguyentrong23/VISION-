import math
import cv2
import numpy as np
import math
import imutils
import time

def fit_angel_pca(contours, src):
    data = contours[0][:, 0, :].astype(np.float32)
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
sr0 = cv2.imread("data/imgSrc/sample-1-2.bmp")
img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
_, threshold = cv2.threshold(blurred, 80, 120, cv2.THRESH_BINARY)
edges_src = cv2.Canny(threshold, 120, 150)
contours, _ = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#  đọc  và tiền xử lý template
sr1= cv2.imread("data/imgSrc/teamplate_0_degree.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
edges_template = cv2.Canny(blurred, 100, 150)

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

low_angel = float(-angel_cons - 4)
high_angel = float(-angel_cons + 4)
threshold = 0.2
i = low_angel
while i< high_angel:
       rotated_template = imutils.rotate(edges_template,i)
       res = cv2.matchTemplate(edges_src,rotated_template, method)
       # xác dịnh tọa độ và vẽ khung cho template trên ảnh
       minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
       if(maxval>=threshold):
           threshold = maxval
           template_copy = rotated_template
           best_angel = i
           topleft = maxloc
       i += 0.5
print(threshold, ':', best_angel)
bottomright= (topleft[0]+w,topleft[1]+h)
cv2.rectangle(sr0,topleft,bottomright,(0,255,255),1)
cv2.imshow("template_copy",template_copy)
cv2.imshow("dectect",sr0)
end_time = time.time()


execution_time = end_time - start_time
print(f"Thời gian chạy: {execution_time} giây")
cv2.waitKey(0)
cv2. destroyAllWindows