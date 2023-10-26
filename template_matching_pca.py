import cv2
import numpy as np
import imutils as imu
import math

def fit_angel_pca(contours,src):
    data = contours[0][:, 0, :].astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data, mean=None)
    angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
    print("\nEigenvectors:")
    print(eigenvectors[0])
    print("angel:", angel)
    return mean,angel

def fit_angel_pca1(contours):
    data = contours[0][:, 0, :].astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data, mean=None)
    angel = math.atan2(eigenvectors[1][1], eigenvectors[1][0]) * (180 / math.pi)
    print("angel1111:", angel)
    return mean,angel

# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("data/sample-2-4.bmp")
img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
_, threshold = cv2.threshold(blurred, 80, 120, cv2.THRESH_BINARY)
edges_src = cv2.Canny(threshold, 120, 150)
contours, _ = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#  đọc  và tiền xử lý template
sr1= cv2.imread("data/sample-for-tets.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
edges_template = cv2.Canny(blurred, 100, 150)

# init parameter
# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
threshold = 0.0
topleft = [0,0]
res_copy = np.zeros_like(edges_src)
best_angel = 0
edges_src_best = np.zeros_like(edges_src)

# resolve angel problem by pca 
center_cons, angel_cons = fit_angel_pca(contours,sr0)
center_cons1, angel_cons1 = fit_angel_pca1(contours)
# # resolve angel problem
# for i in range(0,360,1):
#     edges_src_copy = imu.rotate(edges_src,i)
#     res = cv2.matchTemplate(edges_src_copy,edges_template, method)
#     # xác dịnh tọa độ và vẽ khung cho template trên ảnh
#     minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
#     if(maxval>=threshold):
#         res_copy = res
#         threshold = maxval
#         print(threshold, ':', i)
#         edges_src_best = edges_src_copy
#         best_angel = i
#         topleft = maxloc

# #     de ve hinh chu nhat thi can biet toa do 2 goc cheo
# bottomright= (topleft[0]+w,topleft[1]+h)

# sr0 = imu.rotate(sr0,best_angel)
# # cv2.imshow("xoay", team)
# cv2.imshow("xoay", edges_src_best)


# cv2.rectangle(sr0,topleft,bottomright,(0,255,255),1)

# cv2.imshow("dectect",sr0)

cv2.waitKey(0)
cv2. destroyAllWindows