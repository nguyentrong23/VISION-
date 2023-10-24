import cv2
import numpy as np
import imutils as imu

def generate_pyramid(image, levels):
    pyramid = [image]
    for i in range(levels):
        # Giảm kích thước ảnh theo tỷ lệ 0.5
        smaller_image = cv2.resize(image, (0, 0), fx=0.9, fy=0.9)
        pyramid.append(smaller_image)
        image = smaller_image
    return pyramid


# doc anh va template
sr0 = cv2.imread("data/sample-2-4.bmp")
img_src = cv2.cvtColor(sr0,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3,3), 0)
edges_src = cv2.Canny(blurred, 100, 150)

sr1= cv2.imread("data/sample-for-tets.bmp")
edges_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(edges_template, (3,3), 0)
edges_template = cv2.Canny(blurred, 100, 150)
edges_template = cv2.resize(edges_template, (54, 54))
cv2.imshow("dectect",edges_template)

# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = edges_template.shape[::]# căn chỉnh lại so với góc quay
threshold = 0.0
topleft = [0,0]
res_copy = np.zeros_like(edges_src)
best_angel = 0
edges_src_best = np.zeros_like(edges_src)


max_val_pyramid = 0
pyramid = generate_pyramid(edges_src, levels=10)
for level, img in enumerate(pyramid):
    print(img.shape[::])
    if img.shape[0] >= edges_template.shape[0] and img.shape[1] >= edges_template.shape[1]:
        match = cv2.matchTemplate(img,edges_template, method)
        _, maxval, _, maxloc = cv2.minMaxLoc(match)
        print(maxval)
        if (maxval > max_val_pyramid):
            max_val_pyramid = maxval
            max_level = level
            edges_src_best = pyramid[max_level]
cv2.imshow("src fir template",edges_src_best)

# # resolve angel problem
# for i in range(0,361,1):
#     edges_template_copy = imu.rotate(edges_template,i)
#     res = cv2.matchTemplate(edges_src_best,edges_template_copy, method)
#     # xác dịnh tọa độ và vẽ khung cho template trên ảnh
#     minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
#     if(maxval>=threshold):
#         res_copy = res
#         threshold = maxval
#         print(threshold, ':', i)
#         edges_template_best = edges_template_copy
#         best_angel = i
#         topleft = maxloc
#
# #  de ve hinh chu nhat thi can biet toa do 2 goc cheo
# bottomright= (topleft[0]+w,topleft[1]+h)
#
# # cv2.imshow("xoay", team)
# cv2.imshow("xoay", edges_template_best)
#
#
# cv2.rectangle(edges_src_best,topleft,bottomright,(0,255,255),1)
#
# cv2.imshow("dectect",edges_src_best)

cv2.waitKey(0)
cv2. destroyAllWindows