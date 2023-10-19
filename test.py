import cv2
import numpy as np
import imutils as imu


def generate_pyramid(image, levels):
    pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def template_matching(pyramid, template):
    max=0
    max_l = (0,0)
    max_level = 0
    for level, img in enumerate(pyramid):
        if img.shape[0] >= template.shape[0] and img.shape[1] >= template.shape[1]:
            match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            minval, maxval, minloc, maxloc = cv2.minMaxLoc(match)
            if(maxval>=max):
                max = maxval
                max_l = maxloc
                max_level = level
                print(max, ':', level)

    return pyramid[max_level], max_level, max_l


# doc anh va template
sr0 = cv2.imread("data/sample-2-3.bmp")
img_src = cv2.cvtColor(sr0,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3,3), 0)
edges_src = cv2.Canny(blurred, 100, 150)


sr1= cv2.imread("data/sample_for_template.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
edges_template = cv2.Canny(blurred, 100, 150)

pyramid = generate_pyramid(edges_src, levels=6)
best_pyramid, max_level, max_loc = template_matching(pyramid, edges_template)
cv2.imshow('best pyramid scale ', best_pyramid)



# # list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
# method = eval("cv2.TM_CCOEFF_NORMED")
# h, w = img_template.shape[::]# căn chỉnh lại so với góc quay
# threshold = 0.0
# topleft = [0,0]
# res_copy = np.zeros_like(edges_src)
# best_angel = 0
# edges_src_best = np.zeros_like(edges_src)
# # resolve angel problem
# for i in range(0,361,1):
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
#
# #     de ve hinh chu nhat thi can biet toa do 2 goc cheo
# bottomright= (topleft[0]+w,topleft[1]+h)
#
# sr0 = imu.rotate(sr0,best_angel)
# # cv2.imshow("xoay", team)
# cv2.imshow("xoay", edges_src_best)
#
#
# cv2.rectangle(sr0,topleft,bottomright,(0,255,255),1)
#
# cv2.imshow("dectect",sr0)

cv2.waitKey(0)
cv2. destroyAllWindows