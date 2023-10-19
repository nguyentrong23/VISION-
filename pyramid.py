import cv2

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
# Load the image and template
image = cv2.imread('data/sample-2-5.bmp', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('data/sample_for_template.bmp', cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template, (200, 100))
h, w = template.shape[::]
pyramid = generate_pyramid(image, levels=6)


best_pyramid, max_level, max_loc = template_matching(pyramid, template)
cv2.imshow('best pyramid scale ', best_pyramid)

topleft = max_loc
bottomright= (topleft[0]+w,topleft[1]+h)

cv2.rectangle(best_pyramid,topleft,bottomright,(0,255,255),1)
cv2.imshow('best pyramid scale ', best_pyramid)
cv2.waitKey(0)
cv2.destroyAllWindows()
