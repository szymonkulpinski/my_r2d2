import numpy as np
def PIL2CV(image):
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


scaled_cv = PIL2CV(scaled_image)
scaled_cv_dist = PIL2CV(scaled_and_distorted_image['img'])


import cv2
import numpy as np
def PIL2CV(image):
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image
cv2.imshow('scaled_image', PIL2CV(scaled_image))
cv2.waitKey(0)
cv2.imshow('scaled_and_distorted_image', PIL2CV(scaled_and_distorted_image['img']))
cv2.waitKey(0)
cv2.destroyAllWindows()


numpy.array(pil_image)

# pil to opencv
pil_image = PIL.Image.open('Image.jpg').convert('RGB')
def PIL2CV(image):
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image