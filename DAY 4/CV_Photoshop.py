import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\Aayush\Downloads\blue-circle.png")
cv2.imshow("Photos",img)
cv2.waitKey(0)
img_resize=cv2.resize(img,(400,256))
cv2.imshow("Photos",img_resize)
cv2.waitKey(0)
img_flip=cv2.flip(img,0)
cv2.imshow("Photos",img_flip)
cv2.waitKey(0)
img_crop=img[100:300,200:500]
cv2.imshow("Photos",img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()