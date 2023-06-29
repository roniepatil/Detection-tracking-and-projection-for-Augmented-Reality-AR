import numpy as np
import cv2

img = cv2.imread("cityscape1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Perform canny edge detection
edges = cv2.Canny(img,100,100)
cv2.imwrite("city100.png",edges)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()