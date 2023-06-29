import numpy as np
import cv2
from matplotlib import pyplot as plt
import utility_functions as uf

cap = cv2.VideoCapture("1tagvideo.mp4")
total_frames = cap.get(7)
cap.set(1,139)
ret, frame = cap.read()
directions = {
    0 : 'Right',
    1 : 'Up',
    2 : 'Down',
    3 : 'Left'
}
# Detect AR Tag and draw bounding box over the AR tag 
frame, sq_pts = uf.detect_ARTag(frame)
dimensions = 200
# Compute homography
H = uf.homography(sq_pts, dimensions)
H_inverse = np.linalg.inv(H)
# Warp AR Tag to perspective
warped_and_cropped = uf.warp_image(H_inverse, frame, dimensions, dimensions)
# Decode AR Tag to obtain tag ID and orientation
grid_img_of_tag, binary_tag_ID, orientation_of_tag = uf.decode_ARTag(warped_and_cropped)
cv2.putText(warped_and_cropped, directions[orientation_of_tag]+", "+binary_tag_ID+", "+str(int(binary_tag_ID,2)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)
cv2.imshow("1b_frame",frame)
cv2.imshow("1b_warped_and_cropped", warped_and_cropped)
cv2.imshow("1b_grid_img_of_tag", grid_img_of_tag)
print(directions[orientation_of_tag])
print(binary_tag_ID)
print(int(binary_tag_ID,2))

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()