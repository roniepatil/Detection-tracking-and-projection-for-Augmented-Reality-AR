import numpy as np
import cv2
import utility_functions as uf
from scipy.fft import fft2, ifft2, fftshift, ifftshift

K = np.array([[1346.100595,0,932.1633975],[0,1355.933136,654.8986796],[0,0,1]])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 29
cap = cv2.VideoCapture("1tagvideo.mp4")
testudo_turtle = cv2.imread('testudo.png')
# Testudo Mapping
output_testudo_mapping = cv2.VideoWriter("TestudoMapping.avi", fourcc, fps, (1920,1080))

while True:
    ret, frame = cap.read()
    if np.any(frame) == None:
        break
    # Detect AR Tag and draw bounding box over the AR tag 
    frame, sq_pts = uf.detect_ARTag(frame)
    frame_with_sq_countours = frame.copy()
    dimensions = 200
    # Compute homography
    H = uf.homography(sq_pts, dimensions)
    H_inverse = np.linalg.inv(H)
    # Warp AR Tag to perspective
    warped_and_cropped = uf.warp_image(H_inverse, frame, dimensions, dimensions)
    # Decode AR Tag to obtain tag ID and orientation
    grid_img_of_tag, binary_tag_ID, orientation_of_tag = uf.decode_ARTag(warped_and_cropped)
    # Orient testudo turtle image so as to match upright direction of AR tag 
    ARTag_based_oriented_testudo_turtle = uf.orient_testudo_turtle(testudo_turtle, orientation_of_tag)
    height = frame.shape[0]
    weight = frame.shape[1]
    testudo_turtle_dimensions = ARTag_based_oriented_testudo_turtle.shape[0]
    # Compute homography
    homography_of_testudo_turtle=uf.homography(sq_pts,testudo_turtle_dimensions)
    # Warp AR Tag and testudo turtle to perspective
    warped_testudo_turtle = uf.warp_image(homography_of_testudo_turtle, ARTag_based_oriented_testudo_turtle,height,weight)
    full_frame_with_ARTag_and_testudo_turtle = frame.copy()
    stencil_frame = cv2.drawContours(full_frame_with_ARTag_and_testudo_turtle,[sq_pts], -1,(0),thickness=-1)
    # Display/overlay the testudo turtle image on AR tag
    full_frame_with_ARTag_and_testudo_turtle = cv2.bitwise_or(warped_testudo_turtle,stencil_frame)
    output_testudo_mapping.write(full_frame_with_ARTag_and_testudo_turtle)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        output_testudo_mapping.release()
        break
    
cv2.destroyAllWindows()