import numpy as np
import cv2
import utility_functions as uf
from scipy.fft import fft2, ifft2, fftshift, ifftshift

K = np.array([[1346.100595,0,932.1633975],[0,1355.933136,654.8986796],[0,0,1]])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 29
cap = cv2.VideoCapture("1tagvideo.mp4")

# Cube Mapping
output_cube_mapping = cv2.VideoWriter("CubeMapping.avi", fourcc, fps, (1920,1080))

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
    # Compute sides and corners of the cube
    cube_sides=np.array([-(dimensions-1),-(dimensions-1),-(dimensions-1),-(dimensions-1)]).reshape(-1,1)
    corners_of_cube = np.concatenate((sq_pts, cube_sides), axis=1)
    # Compute homography for cube
    homography_of_cube = uf.homography_for_cube(sq_pts,corners_of_cube)
    # Compute projections
    projection_matrix = uf.compute_projection(K, homography_of_cube)
    projected_corners_points_of_cube = uf.project_cube_points(corners_of_cube,projection_matrix)
    # Draw/display the 3D cube over the AR Tag
    full_frame_with_ARTag_and_cube = uf.draw_cube(sq_pts,projected_corners_points_of_cube, frame)
    output_cube_mapping.write(full_frame_with_ARTag_and_cube)
    
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        output_cube_mapping.release()
        break
    
cv2.destroyAllWindows()