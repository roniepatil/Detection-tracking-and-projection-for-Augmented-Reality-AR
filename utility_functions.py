import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def detect_ARTag(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(img,21,cv2.BORDER_DEFAULT)
    th, dst = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(dst,200,300)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    im = cv2.filter2D(edges, -1, kernel)
    im2 = cv2.filter2D(im, -1, kernel)
    xyz = cv2.bitwise_not(im2)
    corners = cv2.goodFeaturesToTrack(xyz,900,0.25,5)
    corners = np.int0(corners)
    corners = corners.reshape(corners.shape[0],2)
    corner1 = (corners[np.argmin(corners[:,0])])
    corner1 = [int(i) for i in corner1]
    corner2 = (corners[np.argmin(corners[:,1])])
    corner2 = [int(i) for i in corner2]
    corner3 = (corners[np.argmax(corners[:,0])])
    corner3 = [int(i) for i in corner3]
    corner4 = (corners[np.argmax(corners[:,1])])
    corner4 = [int(i) for i in corner4]
    sq_pts = np.array([corner1,corner2,corner3,corner4])
    final = sq_pts.reshape((-1,1,2))
    cv2.polylines(xyz,[final],True,(255,255,255), 100)
    sec_corners = cv2.goodFeaturesToTrack(xyz,900,0.25,5)
    sec_corners = np.int0(sec_corners)
    sec_corners = sec_corners.reshape(sec_corners.shape[0],2)
    sec_corner1 = (sec_corners[np.argmin(sec_corners[:,0])])
    sec_corner1 = [int(i) for i in sec_corner1]
    sec_corner2 = (sec_corners[np.argmin(sec_corners[:,1])])
    sec_corner2 = [int(i) for i in sec_corner2]
    sec_corner3 = (sec_corners[np.argmax(sec_corners[:,0])])
    sec_corner3 = [int(i) for i in sec_corner3]
    sec_corner4 = (sec_corners[np.argmax(sec_corners[:,1])])
    sec_corner4 = [int(i) for i in sec_corner4]
    sec_sq_pts = np.array([sec_corner1,sec_corner2,sec_corner3,sec_corner4])
    sec_final = sec_sq_pts.reshape((-1,1,2))
    cv2.polylines(frame,[sec_final],True,(0,0,255), 10)
    sec_sq_pts.reshape((1,4,2))
    return frame, sec_sq_pts

# Compute homography using SVD
def homography(sq_pts, dimension):
    xp = np.array([0,dimension,dimension,0])
    yp = np.array([0,0,dimension,dimension])
    x = np.array([sq_pts[0][0], sq_pts[1][0], sq_pts[2][0], sq_pts[3][0]])
    y = np.array([sq_pts[0][1], sq_pts[1][1], sq_pts[2][1], sq_pts[3][1]])
    A = np.matrix([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                    [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                    [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                    [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                    [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                    [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]], 
                    [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                    [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1,:]/Vh[-1,-1]
    H = np.reshape(l, (3,3))
    return H

def warp_image(H, image, h, w):
    index_y, index_x = np.indices((h, w), dtype=np.float32)
    index_linearized = np.array([index_x.ravel(), index_y.ravel(), np.ones_like(index_x).ravel()])
    map_index = H.dot(index_linearized)
    np.seterr(divide = 'ignore')
    map_x, map_y = map_index[:-1]/map_index[-1]
    map_x = map_x.reshape(h,w).astype(np.float32)
    map_y = map_y.reshape(h,w).astype(np.float32)
    warped_image = np.zeros((h,w,3),dtype="uint8")
    map_x[map_x>=image.shape[1]] = -1
    map_x[map_x<0] = -1
    map_y[map_y>=image.shape[0]] = -1
    map_y[map_y<0] = -1
    for new_x in range(w):
        for new_y in range(h):
            x = int(map_x[new_y, new_x])
            y = int(map_y[new_y, new_x])
            if x == -1 or y == -1:
                pass
            else:
                warped_image[new_y,new_x] = image[y,x]
    return warped_image

# Finds orientation of the ARTag
def decode_ARTag(frame):
    dimension = frame.shape[0]
    grid_img_of_tag = np.zeros((dimension,dimension,3), np.uint8)
    grid_size = 8
    k = dimension//grid_size
    x = 0
    y = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cell = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            region_of_interest = frame[y:y+k, x:x+k]
            if region_of_interest.mean() > 255//2:
                cell[i][j] = 1
                cv2.rectangle(grid_img_of_tag,(x,y),(x+k,y+k),(255,255,255),-1)
            cv2.rectangle(grid_img_of_tag,(x,y),(x+k,y+k),(127,127,127),1)
            x += k
        x = 0
        y += k
    # ID encryption of Tag is present in the inner 4x4 cells of the 8x8 grid of tag
    # alpha  beta
    # delta  gamma
    alpha = str(int(cell[3][3]))
    beta = str(int(cell[3][4]))
    gamma = str(int(cell[4][4]))
    delta = str(int(cell[4][3]))

    # Show binary value on the appropriate cell
    cv2.putText(grid_img_of_tag,alpha,(3*k+int(k*.3),3*k+int(k*.7)),font,.6,(0,0,255),2)
    cv2.putText(grid_img_of_tag,beta,(4*k+int(k*.3),3*k+int(k*.7)),font,.6,(0,0,255),2)
    cv2.putText(grid_img_of_tag,delta,(3*k+int(k*.3),4*k+int(k*.7)),font,.6,(0,0,255),2)
    cv2.putText(grid_img_of_tag,gamma,(4*k+int(k*.3),4*k+int(k*.7)),font,.6,(0,0,255),2)

    # Determintion of orientation of Tag and to show binary values
    if cell[5,5] == 1:
        orientation_of_tag = 3 
        binary_tag_ID = delta+gamma+beta+alpha
        center = (5*k+(k//2),5*k+(k//2))
        cv2.circle(grid_img_of_tag,center,k//4,(0,117,22),-1)
    elif cell[2,5] == 1: 
        orientation_of_tag = 2 
        binary_tag_ID = gamma+beta+alpha+delta
        center = (5*k+(k//2),2*k+(k//2))
        cv2.circle(grid_img_of_tag,center,k//4,(0,117,22),-1)
    elif cell[2,2] == 1: 
        orientation_of_tag = 1 
        binary_tag_ID = beta+alpha+delta+gamma
        center = (2*k+(k//2),2*k+(k//2))
        cv2.circle(grid_img_of_tag,center,k//4,(0,117,22),-1)
    elif cell[5,2] == 1: 
        orientation_of_tag = 0 
        binary_tag_ID = alpha+delta+gamma+beta
        center = (2*k+(k//2),5*k+(k//2))
        cv2.circle(grid_img_of_tag,center,k//4,(0,117,22),-1)
    else:
        orientation_of_tag = 0
        binary_tag_ID = '0000'
    return grid_img_of_tag, binary_tag_ID, orientation_of_tag

def orient_testudo_turtle(image, orientation):
    # Orient the direction of testudo turtle
    if orientation == 0:
        ARTag_based_oriented_testudo_turtle = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 2:
        ARTag_based_oriented_testudo_turtle = cv2.rotate(image,cv2.ROTATE_180)
    elif orientation == 1:
        ARTag_based_oriented_testudo_turtle = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        ARTag_based_oriented_testudo_turtle = image
    return ARTag_based_oriented_testudo_turtle

def homography_for_cube(AR_corners_points, cube_corners):
    # Define the eight points to compute the homography matrix
    x=[]
    y=[]
    xp=[]
    yp=[]
    # Convert corners into x and y coordinates
    for point in AR_corners_points:
        x.append(point[0])
        y.append(point[1])   
    for point in cube_corners:
        xp.append(point[0])
        yp.append(point[1])
    # Make A an 8x9 matrix
    # 9 columns
    n = 9 
    # 8 rows
    m = 8 
    A = np.empty([m, n])
    #A matrix is:
    # Even rows (0,2,4,6): [[-x, -y, -1,0,0,0, x*x', y*x', x'],
    # Odd rows (1,3,5,7): [0,0,0, -x, -y, -1, x*y', y*y', y']]
    val = 0
    for row in range(0,m):
        #Even rows
        if (row%2) == 0: 
            A[row,0] = -x[val]
            A[row,1] = -y[val]
            A[row,2] = -1
            A[row,3] = 0
            A[row,4] = 0
            A[row,5] = 0
            A[row,6] = x[val]*xp[val]
            A[row,7] = y[val]*xp[val]
            A[row,8] = xp[val]
        #odd rows
        else: 
            A[row,0] = 0
            A[row,1] = 0
            A[row,2] = 0
            A[row,3] = -x[val]
            A[row,4] = -y[val]
            A[row,5] = -1
            A[row,6] = x[val]*yp[val]
            A[row,7] = y[val]*yp[val]
            A[row,8] = yp[val]
            val += 1
    # Conduct SVD to get V
    U,S,V = np.linalg.svd(A)
    # Find the eigenvector column of V that corresponds to smallest value (last column)
    x = V[-1]
    # Reshape x into 3x3 matrix to have H
    H = np.reshape(x,[3,3])
    return H

def match_cube_corners_to_tag_corners(AR_corners_points,cube_corners_points):
    connecting_lines = []
    for i in range(len(AR_corners_points)):
        if i==3: 
            p1 = AR_corners_points[i]
            p2 = AR_corners_points[0]
            p3 = cube_corners_points[0]
            p4 = cube_corners_points[i]
        else:
            p1 = AR_corners_points[i]
            p2 = AR_corners_points[i+1]
            p3 = cube_corners_points[i+1]
            p4 = cube_corners_points[i]
        # Array of connecting lines
        connecting_lines.append(np.array([p1,p2,p3,p4], dtype=np.int32))
        # Append AR corners and cube corners
    connecting_lines.append(np.array([AR_corners_points[0],AR_corners_points[1],AR_corners_points[2],AR_corners_points[3]], dtype=np.int32))
    connecting_lines.append(np.array([cube_corners_points[0],cube_corners_points[1],cube_corners_points[2],cube_corners_points[3]], dtype=np.int32))
    return connecting_lines

# Draw cube based on scaled coordinates of cube points
def draw_cube(base, top, frame):
    # Lines connecting top and base of cube
    sides= match_cube_corners_to_tag_corners(base, top)
    for s in sides:
        cv2.drawContours(frame,[s],0,(255,0,0),5)   
    # Draw square at top of cube and around AR tag (base of cube)
    for i in range (4):
        if i==3:
            cv2.line(frame,tuple(top[i]),tuple(top[0]),(0,255,255),5)
        else:
            cv2.line(frame,tuple(top[i]),tuple(top[i+1]),(0,255,255),5)
    return frame

def compute_projection(K, H):
    # Calculate rotation and translation
    H=H*(-1)
    B_tilde=np.dot(np.linalg.inv(K),H)
    # Ensure positive B_tilde
    if np.linalg.norm(B_tilde)>0:
        B=1*B_tilde
    else:
        B=-1*B_tilde
    b_1=B[:,0]
    b_2=B[:,1]
    b_3=B[:,2]
    # Lambda is average length of the first two columns of B
    lambda_=np.sqrt(np.linalg.norm(b_1,2)* np.linalg.norm(b_2,2))
    # Normalize vectors
    rot_1=b_1/ lambda_
    rot_2=b_2/ lambda_
    trans=b_3/ lambda_
    c=rot_1+rot_2
    p=np.cross(rot_1,rot_2)
    d=np.cross(c,p)
    # Orthogonal basis
    rot_1=np.dot(c/ np.linalg.norm(c,2) + d / np.linalg.norm(d,2), 1 / np.sqrt(2))
    rot_2=np.dot(c/ np.linalg.norm(c,2) - d / np.linalg.norm(d,2), 1 / np.sqrt(2))
    rot_3=np.cross(rot_1,rot_2)
    R_t=np.stack((rot_1,rot_2,rot_3,trans)).T
    # P = K * [R | t]
    projection_matrix=np.dot(K,R_t)
    return projection_matrix


def project_cube_points(corners_of_cube, projection_matrix):
    projected_corners_points_of_cube=[]
    # Separate corners of AR tag into x, y, z coordinates
    x = []
    y = []
    z = []
    for point in corners_of_cube:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2]) # Dimensions of cube in z direction
    X_w=np.stack((np.array(x),np.array(y),np.array(z),np.ones(len(x))))
    # Use Projection Matrix to shift back to camera frame
    sX_c2=np.dot(projection_matrix,X_w)
    # Camera frame homography
    X_c2=sX_c2/sX_c2[2,:]
    for i in range(4):
        projected_corners_points_of_cube.append([int(X_c2[0][i]),int(X_c2[1][i])])  
    return projected_corners_points_of_cube