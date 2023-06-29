import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

cap = cv2.VideoCapture("1tagvideo.mp4")
total_frames = cap.get(7)
cap.set(1,400)
# Read single image frame from video
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Apply DFT on image
dft = fft2(img)
# Shift and spread values across 0 
dft_shift = fftshift(dft)
# Magnitude spectrum of fourier analysis
magnitude_spectrum = np.log(np.abs(dft_shift))
rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)
# Initilize mask
mask = np.ones((rows, cols), np.uint8)
r = 100
# r_out = 200
# r_in = 1
center = [crow, ccol]
x, y = np.ogrid[:rows,:cols]
# Create a mask of radius 100
mask_area = (x - center[0])**2 + (y - center[1])**2 <=r*r
# mask_area = np.logical_or(((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_in ** 2),
#                            ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_out ** 2))
mask[mask_area] = 0
fshift = dft_shift*mask
np.seterr(divide = 'ignore') 
fshift_mask_mag = np.log(np.abs(fshift))
# Apply Inverse DFT
f_ishift = ifftshift(fshift)
img_back = ifft2(f_ishift)
img_back = np.abs(img_back)
final_img = np.array(img_back, dtype='uint8')

plt.imsave("FFT_IFFT_result.jpg",img_back, cmap ='gray')
plt.subplot(2, 2, 1), 
plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), 
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), 
plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), 
plt.imshow(img_back, cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
plt.show()

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()