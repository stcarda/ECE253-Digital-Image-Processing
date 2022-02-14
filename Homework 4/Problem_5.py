#-------------------------------------------------
# Sean Carda
# ECE 253 - Image Processing
# Dr. Mohan Trivedi
# Homework 3
#-------------------------------------------------

#---------------------------------
# Global Imports.
import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
#---------------------------------
# Suppress scientific notation representation for numpy arrays.
np.set_printoptions(suppress=True)

# -------------------------------------------------------------------------------------------
# This filter will load an image, perform canny edge operation on it, and replace the white
# pixels with colored pixels.
#--------------------------------------------------------------------------------------------
file_path = 'lane.png'
A = cv2.imread(file_path)

# Compute the canny edge of the image.
A_edge = 255 * cv2.Canny(A, 100, 200, L2gradient=True)
edge_size = A_edge.shape

# Layer the edged image.
output = np.zeros([edge_size[0], edge_size[1], 3])
output[:, :, 0] = A_edge
output[:, :, 1] = A_edge
output[:, :, 2] = A_edge

# For every columns, replace it with a colored pixel column.
for i in range(edge_size[1]):
    new_pixel = np.random.randint(1, 255, size=(1, 3))
    output[:, i, :] = np.where(output[:, i] > 0, new_pixel, 0)

# Show the image.
plt.figure(1)
plt.imshow(np.uint8(output))
plt.title('Colorful Lanes!')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.colorbar()
plt.show()








