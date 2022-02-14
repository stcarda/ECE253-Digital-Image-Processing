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

#--------------------------------------------------------------------------------------------------
# HOUGH TRANSFORM ON TEST IMAGE.

# Method which accepts an image and computes the Hough transform.
def HT(im):
    # Maximum angle for the transform.
    t_max = 90

    # Generate all possible theta values.
    theta = np.int64(np.round(np.linspace(-t_max, t_max, 2 * t_max + 1)))

    # Calculate the maximum distance between opposite corners of the image.
    size = im.shape
    D = np.int64(np.floor(np.sqrt((size[0]**2) + (size[1]**2))))
    rho = np.int64(np.round(np.linspace(-D, D, 2 * D + 1)))

    # Instantiate an output image.
    output = np.zeros([len(rho), len(theta)])

    for r in range(0, size[0]):
        for c in range(0, size[1]):
            if im[r, c] >= 1:
                for t in theta:
                    p = (im[r, c] * c * np.cos(np.deg2rad(t))) + (im[r, c] * r * np.sin(np.deg2rad(t)))
                    p = np.int64(p + np.max(rho))
                    output[p, t + t_max] += 1
        if (r % 50) == 0:
            print('Progress: ' + str(round(100 * r / size[0], 2)) + '%...')

    # Return the transform and the established theta and rho values.
    return [output, rho, theta]

# Method which accepts an image and a list of rho and theta parameters and draws the corresponding
# x-y lines on the image. Parameters = [list of theta, list of rho]
def draw_lines(params, im, width, round):
    # Create a new image which will overlay the lines on top of the image.
    size = im.shape
    output = np.zeros([size[0], size[1], 3])
    output[:, :, 0] = im
    output[:, :, 1] = im
    output[:, :, 2] = im

    # For every parameter combination, draw the corresponding line.
    for combo in params:
        # Grab theta and rho
        theta = combo[0]
        rho = combo[1]

        # Compute the new line parameters.
        if theta == 0:
            theta = 0.01
        elif theta == 90:
            theta = 89.99
        elif theta == -90:
            theta = -89.99

        b = rho / (np.sin(np.deg2rad(theta)))
        if round is True:
            m = np.round(-1 / (np.tan(np.deg2rad(theta))))
        else:
            m = -1 / (np.tan(np.deg2rad(theta)))

        # Compute the two points needed to generate the line
        point_1 = ((-999, int(np.round((-999 * m) + b))))
        point_2 = ((999, int(np.round((999 * m) + b))))

        # Draw the line onto the image.
        cv2.line(output, point_1, point_2, (255, 0, 0), width, lineType=cv2.LINE_AA)

    # Return the image.
    return output


# Method which accepts a thresholded transform and the respective rho and theta values and
# returns the theta-row indices of the transformed image.
def get_params(im, rho, theta):
    # Return the indices at which the transform is nonzero.
    list = np.transpose(np.nonzero(im))

    # Augment the list by the theta values by the maximum theta value.
    list[:, 1] = list[:, 1] - np.max(theta)
    list[:, 0] = list[:, 0] - np.max(rho)

    # Return the flipped array so that theta is an index [~, 0] and rho is at [~, 1]
    list_flip = np.flip(list, 1)
    return list_flip


#--------------------------------------------------------------------------------------------------
# HOUGH TRANSFORM ON TEST IMAGE.
# First, generate an 11 x 11 image with 5 1's located in the center and in the four corners.
test_im = np.zeros([11, 11])
size = test_im.shape
points = [[0, 0], [0, size[1] - 1], [int(size[0] / 2), int(size[1] / 2)], [size[0] - 1, 0], [size[0] - 1, size[1] - 1]]
for point in points:
    test_im[point[0], point[1]] = 1

# Calculate the Hough transform of the test image.
[transform, rho, theta] = HT(test_im)

# Threshold the transformed image and look for intersections greater than 2.
transform_thresh = np.where(transform > 2, transform, 0)

# Draw the lines specified by the transform.
params = get_params(transform_thresh, rho, theta)
print(params)
test_lines = draw_lines(params, test_im, 1, True)

plt.figure(1)
plt.imshow(test_im, cmap='gray')
plt.title('Original Test Image')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()

plt.figure(2)
plt.imshow(transform, extent=[theta[0], theta[len(theta) - 1], rho[len(rho) - 1], rho[0]], cmap='gray')
plt.title('Hough Transform on Test Image')
plt.xlabel('theta (degrees)')
plt.ylabel('p')
plt.colorbar()

plt.figure(3)
plt.plot(3, 5, 1, 9, color='white', linewidth=1)
plt.imshow(test_lines, cmap='gray')
plt.title('Test Image with Lines')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()

#--------------------------------------------------------------------------------------------------
# HOUGH TRANSFORM ON LANE IMAGE III

# Read in the lane image.
lane = cv2.imread('lane.png')
thresh = 240
binary_lane = cv2.Canny(lane[:, :, 0], 175, thresh, apertureSize=3, L2gradient=True) / 255


plt.figure(4)
plt.imshow(lane, cmap='gray')
plt.title('Original Lane Image')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()

plt.figure(5)
plt.imshow(binary_lane, cmap='gray')
plt.title('Original Lane Image')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()

# Compute the Hough transform of the edged image.
[transform, rho, theta] = HT(np.uint64(binary_lane))

# Threshold the transformed image and look for intersections greater than 2.
transform_thresh = np.where(transform > (0.75 * np.max(transform)), transform, 0)

# Draw the lines specified by the transform.
params = get_params(transform_thresh, rho, theta)
print('Parameters for analysis:')
print(params)
print('Drawing Lines...')
lane_lines = draw_lines(params, lane[:, :, 0], 3, False)

plt.figure(4)
plt.imshow(lane, cmap='gray')
plt.title('Original Lane Image')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()

plt.figure(5)
plt.imshow(binary_lane, cmap='gray')
plt.title('Edged Lane Image')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()


plt.figure(6, figsize=(10, 20))
plt.imshow(transform, extent=[theta[0], theta[len(theta) - 1], rho[len(rho) - 1], rho[0]], aspect='auto', cmap='gray')
plt.title('Hough Transform on Lane Image')
plt.xlabel('theta (degrees)')
plt.ylabel('p')
plt.colorbar()

plt.figure(8)
plt.imshow(np.uint8(lane_lines), cmap='gray')
plt.title('Lane Image with Lines')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()


#--------------------------------------------------------------------------------------------------
# HOUGH TRANSFORM ON LANE IMAGE IV

# Set a theta range for which to preserve lane edges.
theta_n_low = -57 + 90
theta_n_high = -50 + 90
theta_p_low = 50 + 90
theta_p_high = 57 + 90

# Threshold the thresholded image again, but only preserving the ranges of theta specified.
t_size = transform_thresh.shape
theta_thresh = np.zeros(transform_thresh.shape)
for i in range(t_size[1]):
    if (i > theta_n_low and i < theta_n_high) or (i > theta_p_low and i < theta_p_high):
        theta_thresh[:, i] = transform_thresh[:, i]


# Draw the lines specified by the transform.
params = get_params(theta_thresh, rho, theta)
print('Parameters for analysis:')
print(params)
print('Drawing Lines...')
new_lane_lines = draw_lines(params, lane[:, :, 0], 3, False)

# Show the new lined image.
plt.figure(1)
plt.imshow(np.uint8(new_lane_lines), cmap='gray')
plt.title('New Lane Image with Lines')
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.show()



