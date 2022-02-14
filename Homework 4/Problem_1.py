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
# Cross Correlation Filter.

# First, read in the first birds image and the template we will use to identify the features
# of interest in the image.
print('Loading images...')
birds = cv2.cvtColor(cv2.imread('birds1.jpeg'), cv2.COLOR_BGR2GRAY)
birds_2 = cv2.cvtColor(cv2.imread('birds2.jpeg'), cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(cv2.imread('template.jpeg'), cv2.COLOR_BGR2GRAY)
print('Done!')

# Define a method for showing images which will reduce the lines of code needed
# to show images.
def imshow(im, title, fig_num, xlabel, ylabel):
    plt.figure(fig_num)
    plt.imshow(im, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Show the original image and template.
imshow(birds, 'Original Birds 1 Image', 1, 'Columns', 'Rows')
imshow(birds_2, 'Original Birds 2 Image', 2, 'Columns', 'Rows')
imshow(template, 'Template for Matching', 3, 'Columns', 'Rows')

# Compute convolution between the birds image and the template.
print('Cross correlating...')
matching = ndi.convolve(np.float64(birds), np.float64(template))
imshow(matching, 'Birds 1 Image Convolved with Template', 4, 'Columns', 'Rows')
print('Done!')
plt.show()

#--------------------------------------------------------------------------------------------------
# Normalized Cross Correlation Filter.

# Define a function that accepts an image and a template and computes the normalized
# cross-correlated image.
def normalized_cc(im, template):
    # Grab the sizes of the image and template.
    im_size = im.shape
    t_size = template.shape

    # Pad the image with the size of the template.
    im_pad = np.pad(im, (int(t_size[0] / 2) + 1, int(t_size[1] / 2)), mode='symmetric')

    # Instantiate an output image.
    output = np.zeros(im_size)

    # Calculate the mean of the template and the term of the normalization contributed by the 
    # template.
    t_avg = np.mean(template)
    t_offset = template - t_avg
    norm_coeff_t = np.sum((template - t_avg)**2)
    for r in range(0, im_size[0]):
        for c in range(0, im_size[1]):
            # Grab the image windowed by the template.
            im_window = im_pad[r:r+t_size[0], c:c+t_size[1]]

            # Compute the average of the image in the current window.
            im_avg = np.mean(im_window)

            # Using the calculated means, compute the cross correlation of the image in the current
            # window.
            im_offset = (im_window - im_avg)
            cross_corr = np.sum(im_offset * t_offset)

            # Compute the normalized coefficients of the image and the template at the current
            # window. 
            norm_coeff_im = np.sum(im_offset**2)
            norm_coeff = norm_coeff_t * norm_coeff_im

            # Comptue the normalized cross-correlation.
            norm_cc = cross_corr / np.sqrt(norm_coeff)
            output[r, c] = norm_cc

    # Return.
    print('Done!')
    return output


# Accepts an image and a point in the image and places a box of given dimension around that
# point in the image.
def identify(im, point, dim):
    # Define an output image which will be boxed.
    size = im.shape
    im_boxed = np.zeros([size[0], size[1], 3])
    im_boxed[:, :, 0] = np.copy(im)
    im_boxed[:, :, 1] = np.copy(im)
    im_boxed[:, :, 2] = np.copy(im)

    # Calculate the height and width of the box.
    width = int(dim[1] / 2)
    height = int(dim[0] / 2)

    # Create points.
    point_1 = (point[1] - width, point[0] - height)
    point_2 = (point[1] + width, point[0] + height)

    # Draw the rectangle.
    cv2.rectangle(im_boxed, point_1, point_2, color=(255, 0, 0), thickness=3)

    # Return.
    return np.uint8(im_boxed)

# Compute the normalized cross correlation on the first birds image.
print('Computing normalized cross correlation for birds 1...')
birds_norm_cc = normalized_cc(np.float64(birds), np.float64(template))
imshow(birds_norm_cc, 'Birds Normalized Cross Correlation', 1, 'Columns', 'Rows')

# Place a box around the point of strongest intensity.
point = np.unravel_index(np.argmax(birds_norm_cc), birds_norm_cc.shape)
print(point)
matched_image = identify(birds, point, template.shape)
imshow(matched_image, 'Birds 1 with Identified Template Match', 2, 'Columns', 'Rows')

# Compute the normalized cross correlation on the second birds image.
print('Computing normalized cross correlation for birds 2...')
birds_norm_cc = normalized_cc(np.float64(birds_2), np.float64(template))
imshow(birds_norm_cc, 'Birds 2 Normalized Cross Correlation', 3, 'Columns', 'Rows')

# Place a box around the point of strongest intensity.
point = np.unravel_index(np.argmax(birds_norm_cc), birds_norm_cc.shape)
print(point)
matched_image = identify(birds_2, point, template.shape)
imshow(matched_image, 'Birds 2 with Identified Template Match', 4, 'Columns', 'Rows')
plt.show()




