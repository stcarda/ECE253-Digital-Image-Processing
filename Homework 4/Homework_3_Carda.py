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
from numpy.core.fromnumeric import ndim
#from scipy import ndimage
import matplotlib.pyplot as plt
import scipy
#---------------------------------
# Suppress scientific notation representation for numpy arrays.
np.set_printoptions(suppress=True)

#------------------------------------------------------------------------------------------------------------
#-----------
# PROBLEM 1
#-----------

# Define a function that accepts as inputs a gratscale image and a threshold and returns an
# image containing the edges of the original.
def compute_canny_edge(im, threshold):
    # Start the smoothing process.
    print('-------------------------------------')
    print('---Smoothing---')

    # Grab the dimensions of the image.
    size = im.shape

    # The first step is to smooth the image using the following gaussian kernel.
    K = (1 / 159) * np.array([[2,  4,  5,  4, 2],
                              [4,  9, 12,  9, 4],
                              [5, 12, 15, 12, 5],
                              [4,  9, 12,  9, 4],
                              [2,  4,  5,  4, 2]])



    # Compute the convolution between the kernel and the padded image.
    print('Beginning the convolution for smoothing...')
    im_smooth = compute_convolution(im, K)
    #im_smooth = ndimage.convolve(im, K)
    #im_smooth = compute_convolution(im, K)

    # Start the gradient calculation process.
    print('-------------------------------------')
    print('---Gradients---')

    # Instantiate the necessary kernels to calculate the respective gradients.
    k_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    k_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

    # Calculate the gradients in the x and y directions.
    print('Computing x-gradient...')
    G_x = compute_convolution(np.float64(im_smooth), k_x)

    print('Computing y-gradient...')
    G_y = compute_convolution(np.float64(im_smooth), k_y)

    # Calculate the magnitude and direction of the gradient.
    G_mag = np.sqrt(G_x**2 + G_y**2)
    G_dir = np.degrees(np.arctan2(G_y, G_x))

    # Start the suppression process.
    print('-------------------------------------')
    print('---NMS---')

    # First, correct the degrees
    print('Correcting the degrees matrix...')
    G_dir_round = np.zeros([size[0], size[1]])
    for r in range(0, size[0]):
        for c in range(0, size[1]):
            G_dir_round[r, c] = round_degrees(G_dir[r, c])

    # Pad the gradient magnitude matrix for NMS checks.
    G_mag_pad = np.pad(G_mag, 1, mode='symmetric')

    # For every pixel in the padded image, determine whether or not to suppress its value or
    # to keep it. 
    print('Suppressing pixels...')
    #G_mag_NMS = np.zeros([size[0], size[1]])
    G_mag_NMS = np.copy(G_mag)
    for r in range(0, size[0]):
        for c in range(0, size[1]):
            # Store the current pixel in a variable.
            pixel = G_mag_pad[r + 1, c + 1]

            # Grab the gradient angle.
            grad_angle = np.radians(G_dir_round[r, c])

            # Compute the row and columns offsets given by the direction matrix.
            direction_y_offset = int(np.round(np.sin(grad_angle)))
            dirrection_x_offset =  int(np.round(np.cos(grad_angle)))

            # Calculate which pixels are neighboring the current pixel based on the offsets.
            G_north = G_mag_pad[r + 1 + direction_y_offset, c + 1 + dirrection_x_offset]
            G_south = G_mag_pad[r + 1 - direction_y_offset, c + 1 - dirrection_x_offset]
            
            # If the current pixel is greater than its neighbors, do not suppress it.
            if pixel > G_north and pixel > G_south:
                G_mag_NMS[r, c] = pixel
            else:
                G_mag_NMS[r, c] = 0

    # Start the thresholding process.
    print('-------------------------------------')
    print('---Thresholding---')
    G_mag_NMS = (255 / np.max(G_mag_NMS)) * G_mag_NMS
    G_threshold = np.where(G_mag_NMS > threshold, 255, 0)

    # Show some of the intermediate steps.
    print('-------------------------------------')
    print('---Displaying---')

    # Show the gradient magnitude image.
    G_mag_img = np.uint8(G_mag * (255 / np.max(G_mag)))
    cv2.imshow('Gradient Magnitude', G_mag_img)
    cv2.imwrite('gradient_magnitude.jpg', G_mag_img)
    cv2.waitKey(0)

    # Show the NMS gradient magnitude image.
    G_mag_NMS_img = np.uint8(G_mag_NMS)
    cv2.imshow('NMS Gradient Magnitude', G_mag_NMS_img)
    cv2.imwrite('nms_gradient_magnitude.jpg', G_mag_NMS_img)
    cv2.waitKey(0)

    # Return the final edged image.
    return np.uint8(G_threshold)


# Function to compute the convolution between an image and a given kernel.
def compute_convolution(im, kernel):
    # Grab the sizes of the given image and kernel.
    size_im = im.shape
    size_k = kernel.shape

    # Grab the offset generated by the kernel assuming the kernel is square with odd dimensionality.
    o = int(np.floor(size_k[0] / 2))

    # Pad the image based on the dimensions of the kernel.
    im_pad = np.pad(im, o, mode='symmetric')

    # Compute the convolution between the kernel and the padded image.
    im_smooth = np.zeros([size_im[0], size_im[1]])
    for r in range(o, size_im[0]):
        for c in range(o, size_im[1]):
            im_smooth[r, c] = np.multiply(kernel, im_pad[(r - o):(r + o + 1), (c - o):(c + o + 1)]).sum()

    # Return the convolved image.
    return im_smooth


# Function to round a given degree value to the nearest 45 degrees.
def round_degrees(degree):
    # Correct the degree value first.
    if degree < 0:
        degree = degree + 360

    # Provide a list of valid degree measurements.
    valid_degrees = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    # Return the corrected degree value.
    return valid_degrees[np.argmin(abs(valid_degrees - degree))]


# Load the geisel.jpg image.
file_path = r"test.jpg"
A = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
size = A.shape

edge = cv2.Canny(A, 100, 255)
cv2.imshow('test', edge)
cv2.waitKey(0)

# Compute the canny edge image.
A_edges = compute_canny_edge(A, 10)

# Show the original image.
cv2.imshow('Original Image', A)
cv2.imwrite('original_geisel.jpg', A)
cv2.waitKey(0)

# Resize the edged image.
cv2.imshow('Canny Edge Image', A_edges)
cv2.imwrite('canny_edge.jpg', A_edges)
cv2.waitKey(0)

"""
#------------------------------------------------------------------------------------------------------------
#-----------
# PROBLEM 1
#-----------
cv2.destroyAllWindows()


# Define a function that will automatically return the log-shifted fourier transform of a
# given image.
def compute_fft(im):
    # First, pad the image to a size of 512 x 512 pixels.
    size = im.shape
    im_pad = np.zeros([512, 512])
    im_pad[0:size[0], 0:size[1]] = im

    # Now, compute the fft of the padded image.
    im_fft = np.fft.fft2(np.uint8(im_pad))

    # Shift the fft.
    im_fft_shift = np.fft.fftshift(im_fft)

    # Take the log of the magnitude of the image.
    im_fft_shift_log = np.log(np.abs(im_fft_shift))

    # Return the image.
    return [np.uint8(im_fft_shift_log), im_fft_shift]


# Function which accepts the parameters n, d0, and center corrdinates u and v to generate the 
# Butterworth notch filter mask of given size.
def generate_butterworth(size, n, D0, U, U_k, V, V_k):
    # Generate an initial mask of all ones.
    mask = np.ones([size[0], size[1]])
    for u in range(0, size[0]):
        for v in range(0, size[0]):
            # Calculate the distances of the current point (u, v) to the specified centers of 
            # the Butterworth filter.
            Dk_pos = np.sqrt((U[u, v] - U_k + 0.01)**2 + (V[u, v] - V_k + 0.01)**2)
            Dk_neg = np.sqrt((U[u, v] + U_k + 0.01)**2 + (V[u, v] + V_k + 0.01)**2)

            # Calculate the two terms in the product.
            term_1 = 1 / (1 + (D0 / Dk_pos)**(2*n))
            term_2 = 1 / (1 + (D0 / Dk_neg)**(2*n))

            # Calculate the product for every center specified.
            mask[u, v] = np.prod(term_1 * term_2)

    # Return the calculated mask.
    return mask


# Define the information necessary for both images.
file_path = [r"Car.tif", r"Street.png"]
u_k_dictionary = {0: [-85, -85, -85, -85],
                  1: [-166, 0]}

v_k_dictionary = {0: [-171, -85, 85, 171],
                  1: [0, 166]}

n_dictionary = {0: 3,
                1: 3}

D_0_dictionary = {0: 15,
                1: 20}

# For both the Car and Street images, filter them with the Butterworth notch filter.
for n in range(0, 2):
    print('Computing information for ' + file_path[n] + '...')

    # Load the specified image.
    A = cv2.imread(file_path[n])
    A = A[:, :, 0]
    image_size = A.shape

    # Show the original image.
    plt.figure(1)
    plt.imshow(A, cmap='gray')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original Image for ' + file_path[n])

    # Compute the fourier transform.
    print('Compute fft...')
    [A_fft_log, A_fft] = compute_fft(A)
    print('Done!')

    # Show the fft of the image.
    plt.figure(2)
    plt.imshow(15 * A_fft_log, cmap='gray')
    plt.colorbar()
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('2D Log-Magnitude DFT Image for ' + file_path[n])

    # Generate the u and v values for the Butterworth filter.
    x_axis = np.linspace(-256,255,512)
    y_axis = np.linspace(-256,255,512)
    [u,v] = np.meshgrid(x_axis,y_axis)

    # Generate the approximate (u, v) coordinates at which the impulses are located.
    u_k = u_k_dictionary[n]
    v_k = v_k_dictionary[n]

    # Generate the butterworth mask for the image.
    print('Generating mask...')
    mask = generate_butterworth([512, 512], n_dictionary[n], D_0_dictionary[n], u, u_k, v, v_k)
    print('Done!')

    # Show the generated mask.
    plt.figure(3)
    plt.imshow(np.uint8(255 * mask), cmap='gray')
    plt.colorbar()
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('Butterworth Notch Filter Mask for ' + file_path[n])

    # Apply the mask to the DFT of the image.
    print('Filtering with mask...')
    A_fft_masked = A_fft * mask

    # Compute the filtered image.
    A_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(A_fft_masked)))
    A_filtered = A_filtered[0:image_size[0], 0:image_size[1]]

    # Threshold the image by 255 such that pixels over this value are reduced to 255.
    # This prevents artifacts in the uint8 version of the image.
    A_filtered = np.where(A_filtered > 255, 255, A_filtered)
    print('Done!')

    # Show the mask over the DFT.
    plt.figure(4)
    plt.imshow(np.uint8(15 * (A_fft_log * mask)), cmap='gray')
    plt.colorbar()
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('Butterworth Notch Filter Mask Overlay for ' + file_path[n])

    # Show the filtered image.
    plt.figure(5)
    plt.imshow(np.uint8(A_filtered), cmap='gray')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Butterworth Filtered Image for ' + file_path[n])
    plt.show()



"""