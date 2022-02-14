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


# Method to create a dataset on which to cluster our data.
def createDataset(im):
    # Grab the size of the image.
    [N, M, D] = im.shape

    # Seperate the image into it's layered components.
    red = im[:, :, 2]
    green = im[:, :, 1]
    blue = im[:, :, 0]

    # Reshape the vectors to N*M x 1.
    new_shape = (N * M, 1)
    red_v = np.reshape(red, new_shape)
    green_v = np.reshape(green, new_shape)
    blue_v = np.reshape(blue, new_shape)

    # Concatonate the new vectors into a dataset matrix.
    features = np.zeros([N * M, 3])
    features[:, 0] = blue_v[:, 0]
    features[:, 1] = green_v[:, 0]
    features[:, 2] = red_v[:, 0]

    # Return.
    return features


# Method to perform K-means segmentation on a given dataset.
def kMeansCluster(features, centers):
    # Grab the length of the features and the length of the center matrices.
    f_size = features.shape
    q = f_size[0]
    m_size = centers.shape
    k = m_size[0]

    # Instantiate a new cluster matrix which will store the pixels matching the center.
    clusters = {}
    difference = {}
    for i in range(k):
        clusters[i] = None
        difference[i] = None

    # Instantiate a set of new k-mean centers.
    new_centers = centers
    
    # Perform the k-means clustering by looping through the dataset and comparing
    # pixels with k-mean centers.
    iterations = 70
    for iter in range(iterations):
        clusters = {}
        difference = {}
        for j in range(k):
            clusters[j] = None
            difference[j] = None

        # Compute the difference between the features and the centers.
        for j in range(k):
            difference[j] = (np.linalg.norm(features - new_centers[j, :], axis=1))**2

        # Generate a matrix from the calculated norms.
        diff_mat = np.zeros([q, k])
        for j in range(k):
            diff_mat[:, j] = difference[j]

        # Compute the argmin of the rows of the difference matrix.
        indices = np.argmin(diff_mat, axis=1)

        # For the calculated indices, generate an array of the indices for a specified minimum
        # index. Then, using those indices, find the features in those locations.
        for j in range(k):
            index_list = np.argwhere(indices == j)
            clusters[j] = np.array(features[index_list, :])
            if clusters[j].size != 0:
                new_centers[j, :] = np.mean(clusters[j], axis=0)
            else:
                new_centers[j, :] = np.random.randint(0, 255, size=(1, 3))

        print(new_centers)

    # Return the new calculated centers and clusters.
    return [clusters, new_centers, indices]


# Method to map values in the old array to values calculated by k-means clustering.
def mapValues(im, idx, centers):
    # Grab the various dimensions of the given objects.
    [N, M, D] = im.shape
    k_size = centers.shape
    k = k_size[0]

    # Replace the values of the old image with the k-mean centers. For every k, 
    # find the values in the idx array that match the current minimum index. Find the
    # center that matches this index and replace the rows of the output with this value.
    output = np.zeros([N * M, 3])
    for i in range(k):
        index_list = np.argwhere(idx == i)
        if index_list.size != 0:
            output[index_list, :] = centers[i, :]

    # Extract color components.
    red = np.reshape(output[:, 0], (N, M))
    green = np.reshape(output[:, 1], (N, M))
    blue = np.reshape(output[:, 2], (N, M))

    # Generate the image from the extracted color components.
    new_im = np.zeros(im.shape)
    new_im[:, :, 0] = blue
    new_im[:, :, 1] = green
    new_im[:, :, 2] = red

    # Return the image.
    return new_im
    

# Method to generate a k x 3 set of random k-mean centers.
def generateMeans(k):
    # Instantiate a random set of k means (centers).
    return np.random.randint(20, 200, size=(k, 3))


# Read in an image.
file_path = r"white-tower.png"
#A = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
A = cv2.imread(file_path)
size = A.shape

# Generate the features of A.
print('Generating the image features...')
features = createDataset(np.int64(A))
print('Done!')

# Generate a random set of k-mean centers.
k = 7
print('\nGenerating Random k-means...')
k_means = generateMeans(k)
print(k_means)
print('Done!')

# Compute the k-means cluster.
print('\nCalculating k-mean clusters...')
[clusters, centers, idx] = kMeansCluster(features, k_means)
print('Done!')

# Display the k-mean centers.
print('K-mean Centers Produced After Clustering')
print(centers)

# Map the old image values to the new k-mean centers.
print('\nGenerating the new image...')
clustered_image = mapValues(A, idx, centers)
print('Done!')
print('')

# Show the two images.
plt.figure(1)
plt.imshow(cv2.cvtColor(np.uint8(A), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.colorbar()

plt.figure(2)
plt.imshow(np.uint8(clustered_image))
plt.title('Clustered Image')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.colorbar()
plt.show()


