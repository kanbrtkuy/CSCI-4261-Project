import numpy as np
import matplotlib.pyplot as plt
 
img = plt.imread('image-1.jpg')
img_row = img.shape[0]
img_col = img.shape[1]
 
def kmean(img, iteration, k):
    img = img.reshape(-1,3) # Reshape the dimension
    img_reshape = np.column_stack((img, np.ones(img_row*img_col))) # Add a column
 
    # Randomly choose k pixels as the initial cluster center
    cluster_orientation = np.random.choice(img_row*img_col, k, replace=False) # Generate k index coordinates, that is, the positions of k centers
    cluster_center = img_reshape[cluster_orientation, :] # Find the RGB pixel value of the corresponding clustering center according to the index coordinates
 
    # Iteration
    distance = [[] for i in range(k)] # Generate a List, where each element is a column vector that stores the distances of all pixels from center j
    for i in range(iteration):
        # Calculate the color distance between all pixels and the clustering center j
        for j in range(k):
            distance[j] = np.sqrt(np.sum(np.square(img_reshape - cluster_center[j]), axis=1)) # Sum of a row
 
        # In the color distance between the current pixel and k centers, find the smallest center and update the label of all pixels of the image
        # Returns the index corresponding to the smallest value in a column, the range is [0, 4], it refers to different label
        orientation_min_dist = np.argmin(np.array(distance), axis=0)   # The samllest value in a column
        img_reshape[:, 3] = orientation_min_dist # Assigns the returned index column vector to the fourth dimension, which is the third column that holds the label
        # Update the jth cluster center
        for j in range(k):
            # Calculate the mean
            one_cluster = img_reshape[img_reshape[:, 3] == j] 
            cluster_center[j] = np.mean(one_cluster, axis=0)
 
    return img_reshape
 
if __name__ == '__main__':

    plt.subplot(3,3,1), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 5)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,2), plt.imshow(labels_img)
    plt.title('K=5'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 6)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,3), plt.imshow(labels_img)
    plt.title('K=6'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 7)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,4), plt.imshow(labels_img)
    plt.title('K=7'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 8)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,5), plt.imshow(labels_img)
    plt.title('K=8'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 9)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,6), plt.imshow(labels_img)
    plt.title('K=9'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 10)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,7), plt.imshow(labels_img)
    plt.title('K=10'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 11)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,8), plt.imshow(labels_img)
    plt.title('K=11'), plt.xticks([]), plt.yticks([])
    labels_vector = kmean(img, 1, 12)
    labels_img = labels_vector[:,3].reshape(img_row, img_col)
    plt.subplot(3,3,9), plt.imshow(labels_img)
    plt.title('K=12'), plt.xticks([]), plt.yticks([])

    plt.show()
