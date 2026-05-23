import cv2
import numpy as np
import random


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.models import *

def k_means(img: ELMImage, k: int, select: int):
    rgb_arr = img.rgb_arr

    centroids = []

    r, g, b = cv2.split(rgb_arr)

    r = np.array(r.flatten())
    g = np.array(g.flatten())
    b = np.array(b.flatten())

    

    # idea for centroid key is the # of the centroid, value coresponds to the coordiante
    if select == 1:
        centroids = generate_rgb_centroids(k)
    elif select == 0:
        centroids = generate_hsv_centroids(k)
    
    pixels = len(r)

    # exit conditions
    # i.e. exit when max iterations reached on the centroid stops moving
    max_iter = 8
    moved = True

    iter = 0

    while moved and iter <= max_iter:
        iter += 1

        # initialize an assignment array of size pixels
        # each indice represents a pixel and contains the closest distance to a centroid
        assignment = [0]*pixels

        # for each pixel
        for i in range(pixels):

            # create distance array size of number of k-means
            # basically we calculate the distance for a given pixel, determine which one it is the min to
            # and then assign it to that centroid
            distances = [0]*k
            for j in range(k):
                distances[j] = distance(centroids[j], [r[i], g[i], b[i]])

            # find the smallest distance, corresponding to a centroid k
            # closest is an indice corresponding to closest k centroid
            closest = np.argmin(distances)


            assignment[i] = closest

        # use prev_centroid to compare with curr centroid to see if centroid have moved or not
        prev_centroids = centroids.copy()
        
        # calculate the mean for each cluster
        for i in range(k):

            # get a list of pixel indices that correspond to the i-th k-mean
            ind = [j for j in range(pixels) if assignment[j] == i]

            # calculate the new mean for each cluster
            # i.e. we find the mean for each channel by averaging all the relevant pixels' channels
            if len(ind) != 0:
                centroids[i][0] = np.mean(r[ind])
                centroids[i][1] = np.mean(g[ind])
                centroids[i][2] = np.mean(b[ind])
            
            # if cluster empty then must mean that color is not part of image
            else:
                centroids[i][0] = 0
                centroids[i][1] = 0
                centroids[i][2] = 0

        # check if the centroids have moved after each iteration
        if np.array_equal(centroids, prev_centroids):
            moved =  False
    
    r_copy = np.array(r.copy())
    g_copy = np.array(g.copy())
    b_copy = np.array(b.copy())

    # update pixels to match their corresponding cluster
    for i in range(k):
        ind = [j for j in range(pixels) if assignment[j] == i]

        r_copy[ind] = centroids[i][0]
        g_copy[ind] = centroids[i][1]
        b_copy[ind] = centroids[i][2]
    
    img2 = np.array([r_copy, g_copy, b_copy])

    img2 = img2.transpose()

    img2 = img2.reshape(img.shape)

    plt.axis('off')
    plt.imshow(img2)

    plt.savefig("2k-means.png", format="png", dpi=600)



        







    
 
def generate_rgb_centroids(k: int):
    centroids = []

    for i in range(k):
        # assign randomly generated centroids
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        centroids.append((r,g,b))
    
    return centroids

def generate_hsv_centroids(k: int):
    centroids = []

    for i in range(k):
        h = random.randint(0, 179)
        s = random.randint(0, 255)
        v = random.randint(0, 255)
        centroids.append((h,s,v))

    return centroids




# Calculates the Euclidean distance between the two based on the
# Since we are using image segmentation use the channels instead of actual distance
# Maybe make arguments as (x,y) tuple
# p1 and p2 are pixels in the image, colour is split into three channels in a list
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)






