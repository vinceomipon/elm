import cv2
import numpy as np
import random


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.models import *

class k_means_calc:
    # only pass centroids for argument purposes, to verify that implementation works
    def __init__(self, k: int, img: ELMImage, centroids=None):
        self.k = k
        self.img = img
        if centroids is None:
            self.centroids = []
        else:
            self.centroids = centroids
        
        

    

    # Helper function which assigns all pixels in the image to a specific centroid
    def assign_pixel_to_centroid(self, pixel_length: int):
        r_flat, g_flat, b_flat = self.img.flatten_channels(self.img.rgb_arr)
        assignment = [0]*pixel_length

        for i in range(pixel_length):
            assignment[i] = self.calc_min_distance(r_flat, g_flat, b_flat, i)

        return assignment
    
    
        

    # Helper function to find the smallest distance between k-centroids
    def calc_min_distance(self, r_flat, g_flat, b_flat, i: int):
        distances = [0]*self.k

        # for each k mean, calculate the distance
        for j in range(self.k):
            distances[j] = self.distance(self.centroids[j], [r_flat[i], g_flat[i], b_flat[i]])

        # return the indice with the smallest distance (indice implies kth-centroid)
        return np.argmin(distances)




    def k_means(self, select: int):



        r, g, b = self.img.flatten_channels(self.img.rgb_arr)
        
        # idea for centroid key is the # of the centroid, value coresponds to the coordiante
        if select == 1:
            self.generate_rgb_centroids()
        elif select == 0:
            self.generate_hsv_centroids()
        
        pixels = len(r)

        # exit conditions
        # i.e. exit when max iterations reached on the centroid stops moving
        moved = True

        while moved:

            # initialize an assignment array of size pixels
            # each indice represents a pixel and contains the closest distance to a centroid
            assignment = self.assign_pixel_to_centroid(pixels)

            # use prev_centroid to compare with curr centroid to see if centroid have moved or not
            prev_centroids = self.centroids.copy()
            
            # calculate the mean for each cluster
            for i in range(self.k):

                # get a list of pixel indices that correspond to the i-th k-mean
                ind = [j for j in range(pixels) if assignment[j] == i]

                # calculate the new mean for each cluster
                # i.e. we find the mean for each channel by averaging all the relevant pixels' channels
                if len(ind) != 0:
                    self.centroids[i][0] = np.mean(r[ind])
                    self.centroids[i][1] = np.mean(g[ind])
                    self.centroids[i][2] = np.mean(b[ind])
                
                # if cluster empty then must mean that color is not part of image
                else:
                    # pick another random pixel index for new centroid
                    random_idx = np.random.choice(pixels)
                    self.centroids[i][0] = r[random_idx]
                    self.centroids[i][1] = g[random_idx]
                    self.centroids[i][2] = b[random_idx]

            # check if the centroids have moved after each iteration
            if np.array_equal(self.centroids, prev_centroids):
                moved =  False
        
        r_copy = np.array(r.copy())
        g_copy = np.array(g.copy())
        b_copy = np.array(b.copy())

        # update pixels to match their corresponding cluster
        for i in range(self.k):
            ind = [j for j in range(pixels) if assignment[j] == i]

            r_copy[ind] = self.centroids[i][0]
            g_copy[ind] = self.centroids[i][1]
            b_copy[ind] = self.centroids[i][2]
        
        img2 = np.array([r_copy, g_copy, b_copy])

        img2 = img2.transpose()

        img2 = img2.reshape(self.img.bgr_arr.shape)

        plt.axis('off')
        plt.imshow(img2)

        plt.savefig("data/output/two_means/2k-means.png", format="png", dpi=600)
    


    def generate_rgb_centroids(self):

        for i in range(self.k):
            # assign randomly generated centroids
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.centroids.append([r,g,b])

    def generate_hsv_centroids(self):

        for i in range(self.k):
            h = random.randint(0, 179)
            s = random.randint(0, 255)
            v = random.randint(0, 255)
            self.centroids.append([h,s,v])

    # Calculates the Euclidean distance between the two based on the
    # Since we are using image segmentation use the channels instead of actual distance
    # Maybe make arguments as (x,y) tuple
    # p1 and p2 are pixels in the image, colour is split into three channels in a list
    def distance(self, p1, p2):
        a1 = np.array(p1)
        a2 = np.array(p2)
        return np.linalg.norm(a1 - a2)


    






# Implement these later maybe

    
