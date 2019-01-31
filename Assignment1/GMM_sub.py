'''
-----------------------------------------
Implementation of
Adaptive Background Mixture Models for Real-Time Tracking
by
Chris Stauffer & W.E.L. Grimson

Background Subtraction using Gaussian Mixture Models
~~~
Author: Siddharth Nayak
~~~
-----------------------------------------
'''


import numpy as np
import cv2
import matplotlib.pyplot as pyplot

K = 5 # number of gaussian components
std_thresh = 2.5 # number of background components
alpha = 0.01 # learning rate
std_init = 10 # initialise std matrix
height = 144
width = 180
channels = 3
W = (1/K)*np.ones((height, width, K)) # weights
mu = np.random.randint(0,255,(height, width, channels, K)) # initialise pixel means [h,w,c,K]
sigma = std_init * np.ones((height,width,channels,K)) # initialise pixel stds [h,w,c,K]

def process(frames):
    for frame in frames:
        mu_diff = np.abs(np.expand_dims(frame,axis=3)-mu)
        
        for i in range(height):
            for j in range(width):
                match = 0
                for k in range(K):
                    if abs(np.linalg.norm(mu_diff[i,j,:,k])<= std_thresh * sigma[i,j,:,k][0]):
                        match = 1
                        # update weights
                        W[i,j,k] = (1-alpha)*W[i,j,k] + alpha  # equation 5 in paper
                        sigma_matrix = sigma[i,j,:,k]**2 * np.eye(3) # sigma^2.I
                        sigma_inv = np.linalg.pinv(sigma_matrix)
                        # equation 3 in paper
                        eta = (0.5/np.pi)**(K/2)*(1/np.linalg.det(sigma_matrix))*np.exp(-0.5*np.dot((np.dot(frame[i,j,:]-mu[i,j,:,k],sigma_inv)),frame[i,j,:]-mu[i,j,:,k]))
                        p = alpha * eta # equation 8
                        mu[i,j,k] = (1-p)*mu[i,j,k] + p*frame[h,w,:] # equation 6 in paper
                        sigma[i,j,:,k] = np.sqrt((1-p) * sigma[i,j,k]**2 + p * (np.dot(frame[i,j,:] - mean[i,j,:,k] , frame[i,j,:] - mean[i,j,:,k])) # equation 7 in paper
                    else:
                        W[i,j,k] = (1-alpha)*W[i,j,k]
                        
                if match = 0:
                    min_index = np.argmin(W[i,j,:])
                    min_weight = np.min(W[i,j,:])
                    mu[h,w,:,min_index] = frame[i,j,:]
                    sigma[i,j,:,min_index] = std_init



    