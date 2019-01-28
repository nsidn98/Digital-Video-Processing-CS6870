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
W = (1/K)*np.ones((height, width, K)) # weights
mu = np.random.randint(0,255,(height, width, 3, K)) # initialise pixel means [h,w,c,K]
sigma = 10 * np.ones(K)

def process(frames):
    for frame in frames:
        mu_diff = np.abs(np.expand_dims(frame,axis=3)-mu)
        
        for i in range(height):
            for j in range(width):
                match = 0
                for k in range(K):
                    if abs():
    



    