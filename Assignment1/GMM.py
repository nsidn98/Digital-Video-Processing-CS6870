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
from tqdm import tqdm

K = 5 # number of gaussian components
std_thresh = 2.5 # number of background components
alpha = 0.05 # learning rate
std_init = 10 # initialise std matrix
height = 144
width = 180
channels = 3
foreground = np.zeros((height,width))
W = (1/K)*np.ones((height, width, K)) # weights
mu = np.random.randint(0,255,(height, width, channels, K)) # initialise pixel means [h,w,c,K]
sigma = std_init * np.ones((height,width,channels,K)) # initialise pixel stds [h,w,c,K]
T = 0.7
fg = np.zeros((height,width))

vidcap = cv2.VideoCapture('Assignment1/Videos/Run.avi')
success,image = vidcap.read()
count = 0
frames=[]
while success:
    frames.append(image)
    success,image = vidcap.read()
    count += 1
    
vidcap.release()

def process(frames):
    count = 0
    
    f=[]
    for fr in tqdm(range(len(frames))):
        fg = np.zeros((height,width))
        frame = frames[fr]
        mu_diff = np.abs(np.expand_dims(frame,axis=3)-mu)
        
        for i in (range(height)):
            for j in range(width):
                match = -1
                for k in range(K):
                    if abs(np.linalg.norm(mu_diff[i,j,:,k]))<= std_thresh * sigma[i,j,:,k][0]:
                        match = k
                        # update weights
                        W[i,j,k] = (1-alpha)*W[i,j,k] + alpha  # equation 5 in paper
                        sigma_matrix = sigma[i,j,:,k]**2 * np.eye(3) # sigma^2.I
                        sigma_inv = np.linalg.pinv(sigma_matrix)
                        # equation 3 in paper
                        eta = (0.5/np.pi)**(3/2)*(1/np.linalg.det(sigma_matrix)**0.5)*np.exp(-0.5*np.matmul((np.matmul(frame[i,j,:]-mu[i,j,:,k],sigma_inv)),frame[i,j,:]-mu[i,j,:,k]))
                        p = alpha * eta # equation 8
                        mu[i,j,:,k] = (1-p)*mu[i,j,:,k] + p*(frame[i,j,:]) # equation 6 in paper
                        sigma[i,j,:,k] = np.sqrt((1-p) * sigma[i,j,:,k]**2 + p * (np.matmul(frame[i,j,:] - mu[i,j,:,k] , frame[i,j,:] - mu[i,j,:,k]))) # equation 7 in paper
                        
                    else:
                        W[i,j,k] = (1-alpha)*W[i,j,k]
                        
                if match == -1:
                    min_index = np.argmin(W[i,j,:])
                    min_weight = np.min(W[i,j,:])
                    mu[i,j,:,min_index] = frame[i,j,:]
                    sigma[i,j,:,min_index] = std_init
                
                # rank the components according to W/sigma
                rank = W[i,j,:]/sigma[i,j,0,:]
                rank_ind =np.argsort(rank)[::-1]
                sum_W = 0
                bg_index = []
                # equation 9 in paper
                for l in range(K):
                    sum_W += W[i,j,rank_ind[l]]
                    bg_index.append(rank_ind[l])
                    if sum_W > T:
                        break
                if len(bg_index)== K:
                    fg[i,j] = 1
                    
                elif match in bg_index:
                    fg[i,j] = 0
                else:
                    fg[i,j] = 1
        count += 1
        f.append(fg)
    return f
    
f = process(frames)


for i in range(len(f)):
    
    cv2.imshow('df',f[i])
    cv2.waitKey(40)
    # g = cv2.cvtColor(f[i],cv2.COLOR_GRAY2BGR)
    # out.write(np.expand_dims(g,axis=2))



# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width,height))

for i in range(len(f)):
    
    g = cv2.cvtColor(f[i].astype(np.float32),cv2.COLOR_GRAY2BGR)
        # write the flipped frame
    out.write(np.uint8(g))

    

# Release everything if job is finished
out.release()
