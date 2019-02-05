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

############### Capture the frames ############
vidcap = cv2.VideoCapture('Assignment1/Videos/Run.avi')
success,image = vidcap.read()
count = 0
frames=[]
while success:
    frames.append(image)
    success,image = vidcap.read()
    count += 1
    
vidcap.release()

class GMM():
    def __init__(self,frames,K=5,std_thresh=2.5,alpha=0.005,std_init=10,T=0.5,channels=3):
        ############# Parameters in the paper #########
        self.K = K # number of gaussian components
        self.std_thresh = std_thresh # number of background components
        self.alpha = alpha # learning rate 0.005
        self.std_init = std_init # initialise std matrix
        self.T = T #0.5
        self.height = frames[0].shape[0]
        self.width = frames[0].shape[1]
        self.channels = channels
        self.frames = frames
        # self.foreground = np.zeros((height,width))
        self.W = (1/K)*np.ones((self.height, self.width, K)) # weights
        self.mu = np.random.randint(0,255,(self.height, self.width, channels, K)) # initialise pixel means [h,w,c,K]
        self.sigma = std_init * np.ones((self.height,self.width,channels,K)) # initialise pixel stds [h,w,c,K]
        # fg = np.zeros((height,width))
############ Process the frames ############
    def process(self):
        f=[]
        fg = np.zeros((self.height,self.width))
        for fr in tqdm(range(len(self.frames))):
            fg = np.zeros((self.height,self.width))
            frame = self.frames[fr]
            mu_diff = np.abs(np.expand_dims(frame,axis=3)-self.mu)
            
            for i in (range(self.height)):
                for j in range(self.width):
                    match = -1
                    for k in range(self.K):
                        if abs(np.linalg.norm(mu_diff[i,j,:,k]))<= self.std_thresh * self.sigma[i,j,:,k][0]:
                            match = k
                            # update weights
                            self.W[i,j,k] = (1-self.alpha)*self.W[i,j,k] + self.alpha  # equation 5 in paper
                            sigma_matrix = self.sigma[i,j,:,k]**2 * np.eye(3) # sigma^2.I
                            sigma_inv = np.linalg.pinv(sigma_matrix)
                            # equation 3 in paper
                            eta = (0.5/np.pi)**(3/2)*(1/np.linalg.det(sigma_matrix)**0.5)*np.exp(-0.5*np.matmul((np.matmul(frame[i,j,:]-self.mu[i,j,:,k],sigma_inv)),frame[i,j,:]-self.mu[i,j,:,k]))
                            p = self.alpha * eta # equation 8
                            self.mu[i,j,:,k] = (1-p)*self.mu[i,j,:,k] + p*(frame[i,j,:]) # equation 6 in paper
                            self.sigma[i,j,:,k] = np.sqrt((1-p) * self.sigma[i,j,:,k]**2 + p * (np.matmul(frame[i,j,:] - self.mu[i,j,:,k] , frame[i,j,:] - self.mu[i,j,:,k]))) # equation 7 in paper
                            
                        else:
                            self.W[i,j,k] = (1-self.alpha)*self.W[i,j,k]
                            
                    if match == -1:
                        min_index = np.argmin(self.W[i,j,:])
                        min_weight = np.min(self.W[i,j,:])
                        self.mu[i,j,:,min_index] = frame[i,j,:]
                        self.sigma[i,j,:,min_index] = self.std_init
                    
                    # rank the components according to W/sigma
                    rank = self.W[i,j,:]/self.sigma[i,j,0,:]
                    rank_ind =np.argsort(rank)[::-1]
                    sum_W = 0
                    bg_index = []
                    # equation 9 in paper
                    for l in range(self.K):
                        sum_W += self.W[i,j,rank_ind[l]]
                        bg_index.append(rank_ind[l])
                        if sum_W > self.T:
                            break
                    if len(bg_index)== self.K:
                        fg[i,j] = 1
                        
                    elif match in bg_index:
                        fg[i,j] = 0
                    else:
                        fg[i,j] = 1
                        
            f.append(fg)
            # out.write(np.uint8(cv2.cvtColor(fg.astype(np.float32),cv2.COLOR_GRAY2BGR)))
            # out1.write(np.uint8(cv2.cvtColor(fg.astype(np.float32),cv2.COLOR_GRAY2BGR)))
        return f
         
         
gmm = GMM(frames=frames)
f = gmm.process()
# f = process(frames)
# out.release()
# out1.release()

for i in range(len(f)):
    
    cv2.imshow('df',f[i])
    cv2.waitKey(100)
    # g = cv2.cvtColor(f[i],cv2.COLOR_GRAY2BGR)
    # out.write(np.expand_dims(g,axis=2))

############################################################
file = 'Assignment1/Dataset 1/input/'
im_list = np.sort(os.listdir(file))
images = []
for i in tqdm(range(len(im_list))):
    img = cv2.imread(os.path.join(file,im_list[i]))
    images.append(img)


