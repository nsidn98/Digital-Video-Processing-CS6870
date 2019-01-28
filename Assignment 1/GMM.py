import numpy as np
import cv2
import matplotlib.pyplot as pyplot

vidcap = cv2.VideoCapture('Videos/Run.avi')
success,image = vidcap.read()
count = 0
frames=[]
while success:
    frames.append(image)
    success,image = vidcap.read()
    count += 1


  
class GMM():
    def __init__(self,frames,height,width,gaussian_no=3,background_comp=3,std_thresh=2.5,alpha=0.01,thresh=0.25,std_init=10):
        self.gaussian_no = gaussian_no # typically 3-5 as mentioned in paper
        self.background_comp = background_comp # number of background components
        self.std_thresh = std_thresh # standard deviation threshold:2.5 as mentioned in paper
        self.alpha = alpha # learning rate (between 0 and 1) (from paper 0.01)
        self.thresh = thresh # foreground threshold (0.25 or 0.75 in the paper)
        self.std_init = 10 # initial standard deviation for new components
        self.height = height
        self.width = width
        self.W = (1/self.gaussian_no) * np.ones((self.height,self.width,self.gaussian_no)) # initialise the weight array
        self.mean = np.random.randint(0,255,(self.height,self.width,self.gaussian_no)) # initialise pixel means
        self.std = self.std_init * np.ones((self.height,self.width,self.gaussian_no)) # initialise pixel standard deviations
        #self.u_diff = np.zeros((self.height,self.width,self.gaussian_no)) #difference of each pixel from mean
        self.background = np.zeros((self.height,self.width))
        self.frames = frames
        
    def process(self):
        fg_arr=[]
        # frames is array [height,width,channels,frames]
        fg = np.zeros((self.height, self.width));
        
        frame_no = 0
        for frame in self.frames:
            
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # r_frame = frame[:,:,0]
            # g_frame = frame[:,:,1]
            # b_frame = frame[:,:,2]
            #
            
            u_diff = np.abs(np.expand_dims(frame,axis=2)-self.mean)
            # u_diff_g = np.abs(g_frame-self.mean)
            # u_diff_b = np.abs(b_frame-self.mean)
            
            # update gaussian for each pixel
            for h in range(self.height):
                for w in range(self.width):
                    match = 0
                    for g in range(self.gaussian_no):
                        if np.abs(u_diff[h,w,g]) <= self.background_comp * self.std[h,w,g]:
                            match = 1 # flag for indicating signal matches
                            # update weights, mean, std, p
                            self.W[h,w,g] = (1-self.alpha)*self.W[h,w,g] + self.alpha  # equation 5 in paper
                            np.dot((),np.linalg.inv(self.std))
                            p = self.alpha * (np.exp(-0.5*()))
                            self.mean[h,w,g] = (1-self.p)*self.mean[h,w,g] + self.p*frame[h,w] # equation 6 in paper
                            self.std[h,w,g] = np.sqrt((1-self.p) * self.std[h,w,g]**2 + self.p * (frame[h,w] - self.mean[h,w,g])**2) # equation 7 in paper
                        else : # pixel does not match
                            self.W[h,w,g] = (1-self.alpha) * self.W[h,w,g]
                            
                    self.W[h,w,:] = self.W[h,w,:]/np.sum(self.W[h,w,:])
                    
                    self.background[h,w] = 0
                    for g in range(self.gaussian_no):
                        self.background = self.background + self.mean[h,w,g] * self.W[h,w,g]
                        
                    # if no components match them new component
                    if match == 0:
                        min_index = np.argmin(self.W[h,w,:])
                        min_weight = np.min(self.W[h,w,:])
                        self.mean[h,w,min_index] = frame[h,w]
                        self.std[h,w,min_index] = self.std_init
                    
                    rank = self.W[h,w,:]/self.std[h,w,:] # calculate the rank of components
                    rank_ind = np.linspace(0,self.gaussian_no-1,self.gaussian_no)
                    for k in range(1,len(rank_ind)):
                        for m in range(k-1):
                            if rank[k]>rank[m]:
                                rank_temp = rank[m]
                                rank[m] = rank[k]
                                rank[k] = rank_temp
                                
                                rank_ind_temp = rank_ind[m]
                                rank_ind[m] = rank_ind[k]
                                rank_ind[k] = rank_ind_temp
                    
                    match = 0
                    k = 0
                    fg[h,w] = 0
                    while ((match == 0)&(k<self.background_comp)):
                        if float(self.W[h,w,int(rank_ind[k])]) >= float(self.thresh):
                            if float(np.abs(u_diff[h,w,int(rank_ind[k])])) <= float(self.std_thresh*self.std[h,w,int(rank_ind[k])]):
                                fg[h,w] = 0
                                
                                match = 1
                            else:
                                fg[h,w] = frame[h,w]
                                
                        k+=1
                    
            fg_arr.append(fg)
            frame_no+=1
            print(frame_no)
        return fg_arr
gmm = GMM(frames,144,180)
f = gmm.process()
            
                
