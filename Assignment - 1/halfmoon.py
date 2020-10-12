import numpy as np

#Generate halfmoon dataset
def halfmoon(rad, width, d, n_samp):

    if rad < (width // 2):
        return print('The radius should be at least larger than half the width')
     
    if n_samp % 2 != 0:  
        return print('Please make sure the number of samples is even') 
    
    data = np.zeros((3,n_samp))
      
    aa = np.random.random_sample((2,n_samp//2))  
    radius = (rad-width//2) + width*aa[0,:] 
    theta = np.pi*aa[1,:]        
      
    x     = radius*np.cos(theta)  
    y     = radius*np.sin(theta)  
    label = np.ones((1,len(x)))         # label for Class 1  
      
    x1    = radius*np.cos(-theta) + rad  
    y1    = radius*np.sin(-theta) - d  
    label1= -1*np.ones((1,len(x1)))     # label for Class 2  
     
    data[0,:]=np.concatenate([x,x1])
    data[1,:]=np.concatenate([y,y1])
    data[2,:]=np.concatenate([label,label1],axis=1)

    data = np.transpose(data)
    
    return data