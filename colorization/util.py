import numpy as np
import cv2

def mean_padding(image,n):
    mean=np.mean(image)
    h,w=np.shape(image)
    image_pad=np.zeros((h+2*n,w+2*n))
    image_pad.fill(mean)
    image_pad[n:n+h,n:n+w]=image
    return image_pad

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

def gradient_clip(w,threshold=1):
    if np.linalg.norm(w)>1:
        w_clipped=threshold/np.linalg.norm(w)*w
        return w_clipped
    else:
        return w