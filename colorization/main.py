import numpy as np
import cv2
from ANN import *
import matplotlib.pyplot as plt
from PIL import Image
import glob
from util import *
import random

image = []
for filename in glob.glob('Dataset/beach/*.jpg'):
    im = Image.open(filename)
    im_array = np.array(im)
    im_resize = cv2.resize(im_array, (64,64))
    image.append(im_resize)
image = np.array(image)

more_image=[]
for filename in glob.glob('Dataset/sunrise/*.jpeg'):
    im = Image.open(filename)
    im_array = np.array(im)
    im_resize = cv2.resize(im_array, (64,64))
    more_image.append(im_resize)
for filename in glob.glob('Dataset/forest/*.jpeg'):
    im = Image.open(filename)
    im_array = np.array(im)
    im_resize = cv2.resize(im_array, (64,64))
    more_image.append(im_resize)
more_image = np.array(more_image)

more_image=np.concatenate((image,more_image))
random.shuffle(more_image)


n, h, w, c = image.shape

image_bw = np.zeros((n, h, w))
for i in range(n):
    image_bw[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2GRAY)
test=image_bw[3]
test2=cv2.cvtColor(more_image[3],cv2.COLOR_RGB2GRAY)
test3=cv2.cvtColor(more_image[6],cv2.COLOR_RGB2GRAY)

print(np.shape(more_image))
model2=ANN_reg()
loss_ANN=model2.train(more_image,epoch=1,lr=0.01)
img_out2=model2.pred(test2)
img_out3=model2.pred(test3)
fig,a=plt.subplots(1,4)
a[0].imshow(more_image[3])
a[1].imshow(img_out2)
a[2].imshow(more_image[6])
a[3].imshow(img_out3)
plt.show()