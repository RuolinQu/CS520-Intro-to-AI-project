import numpy as np
import cv2
import matplotlib.pyplot as plt
from util import *

class ANN_reg:
    def __init__(self):
        self.w1 = np.random.rand(16, 25)
        self.b1 = np.random.rand(16).reshape((16, 1))
        self.w2 = np.random.rand(2, 16)
        self.b2 = np.random.rand(2).reshape((2, 1))

    def forward(self, x):
        x = self.w1.dot(x) + self.b1
        for i in x:
            if i < 0:
                i = 0
        out = self.w2.dot(x) + self.b2
        return out

    #predict fuction
    def pred(self, img_bw):

        img_pad=mean_padding(img_bw,2)
        #mean padding, extend the image boundary by 2

        img_pred=np.zeros((64,64,3))
        #creat an empty matrix for LAB color space

        img_pred[:,:,0]=img_bw
        #fill L directly from the gray map

        for i in range(64):
            for j in range(64):
                area = img_pad[i:i + 5, j:j + 5]
                x = area.reshape((25, 1))
                out=self.forward(x).reshape(-1)
                img_pred[i,j,1:]=out
        #predict A and B

        img_pred=img_pred.astype(np.uint8)
        img_out=cv2.cvtColor(img_pred,cv2.COLOR_Lab2RGB)
        #convert to RGB

        return img_out
    def train(self, trainset, epoch, lr):
        loss = []
        for e in range(epoch):
            for i in range(len(trainset)):
                print("image",i)
                loss_ = 0
                img=trainset[i]
                img_lab=cv2.cvtColor(img,cv2.COLOR_RGB2Lab)
                img_bw=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                img_pad=mean_padding(img_bw,2)
                for y in range(64):
                    for x in range(64):
                        #layer 1 , 25 to 16
                        area = img_pad[y:y+5,x:x+5]
                        x0 = area.reshape((25,1))
                        label = img_lab[y,x,1:].reshape((2,1))
                        x1 = self.w1.dot(x0) + self.b1

                        #layer 2, 16 to 2
                        for a in x1:
                            if a < 0:
                                a = 0
                        out = self.w2.dot(x1) + self.b2

                        # loss and backpropagation
                        loss_ += np.linalg.norm(out-label,2)
                        dout = out - label
                        db2 = dout
                        db2=gradient_clip(db2,1)
                        #graadient_clip, to avoid Gradient Exploding or Gradient Vanishing

                        dw2 = dout.dot(x1.T)
                        dw2 = gradient_clip(dw2, 1)
                        dx1 = self.w2.T.dot(dout)
                        dx1 = gradient_clip(dx1, 1)
                        for i in range(16):
                            if x1[i] == 0:
                                dx1[i] = 0
                        dw1 = dx1.dot(x0.T)
                        dw1 = gradient_clip(dw1, 1)
                        db1 = np.sum(dx1)
                        db1 = gradient_clip(db1, 1)
                        self.w2 -= lr * dw2
                        self.b2 -= lr * db2
                        self.w1 -= lr * dw1
                        self.b1 -= lr * db1
                print("loss=",loss_)
                loss.append(loss_)

        plt.plot(loss)
        plt.show()
        return loss
