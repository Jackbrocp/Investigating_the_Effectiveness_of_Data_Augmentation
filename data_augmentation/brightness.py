import numpy as np
import random
from PIL import Image,ImageEnhance

class Brightness():
    def __init__(self, mag=0.1, p=1.0):
        self.mag = mag
        self.p = p
        # self.num = 0
    def __call__(self, img):
        # p = random.uniform(0, 1)
        if  random.uniform(0, 1) > self.p:
            return img
        else:
            # self.num += 1
            # print('NUM:',p,self.p,self.num)
            return  ImageEnhance.Brightness(img).enhance(self.mag)
         
if __name__ == '__main__':
    b = Brightness(mag=0.5)
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')