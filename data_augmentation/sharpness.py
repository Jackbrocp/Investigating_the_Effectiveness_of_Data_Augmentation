import numpy as np
import random
from PIL import Image, ImageEnhance

class Sharpness():
    def __init__(self,p=1.0,mag=0.9):
        self.p = p
        self.mag = mag
    def __call__(self,img):
        if random.uniform(0, 1) > self.p:
            return img
        else:
            return ImageEnhance.Sharpness(img).enhance(1+self.mag)

if __name__ == '__main__':
    b = Sharpness()
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')