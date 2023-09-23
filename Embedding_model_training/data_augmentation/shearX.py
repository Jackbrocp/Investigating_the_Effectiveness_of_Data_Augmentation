import numpy as np
import random
from PIL import Image,ImageFilter, ImageChops

class ShearX():
    def __init__(self,p=1.0, off=0.1, mode='variable'):
        self.p = p
        self.mode = mode
        self.off = off
    def __call__(self,img):
        if random.uniform(0, 1) > self.p:
            return img
        direction = [-1,1]
        if self.mode == 'variable':
            off = int(img.size[1]*self.off)
            img = ImageChops.offset(img, xoffset=np.random.randint(off)*random.choice(direction), yoffset=0)
        elif self.mode == 'fixed':
            off = int(img.size[1]*self.off)
            img = ImageChops.offset(img, off*random.choice(direction), 0)
        return img

if __name__ == '__main__':
    b =  ShearX(off=0.2, mode='fixed')
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')