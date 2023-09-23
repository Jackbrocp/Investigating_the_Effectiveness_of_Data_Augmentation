import numpy as np
import random
from PIL import Image, ImageOps

class Posterize():
    def __init__(self, v=1, p=1.0):
        self.v = v
        self.p = p
    def __call__(self,img):
        if random.uniform(0, 1) > self.p:
            return img
        else:
            return  ImageOps.posterize(img, self.v)

if __name__ == '__main__':
    b = Posterize(v=2)
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')
