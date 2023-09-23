import numpy as np
import random
from PIL import Image, ImageOps

class Invert():
    def __init__(self, p=1.0):
        self.p = p
    def __call__(self,img):
        if random.uniform(0, 1) < self.p:
            return img
        else:
            return ImageOps.invert(img)

if __name__ == '__main__':
    b = Invert()
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')