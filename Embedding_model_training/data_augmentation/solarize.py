import numpy as np
import random
from PIL import Image, ImageOps

class Solarize():
    def __init__(self, v=100, p=1.0):
        self.v = v
        self.p = p
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        else:
            return ImageOps.solarize(img, self.v)
        
if __name__ == '__main__':
    b = Solarize(v=100)
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')