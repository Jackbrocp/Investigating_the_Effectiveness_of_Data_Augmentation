import numpy as np
import random
from PIL import Image,ImageEnhance

class Contrast():
    def __init__(self, mag=0.1, p=1.0):
        self.mag = mag
        self.p = p
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        else:
            return  ImageEnhance.Contrast(img).enhance(self.mag)


if __name__ == '__main__':
    b = Contrast(mag=0.5)
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')