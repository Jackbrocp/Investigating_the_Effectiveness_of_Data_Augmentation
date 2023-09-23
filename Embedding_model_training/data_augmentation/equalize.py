import numpy as np
import random
from PIL import Image,ImageFilter,ImageOps


class Equalize():
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        else:
            img = ImageOps.equalize(img, mask=None)
            return img

if __name__ == '__main__':
    b = Equalize()
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')