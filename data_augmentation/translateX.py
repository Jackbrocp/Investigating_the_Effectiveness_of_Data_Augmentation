import numpy as np
import random
import PIL
from PIL import Image

class TranslateX():
    def __init__(self, v= 10, p=1.0):
        self.v = v
        self.p = p
    def __call__(self, img):
        # self.v = self.v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, self.v, 0, 1, 0))

if __name__ == '__main__':
    b = TranslateX(v=0.2)
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')