import numpy as np
import random
from PIL import Image,ImageFilter


class Blur():
    '''
    Add blur into the images.
    Apply transformation with probability p.
    '''
    def __init__(self, p=1.0, blur='normal'):
        self.p = p 
        self.blur = blur
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        else:
            if self.blur == 'normal':
                img = img.filter(ImageFilter.BLUR)
            elif self.blur == 'Gaussian':
                img = img.filter(ImageFilter.GaussianBlur)
            elif self.blur == 'mean':
                img = img.filter(ImageFilter.BoxBlur)
            return img

if __name__ == '__main__':
    b = Blur()
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')
