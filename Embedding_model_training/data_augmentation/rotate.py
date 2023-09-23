import numpy as np
import random
from PIL import Image,ImageFilter

class Rotate():
    def __init__(self, deg=None, p=1.0, mode='variable'):
        self.deg = deg
        self.p = p
        self.mode = mode
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        direction = [-1,1]
        if self.mode == 'variable':
            # rotate with a random degree
            deg = np.random.randint(self.deg+1)
            img = img.rotate(deg * random.choice(direction))
        elif self.mode == 'fixed':
            img = img.rotate(self.deg * random.choice(direction))
        else: # square roatation
            deg = [0, 90, 180, 270]
            img  = img.rotate(random.choice(deg))
        return img

if __name__ == '__main__':
    b = Rotate(deg=90, mode='square')
    img = Image.open('test.jpg')
    res = b(img)
    res.save('res.jpg')